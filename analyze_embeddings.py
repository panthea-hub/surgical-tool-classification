"""Plot validation embeddings from the input to ResNet18's final fc layer.

Usage: python analyze_embeddings.py --checkpoint checkpoints/<checkpoint>.pt
"""
import argparse
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import plotly.graph_objects as go
import torch
from torch.utils.data import DataLoader
from umap import UMAP

import config
from evaluate_model import EvalDataset
from train import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_names = checkpoint["class_names"]
    dataset = EvalDataset(
        os.path.join(config.DATA_ROOT, "validation"),
        class_names,
        checkpoint["image_size"],
        checkpoint["normalization_mean"],
        checkpoint["normalization_std"],
    )
    # Stable ordering makes repeated UMAP runs comparable.
    dataset.samples.sort(key=lambda sample: sample[0])
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    model = build_model(num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    feature_batches = []

    def capture_features(module, inputs):
        feature_batches.append(inputs[0].detach().cpu().clone())

    hook = model.fc.register_forward_pre_hook(capture_features)
    filenames, true_labels, predicted_labels = [], [], []
    try:
        with torch.no_grad():
            for images, labels, paths in loader:
                predictions = model(images.to(device)).argmax(dim=1)
                filenames.extend(Path(path).name for path in paths)
                true_labels.extend(labels.tolist())
                predicted_labels.extend(predictions.cpu().tolist())
    finally:
        hook.remove()

    embeddings = torch.cat(feature_batches).numpy()
    if embeddings.shape != (len(dataset), 512):
        raise ValueError(f"Expected one 512-dimensional embedding per image, got {embeddings.shape}")
    if len(dataset) < 4:
        raise ValueError("UMAP visualization requires at least four validation images.")
    coordinates = UMAP(
        n_components=2, n_neighbors=min(15, len(dataset) - 1),
        min_dist=0.1, metric="euclidean", random_state=42, n_jobs=1,
    ).fit_transform(embeddings)
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    correct = true_labels == predicted_labels

    output_dir = Path("runs/embedding_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "embedding_umap.png"
    csv_path = output_dir / "embedding_umap.csv"
    html_path = output_dir / "embedding_umap.html"
    with csv_path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow([
            "filename", "true_class", "predicted_class",
            "embedding_x", "embedding_y", "correct",
        ])
        for filename, actual, predicted, xy, is_correct in zip(
            filenames, true_labels, predicted_labels, coordinates, correct
        ):
            writer.writerow([
                filename, class_names[actual], class_names[predicted],
                float(xy[0]), float(xy[1]), bool(is_correct),
            ])

    colors = {"clipper": "#0072B2", "grasper": "#009E73", "hook": "#E69F00", "scissor": "#CC79A7"}
    fig, ax = plt.subplots(figsize=(10, 7))
    for index, name in enumerate(class_names):
        for is_correct, marker in [(True, "o"), (False, "X")]:
            mask = (true_labels == index) & (correct == is_correct)
            ax.scatter(
                coordinates[mask, 0], coordinates[mask, 1],
                color=colors[name], marker=marker,
                s=40 if is_correct else 100, alpha=0.8 if is_correct else 1.0,
                edgecolors="none" if is_correct else "black", linewidths=0.8,
                zorder=2 if is_correct else 3,
            )
    legend = [
        Line2D([], [], marker="o", linestyle="none", color=colors[name], label=name)
        for name in class_names
    ]
    legend.extend([
        Line2D([], [], marker="o", linestyle="none", color="gray", label="Correct"),
        Line2D([], [], marker="X", linestyle="none", color="gray",
               markeredgecolor="black", markersize=9, label="Misclassified"),
    ])
    ax.legend(handles=legend, title="True class / prediction", loc="best")
    ax.set(title="ResNet18 validation embeddings — UMAP", xlabel="UMAP 1", ylabel="UMAP 2")
    fig.text(0.5, 0.02, "2D projection of 512D features; visual overlap alone does not prove class inseparability.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    interactive_fig = go.Figure()
    for index, name in enumerate(class_names):
        for is_correct, symbol, status in [
            (True, "circle", "Correct"),
            (False, "x", "Misclassified"),
        ]:
            mask = (true_labels == index) & (correct == is_correct)
            point_indices = np.flatnonzero(mask)
            interactive_fig.add_trace(go.Scatter(
                x=coordinates[mask, 0],
                y=coordinates[mask, 1],
                mode="markers",
                name=f"{name} — {status}",
                marker={
                    "color": colors[name],
                    "symbol": symbol,
                    "size": 8 if is_correct else 12,
                    "line": {"color": "black", "width": 1 if not is_correct else 0},
                },
                customdata=[
                    [
                        filenames[i],
                        class_names[true_labels[i]],
                        class_names[predicted_labels[i]],
                        status,
                    ]
                    for i in point_indices
                ],
                hovertemplate=(
                    "filename: %{customdata[0]}<br>"
                    "true class: %{customdata[1]}<br>"
                    "predicted class: %{customdata[2]}<br>"
                    "status: %{customdata[3]}<br>"
                    "UMAP 1: %{x:.4f}<br>"
                    "UMAP 2: %{y:.4f}<extra></extra>"
                ),
            ))
    interactive_fig.update_layout(
        title="ResNet18 validation embeddings — UMAP",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        legend_title="True class / prediction",
    )
    interactive_fig.write_html(html_path, include_plotlyjs=True)

    print(f"images analyzed: {len(dataset)}")
    print(f"misclassified images: {int((~correct).sum())}")
    print(f"output plot: {plot_path}")
    print(f"output CSV: {csv_path}")
    print(f"interactive plot: {html_path}")


if __name__ == "__main__":
    main()
