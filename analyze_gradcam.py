"""Generate predicted-class Grad-CAM for one image or an error-analysis directory.

Usage: python analyze_gradcam.py --checkpoint checkpoints/<checkpoint>.pt --image <path>
       python analyze_gradcam.py --checkpoint checkpoints/<checkpoint>.pt --error-analysis-dir runs/error_analysis
"""
import argparse
from pathlib import Path
import shutil

from matplotlib import cm
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from train import build_model


def preprocess(image, checkpoint):
    """Match predict.py's RGB resize, scaling, and checkpoint normalization."""
    size = checkpoint["image_size"]
    image = image.convert("RGB").resize((size, size))
    tensor = torch.from_numpy(np.array(image, dtype=np.float32)).permute(2, 0, 1) / 255.0
    mean = torch.tensor(checkpoint["normalization_mean"]).view(3, 1, 1)
    std = torch.tensor(checkpoint["normalization_std"]).view(3, 1, 1)
    return (tensor - mean) / std


def gradcam(model, tensor, original_size):
    """Return the normalized spatial map and unchanged classification logits."""
    captured = {}

    def capture_gradient(gradient):
        captured["gradients"] = gradient.detach()

    def capture_activation(module, inputs, output):
        captured["activations"] = output.detach()
        captured["gradient_hook"] = output.register_hook(capture_gradient)

    hook = model.layer4[-1].register_forward_hook(capture_activation)
    try:
        # build_model freezes the backbone; input gradients keep its graph alive.
        tensor = tensor.detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logits = model(tensor)
        predicted = logits.argmax(dim=1).item()
        logits[0, predicted].backward()
        weights = captured["gradients"].mean(dim=(2, 3), keepdim=True)
        heatmap = (weights * captured["activations"]).sum(dim=1, keepdim=True).relu()
        heatmap -= heatmap.min()
        heatmap /= heatmap.max().clamp_min(1e-8)
        heatmap = F.interpolate(
            heatmap, size=(original_size[1], original_size[0]),
            mode="bilinear", align_corners=False,
        )
        return heatmap[0, 0].cpu().numpy(), logits.detach()
    finally:
        hook.remove()
        if "gradient_hook" in captured:
            captured["gradient_hook"].remove()
        model.zero_grad(set_to_none=True)


def analyze_image(model, checkpoint, device, image_path, output_dir, output_name=None):
    """Run the existing single-image analysis and return its predicted class."""
    class_names = checkpoint["class_names"]
    with Image.open(image_path) as source:
        original = source.convert("RGB")
    tensor = preprocess(original, checkpoint).unsqueeze(0).to(device)
    heatmap, logits = gradcam(model, tensor, original.size)
    predicted = logits.argmax(dim=1).item()
    confidence = logits.softmax(dim=1)[0, predicted].item()

    colored = Image.fromarray((cm.get_cmap("inferno")(heatmap)[..., :3] * 255).astype(np.uint8))
    overlay = Image.blend(original, colored, alpha=0.4)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = image_path.stem if output_name is None else output_name
    outputs = {
        "original": output_dir / f"{output_name}_original.png",
        "heatmap": output_dir / f"{output_name}_heatmap.png",
        "overlay": output_dir / f"{output_name}_overlay.png",
    }
    # Refuse an output collision or symlink that would overwrite the source.
    if any(path.resolve() == image_path.resolve() for path in outputs.values()):
        raise ValueError("An output path points to the source image.")
    for name, image in [("original", original), ("heatmap", colored), ("overlay", overlay)]:
        image.save(outputs[name])

    print(f"image: {image_path.name}")
    if image_path.parent.name in class_names:
        print(f"true class (from parent directory): {image_path.parent.name}")
    print(f"predicted class: {class_names[predicted]}")
    print(f"prediction confidence: {confidence:.2%}")
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return class_names[predicted]


def analyze_errors(model, checkpoint, device, error_dir, output_dir):
    """Read only the missed-class view; validate its labels and deduplicate files."""
    if not error_dir.is_dir():
        raise FileNotFoundError(f"Error-analysis directory does not exist: {error_dir}")
    images = {}
    for path in sorted(error_dir.glob("missed_*/predicted_*/*")):
        if not path.is_file() or path.suffix.lower() != ".png":
            continue
        actual = path.parent.parent.name[len("missed_"):]
        expected = path.parent.name[len("predicted_"):]
        if actual not in checkpoint["class_names"] or expected not in checkpoint["class_names"]:
            raise ValueError(f"Unknown class in error-analysis path: {path}")
        key = (actual, path.name)
        if key in images:
            raise ValueError(f"Duplicate image with ambiguous prediction folders: {path}")
        images[key] = (path, expected)

    if not images:
        raise ValueError(f"No valid misclassified PNG images found in: {error_dir}")

    for batch_dir in output_dir.glob("actual_*"):
        if batch_dir.is_dir():
            shutil.rmtree(batch_dir)

    matched = 0
    for (actual, filename), (path, expected) in images.items():
        destination = output_dir / f"actual_{actual}" / f"predicted_{expected}"
        # Keep the full filename (including extension) to avoid stem collisions.
        predicted = analyze_image(model, checkpoint, device, path, destination, filename)
        if predicted == expected:
            matched += 1
        else:
            print(f"WARNING: {path}: folder predicts {expected}, model predicts {predicted}")

    print(f"images analyzed: {len(images)}")
    print(f"correctly reproduced predictions: {matched}")
    print(f"prediction mismatches: {len(images) - matched}")
    print(f"output directory: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image")
    inputs.add_argument("--error-analysis-dir")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_names = checkpoint["class_names"]
    model = build_model(num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    output_dir = Path("runs/gradcam_analysis")
    if args.image:
        analyze_image(model, checkpoint, device, Path(args.image), output_dir)
    else:
        analyze_errors(model, checkpoint, device, Path(args.error_analysis_dir), output_dir)


if __name__ == "__main__":
    main()
