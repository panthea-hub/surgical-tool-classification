# Repository Structure

## Original Repository Structure

```text
surgical-tool-classification/
│
├── configs/
│   ├── default.yaml
│   └── best_config_v3.yaml
│
├── checkpoints/
├── data/
│
├── docs/
│   ├── change_log.md
│   ├── project_architecture.md
│   ├── project_plan.md
│   └── repository_structure.md
│
├── experiments/
│   └── exp02_efficientnet/
│       ├── model_def.py
│       ├── run_experiment.py
│       └── results.json
│
├── helpers/
│   ├── __init__.py
│   ├── image_helpers.py
│   └── metrics.py
│
├── legacy/
│   ├── cnn_baseline_v2.py
│   └── old_train.py
│
├── notebooks/
│   └── EDA_final.ipynb
│
├── splits/
│   └── split_seed42.json
│
├── README.md
├── config.py
├── contract.py
├── check_submission.py
├── data_utils.py
├── evaluate_model.py
├── labels.csv
├── predict.py
├── run_all.sh
├── stats.json
├── train.py
├── train_v2.py
├── utils.py
└── .gitignore
```

## Current Repository Structure

```text
surgical-tool-classification/
├── archive/
│   ├── configs/
│   ├── experiments/
│   ├── helpers/
│   ├── legacy/
│   ├── notebooks/
│   ├── splits/
│   ├── data_utils.py
│   ├── labels.csv
│   └── utils.py
├── checkpoints/
├── docs/
│   ├── model_baseline.md
│   ├── project_plan.md
│   └── repository_structure.md
├── runs/
├── config.py
├── train.py
├── train_v2.py
├── evaluate_model.py
├── predict.py
├── check_submission.py
├── contract.py
├── run_all.sh
├── setup.sh
├── requirements.txt
├── stats.json
├── README.md
└── .gitignore
```
