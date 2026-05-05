from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    model_name: str = "dccuchile/bert-base-spanish-wwm-cased"
    max_length: int = 256
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    train_batch_size: int = 16
    eval_batch_size: int = 32
    num_train_epochs: int = 3
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.1
    seed: int = 42


@dataclass
class DataConfig:
    dataset_path: Path = Path("../Data/Bin/Data_Bin_Classifier.csv")
    output_dir: Path = Path("./artifacts")
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42


def ensure_output_dirs(base_dir: Path) -> dict:
    base_dir.mkdir(parents=True, exist_ok=True)
    dirs = {
        "base": base_dir,
        "model": base_dir / "model",
        "metrics": base_dir / "metrics",
        "explanations": base_dir / "explanations",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs
