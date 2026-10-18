
"""Check the motor dataset split without fitting or saving a model."""
from pathlib import Path
import sys

# Project root contains both scripts/ and src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders import load_motor_claims
from src.data.schemas import MOTOR_TARGET
from src.data.splits import split_dataframe
from src.data.loaders import load_motor_claims
from src.data.schemas import MOTOR_TARGET
from src.data.splits import split_dataframe


def main():
    seed = 42
    sample_size = 1000

    # Load the same source dataset used by model training.
    raw = load_motor_claims()

    if MOTOR_TARGET not in raw.columns:
        raise ValueError(
            f"Target column '{MOTOR_TARGET}' not found. "
            f"Available columns: {list(raw.columns)}"
        )

    original_rows = len(raw)

    # Reproduce the training script's sampling procedure.
    if len(raw) > sample_size:
        raw = raw.sample(n=sample_size, random_state=seed)

    # Reproduce the training script's stratified split.
    train, val, test = split_dataframe(
        raw,
        MOTOR_TARGET,
        random_state=seed,
    )

    partitions = {
        "Train": train,
        "Validation": val,
        "Test": test,
    }

    print("\nMOTOR DATASET SPLIT AUDIT")
    print("=" * 45)
    print(f"Original dataset rows : {original_rows}")
    print(f"Rows after sampling   : {len(raw)}")
    print(f"Random seed           : {seed}")
    print(f"Target column         : {MOTOR_TARGET}")
    print("-" * 45)

    for name, part in partitions.items():
        count = len(part)
        percentage = count / len(raw) * 100 if len(raw) else 0
        positive_count = int(part[MOTOR_TARGET].astype(int).sum())
        prevalence = (
            positive_count / count * 100 if count else 0
        )

        print(
            f"{name:12} | Rows: {count:5} "
            f"| Share: {percentage:6.2f}% "
            f"| Fraud labels: {positive_count:4} "
            f"| Prevalence: {prevalence:6.2f}%"
        )

    total = sum(len(part) for part in partitions.values())

    print("-" * 45)
    print(f"Total partition rows   : {total}")
    print(f"Split covers all rows  : {total == len(raw)}")
    print("\nNo model was trained and no artifacts were written.")


if __name__ == "__main__":
    main()
