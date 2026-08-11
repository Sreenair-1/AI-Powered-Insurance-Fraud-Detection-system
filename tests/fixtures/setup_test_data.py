import os
from pathlib import Path
import pandas as pd

def setup_test_datasets():
    """Create minimal mock datasets for CI testing."""
    base_path = Path("/data/raw")
    
    # Create directories
    (base_path / "demographics").mkdir(parents=True, exist_ok=True)
    (base_path / "motor").mkdir(parents=True, exist_ok=True)
    (base_path / "health" / "train").mkdir(parents=True, exist_ok=True)
    (base_path / "health" / "test").mkdir(parents=True, exist_ok=True)
    
    # Create minimal motor dataset
    motor_df = pd.DataFrame({
        "claim_id": [1, 2, 3],
        "claim_amount": [1000, 2000, 3000],
        "fraud": [0, 1, 0]
    })
    motor_df.to_csv(base_path / "motor" / "fraud_oracle.csv", index=False)
    
    # Create minimal demographics dataset
    demo_df = pd.DataFrame({
        "state": ["AP", "KA", "TN"],
        "population": [1000000, 2000000, 3000000]
    })
    demo_df.to_csv(base_path / "demographics" / "Indian_Insurance_Data.csv", index=False)
    
    # Create minimal health datasets (add required columns based on your schema)
    health_main = pd.DataFrame({"Provider": [1, 2, 3], "PotentialFraud": [0, 1, 0]})
    health_main.to_csv(base_path / "health" / "train" / "main.csv", index=False)
    