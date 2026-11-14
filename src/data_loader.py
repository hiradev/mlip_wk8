from pathlib import Path

import yaml
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "params.yaml"


def load_data(config_path=DEFAULT_CONFIG):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]
    return train_test_split(
        X, y, test_size=config["test_size"], random_state=config["random_seed"]
    )
