from __future__ import annotations

from pathlib import Path
import pandas as pd


DEFAULT_DATA_PATH = Path("data/raw/stroke.csv")


def load_stroke_csv(path: Path | str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Put CSV at data/raw/stroke.csv (gitignored)."
        )
    df = pd.read_csv(path)
    return df