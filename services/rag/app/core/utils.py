import json
from pathlib import Path


def save_json(file_path: str | Path, data: dict | list, log: bool = True):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    if log:
        print(f"Saved '{file_path}'.")


def fetch_filepaths(dir_path: str | Path, format: str = "xml"):
    dir_path = Path(dir_path)
    return list(dir_path.rglob(f"*.{format}"))
