import json
from pathlib import Path


def save_json(file_path: Path, data: dict | list, log: bool = True):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    if log:
        print(f"Saved '{file_path}'.")
