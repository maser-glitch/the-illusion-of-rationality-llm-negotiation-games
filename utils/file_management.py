import json
import pickle
from pathlib import Path


def save_json(data: dict | list, file_path: str) -> None:
    """
    Writes data to a JSON file.
    Creates parent directories if they don't exist.
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"JSON file created at: {file_path}")

def load_json(path: str):
    """Load and return a Python object from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_pickle(path: str):
    """Load and return a Python object from a PICKLE file."""
    with open(path, "rb") as f:
        pickle_data = pickle.load(f)
    return pickle_data


def load_text(path: str) -> str:
    """
    Read a text file and return its contents as a string.
    Tries UTF-8 first, falls back to Latin-1 with replacement if needed.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1", errors="replace") as f:
            return f.read()

def create_directory(path: str) -> None:
    """
    Creates a directory and any necessary parent directories.
    Does not raise an error if the directory already exists.
    """
    Path(path).mkdir(parents=True, exist_ok=True)
    print(f"Directory ensured at: {path}")

def file_exists(path: str) -> bool:
    """
    Checks if a strictly regular file exists at the given path.
    """
    return Path(path).is_file()