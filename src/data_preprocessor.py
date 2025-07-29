import re
from pathlib import Path
from typing import Dict, List

# No NLTK imports are needed here anymore. The main script will handle it.

def load_books(directory_path: str) -> Dict[str, str]:
    """Loads all .txt files from a directory into a dictionary."""
    book_texts = {}
    data_path = Path(directory_path)
    if not data_path.is_dir():
        raise FileNotFoundError(f"Error: Directory not found at {directory_path}")

    for file_path in sorted(data_path.glob("*.txt")):
        with open(file_path, 'r', encoding='utf-8') as f:
            book_texts[file_path.name] = f.read()
    return book_texts
