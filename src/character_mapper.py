import json
from typing import Dict, List, Any, Optional

class CharacterMapper:
    def __init__(self, file_path: str):
        print("Initializing Character Mapper...")
        canonical_data = self._load_character_data(file_path)
        self.alias_to_canonical_map = self._build_alias_map(canonical_data)
        self.all_canonical_names = [char["canonical_name"] for char in canonical_data]
        print(f"Character map built successfully with {len(self.alias_to_canonical_map)} total aliases.")

    def _load_character_data(self, file_path: str) -> List[Dict[str, Any]]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f: return json.load(f)
        except Exception as e: print(f"Error loading {file_path}: {e}"); return []

    def _build_alias_map(self, canonical_data: List[Dict[str, Any]]) -> Dict[str, str]:
        lookup_map = {}
        for character in canonical_data:
            canonical_name = character["canonical_name"]
            lookup_map[canonical_name.lower()] = canonical_name
            for alias in character.get("aliases", []):
                lookup_map[alias.lower()] = canonical_name
        return lookup_map

    def get_canonical_name(self, mention: str) -> Optional[str]: # Changed to Optional[str]
        return self.alias_to_canonical_map.get(mention.lower())