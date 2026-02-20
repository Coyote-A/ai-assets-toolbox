"""
CivitAI model details loader for the AI Assets Toolbox.
"""
import json
import os
from typing import List, Dict, Any, Optional

# Path to the model details JSON file
CIVITAI_DETAILS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts",
    "civitai_model_details.json"
)


def load_civitai_model_details() -> List[Dict[str, Any]]:
    """
    Load CivitAI model details from the JSON file.
    
    Returns:
        List of model details dictionaries with keys:
        - model_id
        - version_id
        - full_name
        - ui_name
        - tags
        - url
        - trigger_words
        - usage_tips
    """
    try:
        with open(CIVITAI_DETAILS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading CivitAI model details: {e}")
        return []


def get_model_details(model_id: str) -> Optional[Dict[str, Any]]:
    """Get details for a specific model by model_id."""
    models = load_civitai_model_details()
    for model in models:
        if model.get("model_id") == model_id:
            return model
    return None


def get_model_details_by_name(name: str) -> Optional[Dict[str, Any]]:
    """Get details for a specific model by name (matches against ui_name or full_name)."""
    models = load_civitai_model_details()
    name = name.lower()
    
    for model in models:
        if (
            name in model.get("ui_name", "").lower() or
            name in model.get("full_name", "").lower()
        ):
            return model
            
        # Also check against key variations
        key_map = {
            "aesthetic-quality": "aesthetic quality modifiers - masterpiece",
            "character-design": "character design sheet",
            "detailer-il": "detailer il"
        }
        
        if name in key_map and key_map[name] in model.get("full_name", "").lower():
            return model
            
    return None
