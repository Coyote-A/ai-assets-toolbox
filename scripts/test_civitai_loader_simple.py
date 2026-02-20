#!/usr/bin/env python3
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frontend.civitai_models import (
    load_civitai_model_details,
    get_model_details,
    get_model_details_by_name
)

print("=== Testing CivitAI Model Details Loader ===")

models = load_civitai_model_details()
print(f"Loaded {len(models)} models")

for model in models:
    print(f"\n- Model ID: {model.get('model_id')}")
    print(f"  UI Name: {model.get('ui_name')}")
    print(f"  Full Name: {model.get('full_name')}")
    print(f"  Tags: {', '.join(model.get('tags', []))}")
    print(f"  Trigger Words: {', '.join(model.get('trigger_words', []))}")

print("\n=== All tests completed ===")
