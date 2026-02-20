#!/usr/bin/env python3
"""
Test script to verify CivitAI model details loading.
"""
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frontend.civitai_models import (
    load_civitai_model_details,
    get_model_details,
    get_model_details_by_name
)

print("=== Testing CivitAI Model Details Loader ===")

# Test 1: Load all models
print("\n1. Loading all model details:")
models = load_civitai_model_details()
print(f"Loaded {len(models)} models")

for model in models:
    print(f"\n- Model ID: {model.get('model_id')}")
    print(f"  UI Name: {model.get('ui_name')}")
    print(f"  Full Name: {model.get('full_name')}")
    print(f"  Tags: {', '.join(model.get('tags', []))}")
    print(f"  Trigger Words: {', '.join(model.get('trigger_words', []))}")

# Test 2: Get model by ID
print("\n2. Testing get_model_details():")
model_929497 = get_model_details("929497")
if model_929497:
    print(f"Found model 929497: {model_929497['ui_name']}")

# Test 3: Get model by name
print("\n3. Testing get_model_details_by_name():")
aesthetic_model = get_model_details_by_name("aesthetic quality")
if aesthetic_model:
    print(f"Found 'aesthetic quality' model: {aesthetic_model['full_name']}")

character_model = get_model_details_by_name("character design")
if character_model:
    print(f"Found 'character design' model: {character_model['full_name']}")

detailer_model = get_model_details_by_name("detailer il")
if detailer_model:
    print(f"Found 'detailer il' model: {detailer_model['full_name']}")

# Test 4: Verify all hardcoded LoRAs are present
print("\n4. Checking hardcoded LoRA consistency:")
hardcoded_loras = ["aesthetic-quality", "character-design", "detailer-il"]

for lora_name in hardcoded_loras:
    found = False
    for model in models:
        # Check if name matches UI name or full name
        if lora_name.replace("-", " ") in model.get("ui_name", "").lower() or \
           lora_name.replace("-", " ") in model.get("full_name", "").lower():
            found = True
            print(f"✓ '{lora_name}' matches '{model['ui_name']}'")
            break
            
    if not found:
        print(f"✗ '{lora_name}' not found in model details")

print("\n=== All tests completed ===")
