#!/usr/bin/env python3
"""
Test script for automatic LoRA trigger word injection.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frontend.civitai_models import get_model_details_by_name
from frontend.tabs.upscale_tab import HARDCODED_LORAS


def test_trigger_injection(base_prompt):
    print(f"Base prompt: '{base_prompt}'")
    
    # Simulate trigger word injection
    injected_triggers = []
    
    for lora in HARDCODED_LORAS:
        model_details = get_model_details_by_name(lora["name"])
        if model_details and "trigger_words" in model_details:
            for trigger in model_details["trigger_words"]:
                if trigger not in base_prompt and trigger not in injected_triggers:
                    injected_triggers.append(trigger)
    
    if injected_triggers:
        trigger_text = ", ".join(injected_triggers)
        if base_prompt:
            full_prompt = f"{trigger_text}, {base_prompt}"
        else:
            full_prompt = trigger_text
        
        print(f"Triggers added: '{', '.join(injected_triggers)}'")
        print(f"Full prompt: '{full_prompt}'")
    else:
        print("No new triggers added")
    
    return full_prompt


# Test cases
print("=== Test 1: Empty prompt ===")
test_trigger_injection("")

print("\n=== Test 2: Prompt without triggers ===")
test_trigger_injection("beautiful landscape, mountains, sunset")

print("\n=== Test 3: Prompt with some triggers ===")
test_trigger_injection("masterpiece, beautiful landscape, mountains")

print("\n=== All tests completed ===")
