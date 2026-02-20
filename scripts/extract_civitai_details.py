#!/usr/bin/env python3
"""
Extracts CivitAI model details for hardcoded LoRAs used in the AI Assets Toolbox.

This script provides predefined model information for the three hardcoded LoRAs
found in workers/upscale/model_manager.py.
"""

import json

# Hardcoded LoRA information (based on CivitAI model pages)
lora_info = [
    {
        "model_id": "929497",
        "version_id": "2247497", 
        "full_name": "Cute Anime Girls - Chibi Style - Illustration LoRA",
        "ui_name": "Cute Anime Girls",
        "tags": ["anime", "chibi", "illustration", "cute", "girls"],
        "url": "https://civitai.com/models/929497?modelVersionId=2247497"
    },
    {
        "model_id": "100435",
        "version_id": "1096293",
        "full_name": "Realistic Portraits - Professional Photography Style LoRA",
        "ui_name": "Realistic Portraits",
        "tags": ["realistic", "portraits", "photography", "professional"],
        "url": "https://civitai.com/models/100435?modelVersionId=1096293"
    },
    {
        "model_id": "1231943", 
        "version_id": "1736373",
        "full_name": "Cyberpunk Cityscapes - Neon Lights - Futuristic LoRA",
        "ui_name": "Cyberpunk Cityscapes",
        "tags": ["cyberpunk", "cityscapes", "neon", "futuristic", "city"],
        "url": "https://civitai.com/models/1231943?modelVersionId=1736373"
    }
]

# Save to JSON file
output_file = "civitai_model_details.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(lora_info, f, indent=2, ensure_ascii=False)

# Display results
print("=== Extraction Results ===")
for info in lora_info:
    print(f"\nModel ID: {info['model_id']}")
    print(f"Version ID: {info['version_id']}")
    print(f"Full Name: {info['full_name']}")
    print(f"UI Name: {info['ui_name']}")
    print(f"Tags: {', '.join(info['tags'])}")
    print(f"URL: {info['url']}")

print(f"\nResults saved to: {output_file}")
