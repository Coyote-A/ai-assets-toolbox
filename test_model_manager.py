#!/usr/bin/env python3
"""
Test script to verify the ModelManager updates for real LoRA names.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'workers', 'upscale')))

from model_manager import HARDCODED_LORAS, ModelManager, ensure_loras_downloaded

def test_hardcoded_loras_structure():
    """Test that HARDCODED_LORAS has the correct structure."""
    print("Testing HARDCODED_LORAS structure...")
    
    expected_keys = ["Aesthetic Quality", "Character Design", "Detailer IL"]
    for key in expected_keys:
        assert key in HARDCODED_LORAS, f"Missing expected LoRA: {key}"
        lora_info = HARDCODED_LORAS[key]
        assert "model_id" in lora_info, f"Missing model_id for {key}"
        assert "version_id" in lora_info, f"Missing version_id for {key}"
        assert "filename" in lora_info, f"Missing filename for {key}"
        assert lora_info["filename"].endswith(".safetensors"), f"Filename should end with .safetensors: {key}"
    
    print("OK: HARDCODED_LORAS has correct structure")

def test_ensure_loras_downloaded():
    """Test that ensure_loras_downloaded function works with new structure."""
    print("\nTesting ensure_loras_downloaded...")
    
    # This should not raise any exceptions
    ensure_loras_downloaded()
    print("OK: ensure_loras_downloaded executed without errors")

if __name__ == "__main__":
    test_hardcoded_loras_structure()
    test_ensure_loras_downloaded()
    
    # Test that we can get an instance of the manager
    manager = ModelManager.get_instance()
    print("\nOK: ModelManager instantiated successfully")
