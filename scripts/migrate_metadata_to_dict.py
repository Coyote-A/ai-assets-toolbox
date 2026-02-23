"""
One-time migration: JSON files -> Modal Dict

This script migrates existing metadata from JSON files on the volumes to
Modal Dict storage. Run this once after deploying the new code.

Usage:
    modal run scripts/migrate_metadata_to_dict.py

After successful migration, the JSON files (.manifest.json, .progress.json,
.api_tokens.json) can be safely deleted from the volumes if desired.
"""

import modal

app = modal.App("migrate-metadata-to-dict")

# Volumes containing the JSON files
models_volume = modal.Volume.from_name("ai-toolbox-models")
loras_volume = modal.Volume.from_name("ai-toolbox-loras")

# Target Dicts
metadata_dict = modal.Dict.from_name("ai-toolbox-model-metadata", create_if_missing=True)
tokens_dict = modal.Dict.from_name("ai-toolbox-tokens", create_if_missing=True)

# Paths
MODELS_MOUNT = "/vol/models"
LORAS_MOUNT = "/vol/loras"


@app.function(
    volumes={
        MODELS_MOUNT: models_volume,
        LORAS_MOUNT: loras_volume,
    },
    timeout=300,
)
def migrate() -> dict:
    """Migrate all JSON metadata to Modal Dicts."""
    import json
    import os
    
    results = {
        "manifest": {"migrated": False, "entries": 0, "error": None},
        "progress": {"migrated": False, "entries": 0, "error": None},
        "tokens": {"migrated": False, "entries": 0, "error": None},
    }
    
    # Reload volumes to see latest state
    models_volume.reload()
    loras_volume.reload()
    
    # --- Migrate manifest (.manifest.json) ---
    manifest_path = os.path.join(MODELS_MOUNT, ".manifest.json")
    try:
        if os.path.exists(manifest_path):
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            metadata_dict["manifest"] = manifest
            results["manifest"]["migrated"] = True
            results["manifest"]["entries"] = len(manifest)
            print(f"✓ Migrated manifest with {len(manifest)} entries")
        else:
            print("ℹ No manifest file found (starting fresh)")
            results["manifest"]["migrated"] = True
            results["manifest"]["entries"] = 0
    except Exception as e:
        results["manifest"]["error"] = str(e)
        print(f"✗ Failed to migrate manifest: {e}")
    
    # --- Migrate progress (.progress.json) ---
    progress_path = os.path.join(MODELS_MOUNT, ".progress.json")
    try:
        if os.path.exists(progress_path):
            with open(progress_path, "r") as f:
                progress = json.load(f)
            metadata_dict["progress"] = progress
            results["progress"]["migrated"] = True
            results["progress"]["entries"] = len(progress)
            print(f"✓ Migrated progress with {len(progress)} entries")
        else:
            print("ℹ No progress file found (starting fresh)")
            results["progress"]["migrated"] = True
            results["progress"]["entries"] = 0
    except Exception as e:
        results["progress"]["error"] = str(e)
        print(f"✗ Failed to migrate progress: {e}")
    
    # --- Migrate tokens (.api_tokens.json) ---
    tokens_path = os.path.join(LORAS_MOUNT, ".api_tokens.json")
    try:
        if os.path.exists(tokens_path):
            with open(tokens_path, "r") as f:
                tokens = json.load(f)
            tokens_dict["tokens"] = tokens
            results["tokens"]["migrated"] = True
            results["tokens"]["entries"] = 1  # Single entry with both tokens
            print(f"✓ Migrated tokens (hf_token={'set' if tokens.get('hf_token') else 'empty'}, civitai_token={'set' if tokens.get('civitai_token') else 'empty'})")
        else:
            print("ℹ No tokens file found (starting fresh)")
            results["tokens"]["migrated"] = True
            results["tokens"]["entries"] = 0
    except Exception as e:
        results["tokens"]["error"] = str(e)
        print(f"✗ Failed to migrate tokens: {e}")
    
    # Summary
    print("\n--- Migration Summary ---")
    all_success = all(
        r["migrated"] and r["error"] is None 
        for r in results.values()
    )
    if all_success:
        print("✓ All metadata migrated successfully!")
    else:
        print("✗ Some migrations failed. Check errors above.")
    
    return results


@app.local_entrypoint()
def main():
    """Run the migration."""
    print("Starting metadata migration to Modal Dict...")
    print("=" * 50)
    result = migrate.remote()
    print("=" * 50)
    
    # Print final status
    if all(r["migrated"] and r["error"] is None for r in result.values()):
        print("\n✓ Migration completed successfully!")
        print("\nYou can now safely delete the old JSON files from the volumes:")
        print("  - /vol/models/.manifest.json")
        print("  - /vol/models/.progress.json")
        print("  - /vol/loras/.api_tokens.json")
    else:
        print("\n✗ Migration completed with errors. Review the output above.")
