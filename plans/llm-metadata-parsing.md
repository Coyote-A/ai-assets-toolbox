# LLM Metadata Parsing from Description/Readme

## Overview

This document describes the design for using LLM to automatically extract metadata (trigger words, clip skip, recommended weights, etc.) from model descriptions or readme files when metadata is missing or incomplete.

## Current State Analysis

### Existing Metadata Sources

1. **CivitAI API** ([`src/services/civitai.py`](src/services/civitai.py)):
   - Provides `trainedWords` (trigger words)
   - Provides `description` (truncated to 500 chars)
   - Provides `recommendedWeight`
   - Provides `tags`
   - **Gap**: Many models have incomplete API metadata but rich descriptions

2. **ModelInfo Dataclass** ([`src/services/model_metadata.py`](src/services/model_metadata.py:16)):
   ```python
   @dataclass
   class ModelInfo:
       trigger_words: list[str]
       description: str
       default_weight: float
       recommended_weight_min: float
       recommended_weight_max: float
       clip_skip: int
       tags: list[str]
       base_model: str
   ```

3. **Edit Modal UI** ([`src/ui/gradio_app.py:2295`](src/ui/gradio_app.py:2295)):
   - Fields for name, trigger words, weight, CLIP skip, description
   - Currently manual entry only

### Existing CaptionService

The [`CaptionService`](src/gpu/caption.py:78) uses Qwen2.5-VL-3B-Instruct on a T4 GPU:

**Pros for Reuse:**
- Already deployed and working
- Model supports text-only inference (vision-language model can process text)
- Same infrastructure (GPU container, volume mounts)

**Cons for Reuse:**
- Vision-language model is overkill for text extraction
- T4 GPU has cold start overhead (~30-60s)
- Model is optimized for image captioning, not structured extraction

## Design Decision: Extend CaptionService

**Recommendation: Add a new method to CaptionService for text-only metadata extraction.**

### Rationale

1. **Simplicity**: No new service to deploy/maintain
2. **Model Capability**: Qwen2.5-VL can handle text-only tasks effectively
3. **Cost**: Same T4 GPU, no additional resources
4. **Cold Start**: Already paying cold start cost for caption feature

### Alternative Considered: Lightweight Text-Only Service

A separate CPU-based service using a smaller model (e.g., Qwen2.5-1.5B-Instruct):

- **Pros**: Faster cold start, lower cost
- **Cons**: Additional service to maintain, another model to download
- **Decision**: Not worth the complexity for occasional use

## Prompt Design

### System Prompt

```
You are a metadata extraction assistant for AI image generation models (LoRAs, checkpoints).
Extract structured metadata from model descriptions. Be precise and only extract explicitly stated information.
```

### User Prompt Template

```
Extract metadata from this model description. Return ONLY a JSON object with these fields:
- trigger_words: list of trigger words/phrases (strings)
- recommended_weight: single float if one weight mentioned, or null
- recommended_weight_min: float if range mentioned, or null  
- recommended_weight_max: float if range mentioned, or null
- clip_skip: integer 1-12 if mentioned, or null
- tags: list of relevant keywords/categories (strings)
- usage_notes: any special instructions (string, or null)

If information is not mentioned, use null for that field.

Description:
{description_text}
```

### Example Input/Output

**Input:**
```
This LoRA creates a dreamy watercolor style. Use trigger word "wcolorstyle" 
for best results. Recommended strength 0.8-1.2. Works best with CLIP skip 2.
Avoid high denoising strength.
```

**Output:**
```json
{
  "trigger_words": ["wcolorstyle"],
  "recommended_weight": null,
  "recommended_weight_min": 0.8,
  "recommended_weight_max": 1.2,
  "clip_skip": 2,
  "tags": ["watercolor", "style", "dreamy", "artistic"],
  "usage_notes": "Avoid high denoising strength."
}
```

## API Design

### New Method in CaptionService

```python
@modal.method()
def extract_metadata(
    self,
    description: str,
    model_name: str = "",
) -> dict:
    """
    Extract structured metadata from a model description using LLM.
    
    Parameters
    ----------
    description:
        The model description text to parse.
    model_name:
        Optional model name for context (improves extraction quality).
    
    Returns
    -------
    dict
        Extracted metadata fields. Empty dict if extraction fails.
        Fields: trigger_words, recommended_weight, recommended_weight_min,
                recommended_weight_max, clip_skip, tags, usage_notes
    """
```

### Frontend API Endpoint

Add to Gradio UI as a button handler:

```python
def on_auto_fill_from_description(description: str, model_name: str) -> dict:
    """
    Called when user clicks 'Auto-fill from description' button.
    
    Returns Gradio update dict for form fields.
    """
    if not description.strip():
        return gr.update(), gr.update(), ...  # No-op
    
    result = CaptionService().extract_metadata.remote(
        description=description,
        model_name=model_name,
    )
    
    return {
        edit_triggers: ", ".join(result.get("trigger_words", [])),
        edit_weight: result.get("recommended_weight", 1.0),
        edit_clip_skip: result.get("clip_skip", 0),
        # ... other fields
    }
```

## UI Integration

### Button Placement

Add button in edit modal, below the description textbox:

```
┌─────────────────────────────────────────┐
│ ✏️ Edit Model Metadata                  │
├─────────────────────────────────────────┤
│ File: model.safetensors                 │
│ Display Name: [________________]        │
│ Trigger Words: [________________]        │
│ Default Weight: [1.0]                   │
│ CLIP Skip: [0]                          │
│ Description:                            │
│ ┌─────────────────────────────────────┐ │
│ │ This LoRA creates...                │ │
│ │                                     │ │
│ └─────────────────────────────────────┘ │
│ [🤖 Auto-fill from description]         │  ← NEW BUTTON
│                                         │
│ [Cancel] [💾 Save Changes]              │
└─────────────────────────────────────────┘
```

### Button Behavior

1. **Enabled State**: When description field has content
2. **Loading State**: Show spinner during LLM call
3. **Success State**: Fill form fields, show success toast
4. **Error State**: Show error message, don't modify fields

### Gradio Implementation

```python
# In edit modal section
edit_description = gr.Textbox(
    label="Description",
    lines=4,
    placeholder="Paste model description or readme...",
)

auto_fill_btn = gr.Button(
    "🤖 Auto-fill from description",
    variant="secondary",
    size="sm",
)

auto_fill_status = gr.HTML(value="")

# Handler
def on_auto_fill(description: str, model_name: str):
    if not description.strip():
        return {
            edit_triggers: gr.update(),
            edit_weight: gr.update(),
            edit_clip_skip: gr.update(),
            auto_fill_status: "<span style='color:#f88;'>No description to parse</span>",
        }
    
    try:
        result = CaptionService().extract_metadata.remote(
            description=description,
            model_name=model_name,
        )
        
        triggers = ", ".join(result.get("trigger_words", []))
        weight = result.get("recommended_weight") or result.get("recommended_weight_min", 1.0)
        clip_skip = result.get("clip_skip", 0)
        
        return {
            edit_triggers: triggers or gr.update(),
            edit_weight: weight,
            edit_clip_skip: clip_skip or gr.update(),
            auto_fill_status: "<span style='color:#4CAF50;'>✓ Metadata extracted</span>",
        }
    except Exception as e:
        return {
            auto_fill_status: f"<span style='color:#f88;'>Error: {e}</span>",
            # Other fields unchanged
        }

auto_fill_btn.click(
    fn=on_auto_fill,
    inputs=[edit_description, edit_name],
    outputs=[edit_triggers, edit_weight, edit_clip_skip, auto_fill_status],
)
```

## Caching Strategy

### Why Cache?

- LLM calls are slow (~2-5s) and cost GPU time
- Same description may be parsed multiple times
- Users may experiment with different models

### Cache Design

Use `modal.Dict` for distributed caching:

```python
# In app_config.py
metadata_cache = modal.Dict.from_name("ai-toolbox-metadata-cache", create_if_missing=True)

# In caption.py
import hashlib

def _cache_key(description: str) -> str:
    """Generate cache key from description hash."""
    return hashlib.sha256(description.encode()).hexdigest()[:16]

@modal.method()
def extract_metadata(self, description: str, model_name: str = "") -> dict:
    # Check cache first
    cache_key = _cache_key(description)
    if cache_key in metadata_cache:
        logger.info("Cache hit for metadata extraction")
        return metadata_cache[cache_key]
    
    # ... LLM extraction ...
    
    # Store in cache (24h TTL via Modal Dict's TTL feature)
    metadata_cache[cache_key] = result
    return result
```

### Cache Invalidation

- **TTL**: 24 hours (Modal Dict supports TTL)
- **Manual**: No manual invalidation needed (hash-based)
- **Size limit**: Modal Dict handles automatically

## Implementation Steps

### Phase 1: Backend

1. **Add `extract_metadata` method to CaptionService**
   - File: [`src/gpu/caption.py`](src/gpu/caption.py)
   - Add text-only inference method
   - Add JSON parsing with validation
   - Add error handling

2. **Add metadata cache Dict**
   - File: [`src/app_config.py`](src/app_config.py)
   - Add `metadata_cache = modal.Dict.from_name(...)`

3. **Add prompt templates**
   - File: [`src/gpu/caption.py`](src/gpu/caption.py)
   - Add `METADATA_EXTRACTION_PROMPT` constant

### Phase 2: Frontend

4. **Add auto-fill button to edit modal**
   - File: [`src/ui/gradio_app.py`](src/ui/gradio_app.py:2295)
   - Add button below description textbox
   - Add status indicator

5. **Add button handler**
   - File: [`src/ui/gradio_app.py`](src/ui/gradio_app.py)
   - Call `CaptionService().extract_metadata.remote()`
   - Update form fields with results

### Phase 3: Polish

6. **Add loading indicator**
   - Show spinner during LLM call
   - Disable button while processing

7. **Add error handling**
   - Graceful degradation on failure
   - User-friendly error messages

8. **Testing**
   - Test with various description formats
   - Test cache behavior
   - Test error cases

## Technical Details

### JSON Response Parsing

```python
import json
import re

def _parse_llm_json(response: str) -> dict:
    """Parse JSON from LLM response, handling markdown code blocks."""
    # Try direct parse
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    
    # Try extracting from markdown code block
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try finding JSON object in text
    json_match = re.search(r'\{[\s\S]*\}', response)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    logger.warning("Failed to parse LLM response as JSON: %s", response[:200])
    return {}
```

### Text-Only Inference

Qwen2.5-VL supports text-only input by omitting the image:

```python
def _extract_text_only(
    self,
    text: str,
    system_prompt: str,
    max_new_tokens: int = 500,
) -> str:
    """Run text-only inference (no image)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]
    
    text_input = self._processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    inputs = self._processor(
        text=[text_input],
        images=None,  # No images
        return_tensors="pt",
    )
    inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    
    input_token_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_token_len:]
    return self._processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0].strip()
```

## Cost Analysis

### GPU Time

- **Cold start**: ~30-60s (shared with caption feature)
- **Inference**: ~2-5s per extraction
- **GPU**: T4 @ ~$0.40/hour

### Per-Extraction Cost

- Cold start amortized over multiple calls: negligible
- Inference: ~$0.001 per extraction (5s @ $0.40/hr)

### Caching Impact

- Cache hit rate expected: ~70% (same descriptions re-parsed)
- Effective cost: ~$0.0003 per extraction

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| LLM hallucinates metadata | Use conservative prompt; validate outputs |
| JSON parsing fails | Multiple fallback parsing strategies |
| Slow response time | Show loading indicator; cache aggressively |
| GPU unavailable | Graceful error message; retry logic |
| Incorrect extraction | User can edit before saving |

## Future Enhancements

1. **Batch extraction**: Process multiple models at once
2. **Readme file upload**: Parse uploaded .md files
3. **Confidence scores**: Show LLM confidence in extracted values
4. **Manual corrections**: Learn from user corrections
5. **Multi-language support**: Handle non-English descriptions
