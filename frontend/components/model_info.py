"""
UI components for displaying CivitAI model information.
"""
from __future__ import annotations

import gradio as gr
from typing import List, Dict, Any

from frontend.civitai_models import load_civitai_model_details, get_model_details_by_name


def create_model_info_card(model: Dict[str, Any]) -> gr.HTML:
    """
    Create an HTML card to display model information.
    
    Args:
        model: Model details dictionary containing:
            - ui_name
            - full_name
            - tags
            - trigger_words
            - usage_tips
            - url
    """
    tags_html = "".join([
        f'<span class="model-tag">{tag}</span>' 
        for tag in model.get("tags", [])
    ])
    
    trigger_words_html = "".join([
        f'<code class="trigger-word">{word}</code>' 
        for word in model.get("trigger_words", [])
    ])
    
    card_html = f"""
    <div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px; margin: 8px 0; background-color: #f8f9fa;">
        <h3 style="margin: 0 0 8px 0; color: #333;">{model.get('ui_name', 'Unknown Model')}</h3>
        <p style="margin: 0 0 12px 0; color: #666; font-size: 0.9em;">
            {model.get('full_name', 'No description available')}
        </p>
        
        {'' if not tags_html else f'<div style="margin-bottom: 12px;">{tags_html}</div>'}
        
        {'' if not trigger_words_html else f'<div style="margin-bottom: 12px;">\
            <strong>Trigger Words:</strong> {trigger_words_html}\
        </div>'}
        
        {'' if not model.get('usage_tips') else f'<div style="margin-bottom: 12px;">\
            <strong>Usage Tips:</strong> {model.get("usage_tips")}\
        </div>'}
        
        <a href="{model.get('url', '#')}" target="_blank" style="color: #0066cc; text-decoration: none; font-size: 0.9em;">
            View on CivitAI →
        </a>
    </div>
    """
    
    return gr.HTML(card_html)


def create_model_info_section() -> gr.Group:
    """Create a complete section for displaying model information."""
    models = load_civitai_model_details()
    
    with gr.Group():
        gr.Markdown("## CivitAI Model Information")
        gr.Markdown(
            "Information about the hardcoded LoRA models used in the AI Assets Toolbox. "
            "These models are downloaded automatically at startup if they don't exist locally."
        )
        
        for model in models:
            create_model_info_card(model)
            
    return gr.Group()
