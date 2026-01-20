"""
Simple fine-tuning module for distilling gold standard summaries into smaller models.

Assumes:
- Training data is pre-validated (all samples under 64K tokens)
- JSON structure is known: list of {"url": str, "markdown_content": str, "summary": str}
"""

import json
import os
import time
import re
from pathlib import Path
from typing import Optional
from collections import deque

import tiktoken
import matplotlib.pyplot as plt
from openai import OpenAI

# Local code
from agents.config import RATES, PROMPT_REPO_PATH
from agents.summarizer import Summarizer


def prepare_and_train(
    model: str,
    train_json_path: str,
    n_epochs: int = 3,
    api_key: Optional[str] = None,
) -> str:
    """
    Prepare training data and submit the fine-tuning job.
    
    Returns: fine_tuned_model_id (e.g., "ft:gpt-4.1-mini-2025-04-14:...")
    """
    # Load training data
    with open(train_json_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    # Initialize summarizer to get prompts
    summarizer = Summarizer(model=model, prompt_repo_path=PROMPT_REPO_PATH)
    
    # Convert to JSONL format
    train_jsonl = Path("data") / f"train_{model.replace(':', '_')}.jsonl"
    train_jsonl.parent.mkdir(exist_ok=True)
    
    total_tokens = 0
    enc = tiktoken.encoding_for_model("gpt-4o")  # All models use same tokenizer
    
    with open(train_jsonl, 'w', encoding='utf-8') as f:
        for item in train_data:
            user_prompt = summarizer.user_template.format(source=item['markdown_content'])
            messages = [
                {"role": "system", "content": summarizer.system_template},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": item['summary']}
            ]
            
            # Count tokens for cost estimation
            for msg in messages:
                total_tokens += len(enc.encode(msg['content']))
            
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
    
    # Calculate estimated cost
    sft_training_rate = RATES[model]["SFT-training"]  # $ per 1M tokens
    estimated_cost = (total_tokens * n_epochs / 1_000_000.0) * sft_training_rate
    
    print(f"\n{'='*60}")
    print(f"Model: {model}")
    print(f"Training samples: {len(train_data)}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Epochs: {n_epochs}")
    print(f"Estimated training cost: ${estimated_cost:.2f}")
    print(f"Training file: {train_jsonl}")
    print(f"{'='*60}\n")
    
    # Submit fine-tuning job
    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    
    with open(train_jsonl, 'rb') as f:
        train_file = client.files.create(file=f, purpose="fine-tune")
    
    job = client.fine_tuning.jobs.create(
        model=model,
        training_file=train_file.id,
        hyperparameters={"n_epochs": n_epochs}
    )
    
    print(f"Fine-tuning job submitted: {job.id}")
    print(f"Monitor at: https://platform.openai.com/finetune/{job.id}\n")
    
    # Monitor progress and collect loss data
    seen_events = set()
    loss_data = []
    moving_avg_window = 100
    
    while True:
        job = client.fine_tuning.jobs.retrieve(job.id)
        
        events = client.fine_tuning.jobs.list_events(job.id, limit=20)
        for event in reversed(events.data):
            if event.id not in seen_events:
                seen_events.add(event.id)
                
                # Extract loss from event message if present
                # Try multiple patterns that OpenAI might use
                loss_patterns = [
                    r'loss: ([\d.]+)',           # loss: 0.123
                    r'training loss=([\d.]+)',   # training loss=0.123
                    r'loss=([\d.]+)',            # loss=0.123
                    r'loss\s*=\s*([\d.]+)',      # loss = 0.123 (with spaces)
                ]
                
                loss_value = None
                for pattern in loss_patterns:
                    loss_match = re.search(pattern, event.message, re.IGNORECASE)
                    if loss_match:
                        loss_value = float(loss_match.group(1))
                        break
                
                if loss_value is not None:
                    loss_data.append(loss_value)
                
                print(f"[{event.created_at}] {event.message}")
                
                # Debug: print first few events that might contain loss data
                debug_count = sum(1 for e in seen_events if 'step' in event.message.lower() or 'loss' in event.message.lower())
                if debug_count <= 3 and ('step' in event.message.lower() or 'loss' in event.message.lower()):
                    print(f"  DEBUG: Checking message format: '{event.message}'")
        
        if job.status in ("succeeded", "failed", "cancelled"):
            break
        
        time.sleep(10)
    
    # Create loss plot if we have loss data
    if loss_data:
        # Calculate moving average
        if len(loss_data) >= moving_avg_window:
            moving_avg = []
            window = deque(maxlen=moving_avg_window)
            
            for loss in loss_data:
                window.append(loss)
                if len(window) == moving_avg_window:
                    moving_avg.append(sum(window) / len(window))
            
            # Create plot
            plt.figure(figsize=(12, 6))
            plt.plot(range(len(loss_data)), loss_data, alpha=0.3, color='blue', label='Raw Loss')
            if moving_avg:
                plt.plot(range(moving_avg_window-1, len(loss_data)), moving_avg, 
                        color='red', linewidth=2, label=f'Moving Avg (last {moving_avg_window} samples)')
            
            plt.xlabel('Training Steps')
            plt.ylabel('Loss')
            plt.title(f'Fine-tuning Loss for {model}\nJob: {job.id}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save plot
            plot_path = f"data/loss_plot_{model.replace(':', '_')}_{job.id}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"\n📊 Loss plot saved to: {plot_path}")
            plt.close()
        else:
            print(f"\n⚠ Not enough loss data points ({len(loss_data)}) for moving average plot (need ≥{moving_avg_window})")
    else:
        print(f"\n⚠ No loss data found in training events")
        print(f"   Checked {len(seen_events)} events")
        print(f"   Expected patterns: 'loss: X.XX', 'training loss=X.XX', 'loss=X.XX'")
        print(f"   This might be normal for some OpenAI fine-tuning jobs")
    
    if job.status != "succeeded":
        raise RuntimeError(f"Fine-tuning failed: {job.status}")
    
    # Report final costs
    trained_tokens = job.trained_tokens or total_tokens * n_epochs
    actual_cost = (trained_tokens / 1_000_000.0) * sft_training_rate
    
    print(f"\n{'='*60}")
    print(f"✓ Fine-tuning completed!")
    print(f"Fine-tuned model: {job.fine_tuned_model}")
    print(f"Trained tokens: {trained_tokens:,}")
    print(f"Actual cost: ${actual_cost:.2f}")
    print(f"{'='*60}\n")
    
    return job.fine_tuned_model


def get_model_info(model_id: str, api_key: Optional[str] = None) -> dict:
    """
    Retrieve information about a fine-tuned model.
    
    Returns: dict with model details (id, created, base_model, etc.)
    """
    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    
    # Get model details
    model = client.models.retrieve(model_id)
    
    info = {
        "id": model.id,
        "created": model.created,
        "owned_by": model.owned_by,
    }
    
    print(f"\nModel Information:")
    print(f"  ID: {info['id']}")
    print(f"  Created: {info['created']}")
    print(f"  Owned by: {info['owned_by']}")
    
    return info