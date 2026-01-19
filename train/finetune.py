"""
Simple fine-tuning module for distilling gold standard summaries into smaller models.

Assumes:
- Training data is pre-validated (all samples under 64K tokens)
- JSON structure is known: list of {"url": str, "markdown_content": str, "summary": str}
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

import tiktoken
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
    
    # Monitor progress
    seen_events = set()
    while True:
        job = client.fine_tuning.jobs.retrieve(job.id)
        
        events = client.fine_tuning.jobs.list_events(job.id, limit=20)
        for event in reversed(events.data):
            if event.id not in seen_events:
                seen_events.add(event.id)
                print(f"[{event.created_at}] {event.message}")
        
        if job.status in ("succeeded", "failed", "cancelled"):
            break
        
        time.sleep(10)
    
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