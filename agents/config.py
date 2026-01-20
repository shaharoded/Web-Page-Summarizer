# Paths
PROMPT_REPO_PATH = "agents/prompts/"

# Pricing per 1M tokens as of Jan 2026
RATES = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1-2025-04-14": {"input": 2.00, "output": 8.00, "SFT-input": 3.0, "SFT-output": 12.0, "SFT-training": 25.0},
    "gpt-4.1-mini-2025-04-14": {"input": 0.4, "output": 1.6, "SFT-input": 0.8, "SFT-output": 3.2, "SFT-training": 5.0},
    "gpt-4.1-nano-2025-04-14": {"input": 0.1, "output": 0.4, "SFT-input": 0.2, "SFT-output": 0.8, "SFT-training": 1.5},
    "ft:gpt-4.1-mini-2025-04-14:tavily::CzWAcE6p": {"input": 0.8, "output": 3.2}, # SFT model
    "ft:gpt-4.1-nano-2025-04-14:tavily::CzX41hjk": {"input": 0.2, "output": 0.8}, # SFT model
    "ft:gpt-4.1-mini-2025-04-14:tavily::CzqO8otr": {"input": 0.8, "output": 3.2}, # SFT model
    "ft:gpt-4.1-nano-2025-04-14:tavily::Czr44ng2": {"input": 0.2, "output": 0.8}, # SFT model
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5.2-2025-12-11": {"input": 1.75, "output": 14.00}
}