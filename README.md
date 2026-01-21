# Web Page Summarizer: LLM Distillation & Evaluation Framework

A comprehensive framework for benchmarking LLMs for web page summarization task using LLM-as-a-Judge methodology, including distillation flow on OpenAI models.

## Overview

This project implements a complete pipeline for:
1. **Gold Standard Creation**: Generating high-quality reference summaries using state-of-the-art LLMs (GPT-5.2)
2. **Model Distillation**: Fine-tuning smaller, cost-efficient models (GPT-4.1 family) to replicate the performance of larger models using supervised-fine-tuning (SFT) on the OpenAI SDK.
3. **Benchmarking & Evaluation**: Comprehensive evaluation framework using LLM-as-a-judge with G-Eval methodology.
4. **Cost-Performance Analysis**: Detailed tracking of inference costs, latency, and quality metrics.

## Project Structure

```
Web-Page-Summarizer/
├── agents/                          # Core LLM agents and utilities
│   ├── config.py                    # Model pricing and configuration
│   ├── llm.py                       # LLM engine wrapper with token counting
│   ├── summarizer.py                # Summarization agent
│   ├── judge.py                     # Evaluation agent (LLM-as-judge)
│   └── prompts/                     # Prompt templates and schemas
│       ├── judge_schema.json        # Structured output schema for judge
│       ├── judge_system.txt         # Judge system prompt
│       ├── judge_user.txt           # Judge user prompt template
│       ├── summarizer_system.txt    # Summarizer system prompt
│       └── summarizer_user.txt      # Summarizer user prompt template
│
├── train/                           # Fine-tuning utilities
│   └── finetune.py                  # Simplified SFT training pipeline
│
├── evaluation/                      # Benchmarking and evaluation
│   └── benchmark.py                 # BenchmarkingSuite class
│
├── tests/                           # Unit tests
│   └── test_agents.py
│
├── data/                            # Datasets (generated during runtime)
│   └── baseline_1k.json             # Original baseline summaries (for comparison)
│
├── main.ipynb                       # Main workflow notebook
├── requirements.txt                 # Python dependencies
└── .env                             # Env file (including OpenAI API key)

```

## Technical Architecture

```mermaid
flowchart TB
    subgraph Data["📊 Data Preparation"]
        A[Baseline 1K Dataset<br/>Original Summaries] --> B[GPT-5.2 Gold Standard<br/>~1K summaries]
        B --> C{Split ~50%-50%<br/>with constraints}
        C -->|≤1500 chars<br/>AND ≤64K tokens| D[Training Set<br/>Up to 500 samples]
        C -->|Remaining| E[Validation Set<br/>~500 samples]
        E --> F{Sample 20%<br/>≤250K tokens}
        F --> G[Benchmark Subset<br/>99 samples]
        A --> H[Baseline Benchmark<br/>99 matched samples]
    end

    subgraph Train["🎓 Model Distillation"]
        D --> I[SFT Training<br/>3 epochs]
        I --> J[GPT-4.1-mini<br/>Fine-tuned]
        I --> K[GPT-4.1-nano<br/>Fine-tuned]
    end

    subgraph Eval["⚖️ Benchmarking Pipeline"]
        E --> L{Model Type?}
        G --> L
        L -->|Baseline| M[Load Existing<br/>Summaries]
        L -->|Other Models| N[Run Inference<br/>+ Track Latency]
        
        M --> O[LLM-as-Judge<br/>GPT-5.2 Evaluation]
        N --> O
        
        O --> P[Multi-Dimensional<br/>Scoring]
        P --> Q[Relevance 1-5]
        P --> R[Faithfulness 1-5]
        P --> S[Coherence 1-5]
        P --> T[Fluency 1-5]
        P --> U[Conciseness 1-5]
        
        Q & R & S & T & U --> V[Aggregate Statistics<br/>+ Cost Analysis]
        V --> W[📈 Results DataFrame<br/>Quality vs Cost vs Latency]
    end

    subgraph Models["🤖 Models Tested"]
        X1[Baseline]
        X2[gpt-4o-mini]
        X3[gpt-4.1-mini/nano]
        X4[ft:gpt-4.1-mini/nano]
        X5[gpt-5-nano/mini]
        X6[gpt-5.2]
    end

    J --> L
    K --> L
    X1 & X2 & X3 & X4 & X5 & X6 -.->|Evaluated| L

    style A fill:#e1f5ff
    style B fill:#fff4e1
    style D fill:#ffe1f5
    style I fill:#e8ffe1
    style J fill:#e8ffe1
    style O fill:#ffe1e1
    style W fill:#f0e1ff
```

### 1. LLM Engine (`agents/llm.py`)

Core wrapper around OpenAI API with:
- **Token Counting**: Token estimation using tiktoken with model-specific encodings.
- **Cost Tracking**: Real-time cost calculation based on input/output tokens and pricing tables.
- **Model Validation**: Support for base models and fine-tuned models (format: `ft:base-model:org::job-id`)
- **Parameter Adjustment**: Wrapper can work with many model families despite differences in API.
- **Input Reduction**: Context trimming to optimize costs while preserving key information
- **Map-Summarize-Reduce** mechanism to allow long context windows (optional).

### 2. Summarization Agent (`agents/summarizer.py`)

Generates concise, high-quality summaries of web page content:
- Configurable model selection (GPT-4.1, GPT-5.2, fine-tuned variants)
- Template-based prompting for consistency
- Cost tracking per request
- Input reduction for cost optimization
- Retry mechanism to respect output length requirements

### 3. Evaluation Agent (`agents/judge.py`)

LLM-as-a-judge implementation using G-Eval methodology:
- **Structured Outputs**: Uses GPT-5+ structured output format with JSON schema validation
- **Multi-Dimensional Scoring**: Evaluates summaries across 5 dimensions:
  - **Relevance (1-5)**: Captures key information without redundancy
  - **Faithfulness (1-5)**: All claims supported by source (no hallucinations)
  - **Coherence (1-5)**: Logical structure and flow
  - **Fluency (1-5)**: Professional grammar and readability
  - **Conciseness (1-5)**: Information density without filler
- **Length Tracking**: Records character count for analysis (goal - length <= 1500 chars)
- **Evidence-Based Justifications**: Provides reasoning for each score

### 4. Fine-Tuning Pipeline (`train/finetune.py`)

Simplified SFT training workflow:
- **`prepare_and_train()`**: Single function handling:
  - Data preparation and validation
  - Cost estimation (SFT-training rates)
  - Job submission to OpenAI
  - Real-time loss monitoring with moving average plots (last 100 samples)
  - Model ID retrieval upon completion
- **`get_model_info()`**: Retrieves fine-tuned model metadata
- **Assumptions**: Validated 64K token limit, known JSON structure

>> It is recommended for this task (SFT) to use the `gpt-4.1` family.

### 5. Benchmarking Suite (`evaluation/benchmark.py`)

Comprehensive evaluation pipeline with:
- **Multi-Model Comparison**: Tests multiple models in sequence
- **Special Baseline Handling**: Loads pre-existing baseline summaries for comparison
- **Three-Stage Pipeline**:
  1. **Inference**: Generate summaries with latency tracking
  2. **Persistence**: Save results with timestamps
  3. **Judging**: LLM-as-judge evaluation with cost tracking
- **Statistical Aggregation**: Pandas-based averaging across all dimensions
- **Cost Analysis**: Per-1K request cost estimates

## Benchmark Methodology

### Data Preparation

1. **Baseline Dataset**: 1,000 web pages with original summaries
2. **Gold Standard Creation**: GPT-5.2 generates high-quality reference summaries (constrained to ≤1500 characters)
3. **Train/Validation Split**: 
   - Training set: Up to 50% of original dataset from samples meeting BOTH constraints (≤1500 chars AND ≤64K context tokens)
   - Validation set: ALL remaining samples (including those >1500 chars or >64K tokens)

>> This split ensures training data meets fine-tuning requirements while validation includes diverse, challenging examples. Summary length filtering prevents inheriting the base model's tendency for overly long summaries.

5. **Benchmark Subset**: ~20% of validation set (99 items) under 250K tokens for simple, cheaper benchmark used for model selection. 

### Fine-Tuning Process

- **Models**: GPT-4.1-mini, GPT-4.1-nano
- **Training Data**: training gold standard examples (~500)
- **Epochs**: 3
- **Objective**: Distill GPT-5.2 performance into smaller models



### Metrics Tracked

| Metric | Description | Unit |
|--------|-------------|------|
| **Relevance** | Captures key information | 1-5 scale |
| **Faithfulness** | No hallucinations | 1-5 scale |
| **Coherence** | Logical structure | 1-5 scale |
| **Fluency** | Grammar quality | 1-5 scale |
| **Conciseness** | Information density | 1-5 scale |
| **Quality** | Overall quality score (avg of 5 dimensions) | 1-5 scale |
| **Latency** | Inference time | Seconds |
| **Cost per 1K** | Estimated API cost | USD per 1,000 requests |

## Benchmark Results

### Model Comparison on Evaluation Benchmark

<div style="width:100%; max-width:1200px;">
  <img src="images/benchmark_results.png" alt="Benchmark results table" style="width:100%; height:auto; display:block; margin: 0 auto;" />
</div>


### Key Findings

- **Quality Rankings (1-5 scale)**:
  - 🥇 **GPT-5.2**: 4.63 (gold standard / teacher)
  - 🥈 **FT GPT-4.1-mini**: 4.60 (distilled model)
  - 🥉 **GPT-5-mini**: 4.58

- **Latency**:
  - 🥇 **FT GPT-4.1-nano**: 4.90s (distilled model)
  - 🥈 **GPT-4o-mini**: 8.76s
  - 🥉 **FT GPT-4.1-mini**: 13.62s (distilled model)

- **Cost-Performance (1k requests)**:
  - 🥇 **GPT-4o-mini**: 1.24$
  - 🥈 **FT GPT-4.1-nano**: 2.30$ (distilled model)
  - 🥉 **GPT-5-nano**: 2.43$

- **Length Adherence (1-5, higher is better)**
  - ✅ **GPT-4o-mini**: 4.95 (best)
  - ⚠️ **FT GPT-4.1-nano**: 4.12 (distilled model)
  - ⚠️ **GPT-5.2**: 4.37 (gold standard / teacher)
  - ⚠️ **FT GPT-4.1-mini**: 4.41 (distilled model)
  - ⚠️ **GPT-5-nano**: 4.48
  - Note: Lower adherence increases retries (up to `max_retries=3`), nudging cost and latency up.

- **Distillation Effectiveness**
  - 💎 **FT GPT-4.1-mini**: ~99% of teacher quality; cost ↓64% (≈1.78× better than base); closes 87% of the quality gap to the teacher.
  - ⚡ **FT GPT-4.1-nano**: ~94% of teacher quality; cost ↓91% (≈1.68× better than base); closes 60% of the gap while staying very fast.

## Installation & Usage

### Prerequisites

```bash
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file or set environment variable:
```bash
OPENAI_API_KEY=your_api_key_here
```

### Running Tests

The project includes comprehensive unit tests for all agents and core functionality:

```bash
# Run all tests
python -m pytest tests/ -v

# Or using unittest
python -m unittest discover tests/ -v

# Run specific test file
python -m unittest tests.test_agents -v
```

## Limitations

The G-Eval benchmark relies on the underlying LLM capabilities and was not formally evaluated using human annotators to validate it's accuracy. Internal validation of the judge CoT is also important for production grade, ensuring scoring variance, criteria adherance etc. In addition, this benchmark was conducted on a relatively small subset of the data, for cost and time considerations, making it stable enough, yet a larger scale validation may benefit with it.

# Additional Production Improvements
1. Batching the benchmarkings (making them not live) can save ~50% of the evaluation cost.
2. Multiprocessing / async can cut multi requests running time as the major bottleneck is requests. This cannot be performed during benchmarking as rate limits and queue delays affect the latency.
