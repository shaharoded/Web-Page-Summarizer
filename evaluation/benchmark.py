import json
import time
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# Local code
from agents.summarizer import Summarizer
from agents.judge import SummarizationJudge


class BenchmarkingSuite:
    def __init__(self, rates, judge_model="gpt-5.2-2025-12-11"):
        self.rates = rates
        self.judge = SummarizationJudge(model=judge_model)
        self.results = []

    def run_benchmark(self, eval_data, model_list):
        """
        Runs the full flow: Inference -> Saving -> Judging -> Tabulating.
        Special handling: if model_name is "Baseline", loads pre-existing summaries instead of running inference.
        """
        all_model_stats = []
        total_judge_cost = 0.0
        total_models = len(model_list)

        for model_idx, model_name in enumerate(model_list, 1):
            print(f"\n{'='*70}")
            print(f"Model {model_idx}/{total_models}: {model_name}")
            print(f"{'='*70}")
            
            model_inferences = []
            model_costs = 0.0
            latencies = []

            # Special handling for "Baseline" model
            if model_name.lower() == "baseline":
                print("Loading baseline summaries from data/baseline_validation_benchmark.json")
                with open('data/baseline_validation_benchmark.json', 'r', encoding='utf-8') as f:
                    baseline_data = json.load(f)
                
                # Filter baseline to match only URLs in eval_data
                eval_urls = {item['url'] for item in eval_data}
                baseline_filtered = [item for item in baseline_data if item['url'] in eval_urls]
                
                print(f"✓ Filtered to {len(baseline_filtered)} baseline summaries matching eval subset")
                
                # Convert baseline format to inference format
                for item in baseline_filtered:
                    model_inferences.append({
                        "url": item['url'],
                        "source": item['markdown_content'],
                        "summary": item['summary'],
                        "latency": 0.0  # Baseline has no latency
                    })
                
                print(f"✓ Loaded {len(model_inferences)} baseline summaries")
                
                # Save baseline inferences for record keeping
                filename = f"data/inference_baseline.json"
                with open(filename, 'w') as f:
                    json.dump(model_inferences, f)
                print(f"✓ Saved to {filename}")
            else:
                # Normal inference flow for non-baseline models
                summarizer = Summarizer(model=model_name)
                
                # 1. Sequential Inference & Latency Tracking with progress bar
                for item in tqdm(eval_data, desc=f"[{model_idx}/{total_models}] Inference", unit="sample"):
                    start_time = time.perf_counter()
                    summary, cost = summarizer.summarize(item['markdown_content'], get_cost=True)
                    end_time = time.perf_counter()
                    
                    latency = end_time - start_time
                    latencies.append(latency)
                    model_costs += cost
                    
                    model_inferences.append({
                        "url": item['url'],
                        "source": item['markdown_content'],
                        "summary": summary,
                        "latency": round(latency, 3)
                    })

                    time.sleep(1.0)  # To respect rate limits

                # 2. Save Inference to JSON
                filename = f"data/inference_{model_name.replace(':', '_')}.json"
                with open(filename, 'w') as f:
                    json.dump(model_inferences, f)
                print(f"✓ Saved to {filename}")

            # 3. Batch Judging with progress bar
            model_scores = []
            for inf in tqdm(model_inferences, desc=f"[{model_idx}/{total_models}] Judging", unit="sample"):
                eval_result, j_cost = self.judge.evaluate(inf['source'], inf['summary'], get_cost=True)
                total_judge_cost += j_cost
                
                # Skip if judge evaluation failed
                if eval_result is None:
                    print(f"\n⚠ Warning: Judge evaluation failed for {inf['url']}, skipping...")
                    continue
                
                # Merge judge scores with latency
                res = eval_result['scores']
                res['latency'] = inf['latency']
                
                # Calculate quality score per evaluation (average of 5 judge dimensions)
                judge_dimensions = ['relevance', 'faithfulness', 'coherence', 'fluency', 'conciseness']
                res['quality'] = sum(res[dim] for dim in judge_dimensions) / len(judge_dimensions)
                
                model_scores.append(res)

                time.sleep(1.0)  # To respect rate limits

            # 4. Aggregate Stats
            df_model = pd.DataFrame(model_scores)
            stats = df_model.mean().to_dict()
            stats['model'] = model_name
            
            # Calculate cost per 1k (0 for baseline)
            if model_name.lower() == "baseline":
                stats['est_cost_per_1k'] = 0.0
            else:
                stats['est_cost_per_1k'] = (model_costs / len(eval_data)) * 1000
            
            all_model_stats.append(stats)

        print(f"\n{'='*70}")
        print(f"Benchmark Finished. Total Judge Cost: ${total_judge_cost:.4f}")
        print(f"{'='*70}")
        return pd.DataFrame(all_model_stats)