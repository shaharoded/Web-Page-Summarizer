import os
import concurrent.futures

# Local Code
from agents.llm import LLMEngine
from agents.config import PROMPT_REPO_PATH

class Summarizer:
    """
    Wrapper agent around an LLM engine that summarizes web page content using OpenAI API.
    """
    def __init__(self, model="gpt-5.2", prompt_repo_path=PROMPT_REPO_PATH, max_chars=1500):
        # Initialize LLM engine
        self.engine = LLMEngine(
            model=model
        )
        self.prompt_repo = prompt_repo_path
        self.max_chars = max_chars
        self.context_limit = self.engine.context_limit - 5000  # static buffer for prompt overhead, and possible tokenizer inaccuracies
        
        # Load prompt templates
        self.system_template = self._load_template("summarizer_system.txt")
        self.user_template = self._load_template("summarizer_user.txt")
        if '{source}' not in self.user_template:
            raise ValueError("User prompt template must contain '{source}' placeholder.")
    
    def _load_template(self, filename):
        """Retrieves prompt templates from the local repository, raises if not found or empty."""
        path = os.path.join(self.prompt_repo, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Prompt template file not found: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        if not content or not isinstance(content, str) or len(content.strip()) == 0:
            raise ValueError(f"Prompt template '{filename}' is empty or not as expected.")
        return content
    
    def _map_reduce_summarize(self, content):
        """
        Splits large documents into manageable chunks and reduces them.
        Uses parallel execution (ThreadPool) for better performance.
        Chunks based on tokens to respect context limits.
        Uses a small overlap to maintain context across splits.

        NOTE: using this option increases cost due to prompt overhead and multiple jobs.
        NOTE: Overlap size and chunking strategy can be tuned as needed. A static window will work for sentences / links, but not for tables / large sections.
        """
        # 1. Encode content to tokens
        all_tokens = self.engine.tokenizer.encode(content)
        
        # 2. Split tokens into chunks with overlap
        overlap = 50  # Overlap to prevent losing context across splits
        stride = self.context_limit - overlap
        
        token_chunks = [
            all_tokens[i : i + self.context_limit] 
            for i in range(0, len(all_tokens), stride)
        ]
        
        # 3. Decode back to strings (ensures valid text boundaries)
        chunks = [self.engine.tokenizer.decode(chk) for chk in token_chunks]
        
        # Map Phase: Summarize each chunk in parallel
        sub_summaries = []
        total_map_cost = 0.0

        def process_chunk(chunk):
            # No nested retries prevents infinite loops in sub-tasks
            return self.summarize(chunk, get_cost=True, max_retries=0)
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Use map to ensure results are returned in order
            results = executor.map(process_chunk, chunks)
            
            for summary, cost in results:
                if summary:
                    sub_summaries.append(summary)
                    total_map_cost += cost
            
        # Reduce Phase: Combine sub-summaries
        combined_text = "\n\n".join(sub_summaries)
        
        # Final Reduce
        final_summary, reduce_cost = self.summarize(combined_text, get_cost=True)
        return final_summary, total_map_cost + reduce_cost
    
    def summarize(self, content, get_cost=True, max_retries=5, allow_long_context=False):
        """
        Standardizes the summarization request format.
        Respects max_chars limit via self-correction loop.
        Handles long content via Map-Reduce internally.
        Args:
            content (str): The source content to summarize.
            get_cost (bool): Whether to return cost along with summary.
            max_retries (int): Number of retries for length enforcement.
            allow_long_context (bool): Whether to enable handling of long content via Map-Reduce.
        Deployment-ready.
        """
        total_cost = 0.0
        
        # 1. Check for Long Context (Map-Reduce Trigger)
        tokens = self.engine.count_tokens(content)
        if allow_long_context and tokens > self.context_limit:
            final_summary, cost = self._map_reduce_summarize(content)
            total_cost += cost
        else:
            # Standard single-pass summary
            user_prompt = self.user_template.format(source=content)
            messages = [
                {"role": "system", "content": self.system_template},
                {"role": "user", "content": user_prompt}
            ]
            final_summary, cost = self.engine.generate(messages, reduce_input=True, get_cost=True)
            total_cost += cost

        # 2. Length Enforcement Loop (Self-Correction)
        retry_count = 0
        while len(final_summary) > self.max_chars and retry_count < max_retries:
            retry_count += 1
            print(f"Summary length ({len(final_summary)}) exceeds limit. Retrying ({retry_count})...")
            
            refine_messages = [
                {"role": "system", "content": self.system_template},
                {"role": "user", "content": f"The following page summary is too long ({len(final_summary)} chars). "
                                           f"Please rewrite it to be under {self.max_chars} characters while "
                                           f"retaining all key facts:\n\n{final_summary}"}
            ]
            final_summary, cost = self.engine.generate(refine_messages, get_cost=True)
            total_cost += cost

        if get_cost:
            return final_summary, total_cost
        return final_summary