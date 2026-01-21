import os

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
    
    def summarize(self, content, get_cost=True, max_retries=3, allow_long_context=False, reduce_input=True):
        """
        Standardizes the summarization request format.
        Respects max_chars limit via self-correction loop.
        Handles long content via Map-Reduce internally.
        Args:
            content (str): The source content to summarize.
            get_cost (bool): Whether to return cost along with summary.
            max_retries (int): Number of retries for length enforcement.
            allow_long_context (bool): Whether to enable handling of long content via Map-Reduce.
            reduce_input (bool): Whether to minify content before processing.
        Deployment-ready.

        NOTE: This method may make multiple API calls internally to enforce length limits, or if long content handling is enabled, which may increase cost.
        """
        total_cost = 0.0
        
        # Prepare messages
        user_prompt = self.user_template.format(source=content)
        messages = [
            {"role": "system", "content": self.system_template},
            {"role": "user", "content": user_prompt}
        ]
        
        # Generate summary (engine handles map-reduce if needed)
        final_summary, cost = self.engine.generate(
            messages, 
            reduce_input=reduce_input, 
            get_cost=True,
            allow_long_context=allow_long_context
        )
        total_cost += cost
        
        # Handle failed API call early to avoid None downstream
        if final_summary is None:
            if get_cost:
                return None, total_cost
            return None

        # 2. Length Enforcement Loop (Self-Correction)
        retry_count = 0
        while len(final_summary) > self.max_chars and retry_count < max_retries:
            retry_count += 1
            restriction = self.max_chars - (retry_count - 1) * 100 # Every retry make requirement stricter by 100 chars
            
            refine_messages = [
                {"role": "system", "content": self.system_template},
                {"role": "user", "content": f"The following page summary is too long ({len(final_summary)} chars). "
                                           f"Please rewrite it to be under {restriction} characters while "
                                           f"retaining all key facts:\n\n{final_summary}"}
            ]
            final_summary, cost = self.engine.generate(refine_messages, get_cost=True)
            total_cost += cost

            # If refinement fails, abort retries and propagate current state
            if final_summary is None:
                if get_cost:
                    return None, total_cost
                return None

        if get_cost:
            return final_summary, total_cost
        return final_summary