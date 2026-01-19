import os

# Local Code
from agents.llm import LLMEngine
from agents.config import PROMPT_REPO_PATH

class Summarizer:
    """
    Wrapper agent around an LLM engine that summarizes web page content using OpenAI API.
    """
    def __init__(self, model="gpt-5.2", prompt_repo_path=PROMPT_REPO_PATH):
        # Initialize LLM engine
        self.engine = LLMEngine(
            model=model
        )
        self.prompt_repo = prompt_repo_path

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

    def summarize(self, content, get_cost=True):
        """Standardizes the summarization request format."""
        # Inject data into templates
        user_prompt = self.user_template.format(
            source=content 
        )
        messages = [
            {"role": "system", "content": self.system_template},
            {"role": "user", "content": user_prompt}
        ]
        # Calls generate from your existing LLMEngine
        # Summary attempt input reduction for cost / time savings
        return self.engine.generate(messages, reduce_input=True, get_cost=get_cost)