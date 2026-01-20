import os
import json
import re

# Local Code
from agents.llm import LLMEngine
from agents.config import PROMPT_REPO_PATH


class SummarizationJudge:
    """
    An agent that evaluates the quality of summaries using an LLM.
    Mocks the G-Eval approach, using prompt templates for standardized evaluation.

    NOTE: The prompt template and JSON schema determines the evaluation criteria.
    """
    def __init__(self, model="gpt-5.2", prompt_repo_path=PROMPT_REPO_PATH):
        # Validate model family for structured output (must be 5 or higher)
        model_family_match = re.search(r"gpt-(\d+)", model)
        if not model_family_match or int(model_family_match.group(1)) < 5:
            raise ValueError(f"Structured output with schema requires a model family of 5 or higher (got '{model}').")

        # Initialize LLM engine
        self.engine = LLMEngine(
            model=model
        )
        self.prompt_repo = prompt_repo_path

        # Load prompt templates
        self.system_template = self._load_template("judge_system.txt")
        self.user_template = self._load_template("judge_user.txt")
        if '{source}' not in self.user_template or '{summary}' not in self.user_template:
            raise ValueError("User prompt template must contain '{source}' and '{summary}' placeholders.")
        
        # Load output schema for structured response
        try:
            with open(os.path.join(self.prompt_repo, "judge_schema.json"), "r", encoding="utf-8") as f:
                schema_data = json.load(f)
                # Extract just the schema object from the full response_format structure
                self.output_schema = schema_data["json_schema"]["schema"]
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON schema file not found: {os.path.join(self.prompt_repo, 'judge_schema.json')}")
        except json.JSONDecodeError:
            raise ValueError(f"JSON schema file is not a valid JSON: {os.path.join(self.prompt_repo, 'judge_schema.json')}")
        except KeyError:
            raise ValueError(f"JSON schema file has unexpected structure: {os.path.join(self.prompt_repo, 'judge_schema.json')}")

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

    def evaluate(self, source_content, candidate_summary, get_cost=False, length_limit=1500):
        """
        Assembles the evaluation prompt and calls the LLM engine.
        """
        # Inject data into templates
        user_prompt = self.user_template.format(
            source=source_content, 
            summary=candidate_summary
        )

        messages = [
            {"role": "system", "content": self.system_template},
            {"role": "user", "content": user_prompt}
        ]

        # Use the engine for the actual API call, optionally retrieving cost
        # Will return output and cost if get_cost is True, else just output
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "summary_evaluation",
                "schema": self.output_schema,
                "strict": True
            }
        }
        
        if get_cost:
            raw_output, cost = self.engine.generate(
                messages=messages, 
                response_format=response_format,
                get_cost=True
            )
        else:
            raw_output = self.engine.generate(
                messages=messages, 
                response_format=response_format,
                get_cost=False
            )
        
        # Handle failed requests
        if raw_output is None:
            if get_cost:
                return None, 0.0
            return None
        
        # Check if candidate_summary is None (failed inference)
        if candidate_summary is None:
            print(f"Warning: Cannot evaluate None summary")
            if get_cost:
                return None, 0.0
            return None
        
        # Add length score to the evaluation
        evaluation = json.loads(raw_output)
        char_length = len(candidate_summary)
        evaluation["justifications"]["length"] = f"Longer than {length_limit} chars limit" if char_length > length_limit else f"Within limit {length_limit} chars limit"
        evaluation["scores"]["length"] = 5 if char_length <= length_limit else 1

        if get_cost:
            return evaluation, cost
        return evaluation