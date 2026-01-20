import os
import re
import tiktoken
from openai import OpenAI

# Local Code
from agents.config import RATES, CONTEXT_WINDOW


class LLMEngine:
    """
    A wrapper around OpenAI's LLM with token counting and cost calculation.
        1. Token counting using tiktoken.
        2. Cost calculation based on predefined rates.
        3. Single request handling with integrated cost tracking.
    """
    def __init__(self, api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o", rates=RATES):
        # Validate API key
        if not api_key or not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("A valid non-empty OpenAI API key must be provided.")

        # Validate rates and model
        if not rates or not isinstance(rates, dict):
            raise ValueError("A valid rates dictionary must be provided.")
        if model not in rates:
            raise ValueError(f"Your RATES table is not updated for your model selection ({model}), please update it accordingly.")
        self.rates = rates

        # Validate model family: only 4o, 4.1, o1-o4, or >=5
        self.model = model
        self._validate_model_family()
        self.context_limit = CONTEXT_WINDOW.get(model, 128000)  # Default to 128k if not found
        
        # Determine if model supports temperature parameter
        self.supports_temperature = self._check_temperature_support()

        # Initialize OpenAI client and tokenizer, catch errors early
        try:
            self.client = OpenAI(api_key=api_key)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")
        try:
            self.tokenizer = tiktoken.encoding_for_model(model)
        except KeyError:
            # For future/unknown models, find the closest known tokenizer
            fallback_model = self._find_closest_tokenizer(model)
            print(f"Warning: Model '{model}' not recognized by tiktoken. Using '{fallback_model}' tokenizer as fallback.")
            self.tokenizer = tiktoken.encoding_for_model(fallback_model)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize tokenizer for model '{model}': {e}")

    def _find_closest_tokenizer(self, model):
        """Find the closest available tokenizer for an unknown model by trying progressively simpler versions."""
        model_lower = model.lower()
        
        # Fine-tuned models: extract base model from "ft:base-model:org::id" format
        if model_lower.startswith("ft:"):
            parts = model_lower.split(":")
            if len(parts) >= 2:
                model_lower = parts[1]  # Get the base model name
        
        candidates = []
        
        # Extract model family and version to generate fallback candidates
        # For gpt-5.2-2025-12-11 → try gpt-5.2, gpt-5.1, gpt-5, gpt-4o
        # For gpt-4.1-2025-04-14 → try gpt-4.1, gpt-4o
        # For o1-mini → try o1, gpt-4o
        
        # Check for GPT models (gpt-X.Y-date or gpt-X-variant)
        gpt_match = re.match(r'^(gpt-(\d+)(?:\.(\d+))?)(?:-.*)?$', model_lower)
        if gpt_match:
            base = gpt_match.group(1)  # e.g., "gpt-5.2" or "gpt-5"
            major = int(gpt_match.group(2))  # e.g., 5
            minor = gpt_match.group(3)  # e.g., "2" or None
            
            # Add the base version without date
            candidates.append(base)
            
            # If there's a minor version, try decrementing it
            if minor:
                minor_int = int(minor)
                for m in range(minor_int - 1, -1, -1):
                    candidates.append(f"gpt-{major}.{m}")
            
            # Try the major version without minor
            candidates.append(f"gpt-{major}")
            
            # For gpt-5+, also try lower major versions
            if major >= 5:
                for maj in range(major - 1, 4, -1):
                    candidates.append(f"gpt-{maj}")
        
        # Check for o-series models (o1-mini, o2, etc.)
        o_match = re.match(r'^(o(\d+))(?:-.*)?$', model_lower)
        if o_match:
            base = o_match.group(1)  # e.g., "o1"
            candidates.append(base)
        
        # Always fall back to gpt-4o as the final option
        candidates.append("gpt-4o")
        
        # Try each candidate until one works
        for candidate in candidates:
            try:
                tiktoken.encoding_for_model(candidate)
                return candidate
            except KeyError:
                continue
        
        # If nothing works, return gpt-4o (should always work)
        return "gpt-4o"
    
    def _validate_model_family(self):
        """Validates that model belongs to supported families: 4o, 4.1, o1-o4, or >=5."""
        model = self.model.lower()
        
        # Fine-tuned models: extract base model from "ft:base-model:org::id" format
        if model.startswith("ft:"):
            # Extract base model (e.g., "ft:gpt-4.1-mini-2025-04-14:..." -> "gpt-4.1-mini-2025-04-14")
            parts = model.split(":")
            if len(parts) >= 2:
                model = parts[1]  # Get the base model name
        
        # Check for o1-o4 series (e.g., o1, o1-mini, o1-preview, o2, o3, o4)
        if re.match(r'^o[1-4](-.*)?$', model):
            return
        
        # Check for gpt-4o variants (e.g., gpt-4o, gpt-4o-mini)
        if re.match(r'^gpt-4o(-.*)?$', model):
            return
        
        # Check for gpt-4.1 variants
        if re.match(r'^gpt-4\.1(-.*)?$', model):
            return
        
        # Check for gpt-5 or higher (e.g., gpt-5, gpt-5.2, gpt-6...)
        gpt_match = re.match(r'^gpt-(\d+)', model)
        if gpt_match and int(gpt_match.group(1)) >= 5:
            return
        
        raise ValueError(
            f"Model '{self.model}' is not supported. Only models from families "
            f"4o (gpt-4o*), 4.1 (gpt-4.1*), o1-o4 (o1*, o2*, o3*, o4*), or >=5 (gpt-5+) are allowed."
        )
    
    def _check_temperature_support(self):
        """Returns True only for 4o and 4.1 variants that support temperature."""
        model = self.model.lower()
        
        # Fine-tuned models: extract base model from "ft:base-model:org::id" format
        if model.startswith("ft:"):
            parts = model.split(":")
            if len(parts) >= 2:
                model = parts[1]  # Get the base model name
        
        # Only gpt-4o and gpt-4.1 variants support temperature
        if re.match(r'^gpt-4o(-.*)?$', model):
            return True
        if re.match(r'^gpt-4\.1(-.*)?$', model):
            return True
        # All other models (o1-o4, gpt-5+) don't support temperature
        return False

    def _minify_content(
        self, 
        text, 
        strip_urls=True, 
        strip_html=True, 
        simplify_tables=True, 
        strip_navigation=True, 
        normalize_whitespace=True
    ):
        """
        Internal helper to reduce token count by stripping non-essential content.
        
        Args:
            text (str): The raw markdown content to process.
            strip_urls (bool): If True, replaces [label](url) with just [label].
            strip_html (bool): If True, removes common HTML tags like <div>, <span>, etc.
            simplify_tables (bool): If True, removes markdown table formatting (pipes and dashes).
            strip_navigation (bool): If True, removes common navigational boilerplate (e.g., 'Follow us').
            normalize_whitespace (bool): If True, collapses multiple spaces and newlines into one.
        """
        if strip_html:
            text = re.sub(r'<[^>]+>', '', text)

        if strip_urls:
            # Matches markdown links and keeps only the anchor text
            text = re.sub(r'\[(.*?)\]\(.*?\)', r'[\1]', text)

        if simplify_tables:
            # Removes pipe characters and markdown table dividers
            text = re.sub(r'\|', ' ', text)
            text = re.sub(r'[-:]{3,}', '', text)

        if strip_navigation:
            # Removes horizontal rules and common social/nav footers
            text = re.sub(r'={3,}', '', text)
            text = re.sub(r'[-*_]{3,}', '', text)
            text = re.sub(r'(Follow us|Share via|About us).*$', '', text, flags=re.IGNORECASE | re.MULTILINE)

        if normalize_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()
            
        return text

    def count_tokens(self, text):
        """Standard token counting using the model's specific encoding."""
        return len(self.tokenizer.encode(text))

    def get_cost(self, prompt_tokens, completion_tokens):
        """Calculates USD cost based on current model rates."""
        rate = self.rates[self.model] # Rates already validated in __init__, no KeyError expected
        return (prompt_tokens / 1e6 * rate["input"]) + (completion_tokens / 1e6 * rate["output"])

    def generate(self, messages, temperature=0.0, response_format=None, get_cost=False, get_tokens_count=False, reduce_input=False):
        """Single request handler with integrated cost tracking."""
        # Optionally reduce input size for cost / time savings
        if reduce_input:
            for msg in messages:
                if msg['role'] in ['user']:
                    msg['content'] = self._minify_content(
                        msg['content'],
                        strip_urls=True,
                        strip_html=True,
                        simplify_tables=True,
                        strip_navigation=True,
                        normalize_whitespace=True
                    )

        # Build API call parameters, excluding temperature if not supported
        api_params = {
            "model": self.model,
            "messages": messages,
        }
        
        # Only include temperature if model supports it
        if self.supports_temperature:
            api_params["temperature"] = temperature
        
        if response_format is not None:
            api_params["response_format"] = response_format
        
        try:
            response = self.client.chat.completions.create(**api_params)
        except Exception as e:
            # Print warning and skip this request
            print(f"Warning: Request failed with error: {e}")
            # Return None values based on what was requested
            if get_cost and get_tokens_count:
                return None, 0.0, (0, 0)
            elif get_cost:
                return None, 0.0
            elif get_tokens_count:
                return None, (0, 0)
            else:
                return None
        
        if get_cost or get_tokens_count:
            cost = self.get_cost(response.usage.prompt_tokens, response.usage.completion_tokens)
            tokens_count = (response.usage.prompt_tokens, response.usage.completion_tokens)
            if get_cost and get_tokens_count:
                return response.choices[0].message.content, cost, tokens_count
            elif get_cost:
                return response.choices[0].message.content, cost
            elif get_tokens_count:
                return response.choices[0].message.content, tokens_count
        else:
            return response.choices[0].message.content