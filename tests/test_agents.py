import unittest
import os
import sys
from unittest.mock import Mock, patch, MagicMock

# Set API key for tests before importing agents
os.environ['OPENAI_API_KEY'] = 'test-api-key-for-testing'

# Add parent directory to path to import agents
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agents.llm import LLMEngine
from agents.config import RATES


class TestLLMEngine(unittest.TestCase):
    """Test suite for LLMEngine validation and functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.api_key = "test-api-key-12345"
        
    def test_invalid_api_key(self):
        """Test that invalid API keys raise ValueError."""
        with self.assertRaises(ValueError) as context:
            LLMEngine(api_key="")
        self.assertIn("valid non-empty OpenAI API key", str(context.exception))
        
        with self.assertRaises(ValueError):
            LLMEngine(api_key=None)
        
        with self.assertRaises(ValueError):
            LLMEngine(api_key="   ")
    
    def test_invalid_rates(self):
        """Test that invalid rates raise ValueError."""
        with self.assertRaises(ValueError) as context:
            LLMEngine(api_key=self.api_key, rates=None)
        self.assertIn("valid rates dictionary", str(context.exception))
        
        with self.assertRaises(ValueError):
            LLMEngine(api_key=self.api_key, rates={})
    
    def test_model_not_in_rates(self):
        """Test that models not in RATES table raise ValueError."""
        with self.assertRaises(ValueError) as context:
            LLMEngine(api_key=self.api_key, model="gpt-3.5-turbo")
        self.assertIn("RATES table is not updated", str(context.exception))
    
    def test_unsupported_model_families(self):
        """Test that unsupported model families raise ValueError."""
        unsupported_models = [
            "gpt-3.5-turbo",
            "gpt-4-turbo",
            "gpt-4",
            "claude-3",
            "o5",
            "random-model"
        ]
        
        for model in unsupported_models:
            with self.assertRaises(ValueError) as context:
                # Mock to pass rates check
                test_rates = RATES.copy()
                test_rates[model] = {"input": 1.0, "output": 1.0}
                LLMEngine(api_key=self.api_key, model=model, rates=test_rates)
            self.assertIn("not supported", str(context.exception).lower())
    
    @patch('agents.llm.OpenAI')
    @patch('agents.llm.tiktoken.encoding_for_model')
    def test_supported_4o_variants(self, mock_tiktoken, mock_openai):
        """Test that gpt-4o variants are supported."""
        mock_tiktoken.return_value = Mock()
        mock_openai.return_value = Mock()
        
        supported_4o = ["gpt-4o", "gpt-4o-mini"]
        
        for model in supported_4o:
            engine = LLMEngine(api_key=self.api_key, model=model)
            self.assertEqual(engine.model, model)
            self.assertTrue(engine.supports_temperature, 
                          f"{model} should support temperature")
    
    @patch('agents.llm.OpenAI')
    @patch('agents.llm.tiktoken.encoding_for_model')
    def test_supported_4_1_variants(self, mock_tiktoken, mock_openai):
        """Test that gpt-4.1 variants are supported."""
        mock_tiktoken.return_value = Mock()
        mock_openai.return_value = Mock()
        
        supported_4_1 = [
            "gpt-4.1-2025-04-14",
            "gpt-4.1-mini-2025-04-14", 
            "gpt-4.1-nano-2025-04-14"
        ]
        
        for model in supported_4_1:
            engine = LLMEngine(api_key=self.api_key, model=model)
            self.assertEqual(engine.model, model)
            self.assertTrue(engine.supports_temperature,
                          f"{model} should support temperature")
    
    @patch('agents.llm.OpenAI')
    @patch('agents.llm.tiktoken.encoding_for_model')
    def test_supported_o_series(self, mock_tiktoken, mock_openai):
        """Test that o1-o4 series are supported but don't support temperature."""
        mock_tiktoken.return_value = Mock()
        mock_openai.return_value = Mock()
        
        # Add o1-mini to rates for testing
        test_rates = RATES.copy()
        test_rates["o1"] = {"input": 1.0, "output": 4.0}
        test_rates["o2"] = {"input": 1.0, "output": 4.0}
        test_rates["o3-mini"] = {"input": 1.0, "output": 4.0}
        test_rates["o4"] = {"input": 1.0, "output": 4.0}
        
        o_series = ["o1-mini", "o1", "o2", "o3-mini", "o4"]
        
        for model in o_series:
            engine = LLMEngine(api_key=self.api_key, model=model, rates=test_rates)
            self.assertEqual(engine.model, model)
            self.assertFalse(engine.supports_temperature,
                           f"{model} should NOT support temperature")
    
    @patch('agents.llm.OpenAI')
    @patch('agents.llm.tiktoken.encoding_for_model')
    def test_supported_gpt5_variants(self, mock_tiktoken, mock_openai):
        """Test that gpt-5+ variants are supported but don't support temperature."""
        mock_tiktoken.return_value = Mock()
        mock_openai.return_value = Mock()
        
        # Add test models to rates
        test_rates = RATES.copy()
        test_rates["gpt-6"] = {"input": 2.0, "output": 10.0}
        test_rates["gpt-10-turbo"] = {"input": 2.0, "output": 10.0}
        
        gpt5_series = ["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5.2-2025-12-11", "gpt-6", "gpt-10-turbo"]
        
        for model in gpt5_series:
            engine = LLMEngine(api_key=self.api_key, model=model, rates=test_rates)
            self.assertEqual(engine.model, model)
            self.assertFalse(engine.supports_temperature,
                           f"{model} should NOT support temperature")
    
    @patch('agents.llm.OpenAI')
    @patch('agents.llm.tiktoken.encoding_for_model')
    def test_generate_with_temperature_support(self, mock_tiktoken, mock_openai):
        """Test that temperature is included for 4o/4.1 models."""
        mock_tiktoken.return_value = Mock()
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        mock_response.usage = Mock(prompt_tokens=100, completion_tokens=50)
        mock_client.chat.completions.create.return_value = mock_response
        
        engine = LLMEngine(api_key=self.api_key, model="gpt-4o")
        result = engine.generate([{"role": "user", "content": "test"}], temperature=0.7)
        
        # Verify temperature was passed
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertIn("temperature", call_kwargs)
        self.assertEqual(call_kwargs["temperature"], 0.7)
    
    @patch('agents.llm.OpenAI')
    @patch('agents.llm.tiktoken.encoding_for_model')
    def test_generate_without_temperature_support(self, mock_tiktoken, mock_openai):
        """Test that temperature is NOT included for o-series and gpt-5+ models."""
        mock_tiktoken.return_value = Mock()
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        mock_response.usage = Mock(prompt_tokens=100, completion_tokens=50)
        mock_client.chat.completions.create.return_value = mock_response
        
        # Test o1-mini (no temperature support)
        engine = LLMEngine(api_key=self.api_key, model="o1-mini")
        result = engine.generate([{"role": "user", "content": "test"}], temperature=0.7)
        
        # Verify temperature was NOT passed
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertNotIn("temperature", call_kwargs)
        
        # Test gpt-5 (no temperature support)
        engine = LLMEngine(api_key=self.api_key, model="gpt-5")
        result = engine.generate([{"role": "user", "content": "test"}], temperature=0.7)
        
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertNotIn("temperature", call_kwargs)


class TestModelInference(unittest.TestCase):
    """Integration tests to verify all models in config can be used for inference."""
    
    @patch('agents.llm.OpenAI')
    def test_all_config_models_can_initialize(self, mock_openai):
        """Test that all models in RATES config can initialize LLMEngine."""
        # Don't mock tiktoken - we want to test real tokenizer initialization
        mock_openai.return_value = Mock()
        
        api_key = "test-api-key-12345"
        
        for model in RATES.keys():
            with self.subTest(model=model):
                try:
                    engine = LLMEngine(api_key=api_key, model=model)
                    self.assertIsNotNone(engine)
                    self.assertEqual(engine.model, model)
                    self.assertIsNotNone(engine.tokenizer)
                except Exception as e:
                    self.fail(f"Model {model} from config failed to initialize: {e}")
    
    @patch('agents.llm.OpenAI')
    def test_all_config_models_can_generate(self, mock_openai):
        """Test that all models in RATES config can call generate()."""
        # Don't mock tiktoken - we want to test real tokenizer initialization
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test summary"))]
        mock_response.usage = Mock(prompt_tokens=100, completion_tokens=50)
        mock_client.chat.completions.create.return_value = mock_response
        
        api_key = "test-api-key-12345"
        test_messages = [{"role": "user", "content": "Summarize this text"}]
        
        for model in RATES.keys():
            with self.subTest(model=model):
                try:
                    engine = LLMEngine(api_key=api_key, model=model)
                    result = engine.generate(test_messages, temperature=0.5)
                    self.assertIsNotNone(result)
                    self.assertEqual(result, "Test summary")
                except Exception as e:
                    self.fail(f"Model {model} failed to generate: {e}")


class TestSummarizerWithAllModels(unittest.TestCase):
    """Test Summarizer agent with all config models."""
    
    @patch('agents.llm.OpenAI')
    @patch('agents.summarizer.Summarizer._load_template')
    def test_summarizer_with_all_models(self, mock_load_template, mock_openai):
        """Test that Summarizer can be initialized and called with all config models."""
        from agents.summarizer import Summarizer
        
        # Mock template loading
        mock_load_template.side_effect = lambda filename: (
            "You are a summarizer" if "system" in filename else "Summarize: {source}"
        )
        
        # Don't mock tiktoken - we want to test real tokenizer initialization
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Summary output"))]
        mock_response.usage = Mock(prompt_tokens=100, completion_tokens=50)
        mock_client.chat.completions.create.return_value = mock_response
        
        test_content = "This is test content to summarize."
        
        for model in RATES.keys():
            with self.subTest(model=model):
                try:
                    summarizer = Summarizer(model=model)
                    result = summarizer.summarize(test_content, get_cost=False)
                    self.assertIsNotNone(result)
                except Exception as e:
                    self.fail(f"Summarizer with model {model} failed: {e}")


class TestJudgeWithAllModels(unittest.TestCase):
    """Test SummarizationJudge agent with all applicable config models."""
    
    @patch('agents.llm.OpenAI')
    @patch('agents.judge.SummarizationJudge._load_template')
    @patch('builtins.open', create=True)
    def test_judge_with_gpt5_models(self, mock_open, mock_load_template, mock_openai):
        """Test that Judge works with gpt-5+ models (requires gpt-5+)."""
        from agents.judge import SummarizationJudge
        
        # Mock template loading
        mock_load_template.side_effect = lambda filename: (
            "You are a judge" if "system" in filename 
            else "Source: {source}\nSummary: {summary}"
        )
        
        # Mock JSON schema loading
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = '{"type": "object"}'
        mock_open.return_value = mock_file
        
        # Don't mock tiktoken - we want to test real tokenizer initialization
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content='{"scores": {"coherence": 5}, "justifications": {"coherence": "good"}}'))]
        mock_response.usage = Mock(prompt_tokens=100, completion_tokens=50)
        mock_client.chat.completions.create.return_value = mock_response
        
        # Only test gpt-5+ models (Judge requires gpt-5+)
        gpt5_models = [m for m in RATES.keys() if m.startswith("gpt-5")]
        
        test_source = "Original content here."
        test_summary = "Summary of the content."
        
        for model in gpt5_models:
            with self.subTest(model=model):
                try:
                    judge = SummarizationJudge(model=model)
                    result = judge.evaluate(test_source, test_summary, get_cost=False)
                    self.assertIsNotNone(result)
                except Exception as e:
                    self.fail(f"Judge with model {model} failed: {e}")
    
    @patch('agents.llm.OpenAI')
    def test_judge_rejects_non_gpt5_models(self, mock_openai):
        """Test that Judge raises error for models below gpt-5."""
        from agents.judge import SummarizationJudge
        
        # Don't mock tiktoken - we want to test real tokenizer initialization
        mock_openai.return_value = Mock()
        
        # Models that should be rejected (not gpt-5+)
        non_gpt5_models = [m for m in RATES.keys() if not m.startswith("gpt-5")]
        
        for model in non_gpt5_models:
            with self.subTest(model=model):
                with self.assertRaises(ValueError) as context:
                    judge = SummarizationJudge(model=model)
                self.assertIn("model family of 5 or higher", str(context.exception))


if __name__ == '__main__':
    unittest.main()
