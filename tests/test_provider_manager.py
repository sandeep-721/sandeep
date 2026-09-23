from config.llm_config import LLMConfig
from generation.provider_manager import LLMProviderManager


class FakeProvider:
    def __init__(self):
        self.closed = False

    def generate(self, prompt, max_new_tokens=512, temperature=0.1):
        return "fake answer"

    def list_models(self):
        return ["fake-model"]

    def validate_connection(self):
        return True

    def close(self):
        self.closed = True


def test_manager_signature_changes_when_api_key_changes():
    manager = LLMProviderManager()

    config_a = LLMConfig(
        provider="openai_compatible",
        model="model-a",
        base_url="http://localhost:9000/v1",
        api_key_env="TEST_KEY",
    )

    config_b = LLMConfig(
        provider="openai_compatible",
        model="model-a",
        base_url="http://localhost:9000/v1",
        api_key_env="TEST_KEY",
    )

    assert manager._signature_for(config_a) == manager._signature_for(config_b)


def test_manager_signature_changes_when_model_changes():
    manager = LLMProviderManager()

    config_a = LLMConfig(
        provider="ollama",
        model="qwen3",
        base_url="http://127.0.0.1:11434/v1",
    )

    config_b = LLMConfig(
        provider="ollama",
        model="llama3",
        base_url="http://127.0.0.1:11434/v1",
    )

    assert manager._signature_for(config_a) != manager._signature_for(config_b)
