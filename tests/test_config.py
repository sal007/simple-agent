from simple_agent.config import resolve


def test_defaults_to_lm_studio():
    s = resolve({})
    assert (s.provider, s.base_url) == ("openai", "http://localhost:1234/v1")


def test_flags_override_file():
    file_config = {"provider": "openai", "providers": {"anthropic": {"model": "from-file"}}}
    assert resolve(file_config, provider="anthropic").model == "from-file"
    assert resolve(file_config, provider="anthropic", model="from-flag").model == "from-flag"


def test_env_key_wins(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    s = resolve({"providers": {"anthropic": {"api_key": "file-key"}}}, provider="anthropic")
    assert s.api_key == "env-key"
