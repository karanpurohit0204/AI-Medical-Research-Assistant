import os
from pathlib import Path


def test_groq_model_is_loaded_from_repo_env(tmp_path, monkeypatch):
    project_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    os.environ.pop("GROQ_MODEL", None)

    from backend.core.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()

    assert settings.groq_model == "llama-3.3-70b-versatile"
    assert (project_root / ".env").exists()
