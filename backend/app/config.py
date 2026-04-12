from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # OpenRouter
    openrouter_api_key: str = ""
    openrouter_model: str = "anthropic/claude-sonnet"

    # LangFuse
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3001"

    # App limits
    max_file_size_mb: int = 20
    max_pages: int = 50
    max_agent_steps: int = 15
    circuit_breaker_usd: float = 1.0
    confidence_threshold: float = 0.7
    log_level: str = "INFO"

    # Paths
    faiss_index_path: str = "data/faiss/index.faiss"
    metadata_path: str = "data/faiss/metadata.json"
    tmp_dir: str = "tmp"
    log_dir: str = "logs"
