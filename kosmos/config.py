"""
Configuration management using Pydantic for validation.

Loads configuration from environment variables and provides validated settings
for all Kosmos components.
"""

from typing import List, Optional, Literal, Union
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
import os


class ClaudeConfig(BaseSettings):
    """
    Claude/Anthropic configuration.

    Note: This class is maintained for backward compatibility.
    New code should use AnthropicConfig or the provider-agnostic interface.
    """

    api_key: str = Field(
        description="Anthropic API key or '999...' for CLI mode",
        alias="ANTHROPIC_API_KEY"
    )
    model: str = Field(
        default="claude-3-5-sonnet-20241022",
        description="Claude model to use",
        alias="CLAUDE_MODEL"
    )
    max_tokens: int = Field(
        default=4096,
        ge=1,
        le=200000,
        description="Maximum tokens per request",
        alias="CLAUDE_MAX_TOKENS"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Sampling temperature",
        alias="CLAUDE_TEMPERATURE"
    )
    enable_cache: bool = Field(
        default=True,
        description="Enable prompt caching to reduce API costs",
        alias="CLAUDE_ENABLE_CACHE"
    )

    @property
    def is_cli_mode(self) -> bool:
        """Check if using CLI mode (API key is all 9s)."""
        return self.api_key.replace('9', '') == ''

    model_config = SettingsConfigDict(populate_by_name=True)


# Alias for clarity
AnthropicConfig = ClaudeConfig


class OpenAIConfig(BaseSettings):
    """
    OpenAI configuration.

    Supports OpenAI official API and OpenAI-compatible providers
    (Ollama, OpenRouter, Together AI, LM Studio, etc.)
    """

    api_key: str = Field(
        description="OpenAI API key (or dummy key for local models)",
        alias="OPENAI_API_KEY"
    )
    model: str = Field(
        default="gpt-4-turbo",
        description="Model name (e.g., gpt-4-turbo, llama3.1:70b)",
        alias="OPENAI_MODEL"
    )
    max_tokens: int = Field(
        default=4096,
        ge=1,
        le=128000,
        description="Maximum tokens per request",
        alias="OPENAI_MAX_TOKENS"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
        alias="OPENAI_TEMPERATURE"
    )
    base_url: Optional[str] = Field(
        default=None,
        description="Custom base URL for OpenAI-compatible APIs (e.g., http://localhost:11434/v1 for Ollama)",
        alias="OPENAI_BASE_URL"
    )
    organization: Optional[str] = Field(
        default=None,
        description="OpenAI organization ID (optional)",
        alias="OPENAI_ORGANIZATION"
    )

    model_config = SettingsConfigDict(populate_by_name=True)


class ResearchConfig(BaseSettings):
    """Research workflow configuration."""

    max_iterations: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum research iterations",
        alias="MAX_RESEARCH_ITERATIONS"
    )
    enabled_domains: Union[str, List[str]] = Field(
        default=["biology", "physics", "chemistry", "neuroscience"],
        description="Enabled scientific domains",
        alias="ENABLED_DOMAINS"
    )
    enabled_experiment_types: Union[str, List[str]] = Field(
        default=["computational", "data_analysis", "literature_synthesis"],
        description="Enabled experiment types",
        alias="ENABLED_EXPERIMENT_TYPES"
    )
    min_novelty_score: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum novelty score for hypotheses",
        alias="MIN_NOVELTY_SCORE"
    )
    enable_autonomous_iteration: bool = Field(
        default=True,
        description="Enable autonomous research iteration",
        alias="ENABLE_AUTONOMOUS_ITERATION"
    )
    budget_usd: float = Field(
        default=10.0,
        ge=0.0,
        description="Research budget in USD for API costs",
        alias="RESEARCH_BUDGET_USD"
    )

    @field_validator("enabled_domains", mode="before")
    @classmethod
    def parse_domains(cls, v):
        """Parse comma-separated domain string or handle empty values."""
        if v is None or v == "":
            # Return default if empty - will use field default
            return ["biology", "physics", "chemistry", "neuroscience"]
        if isinstance(v, str):
            # Handle comma-separated values
            domains = [d.strip() for d in v.split(",") if d.strip()]
            if not domains:
                return ["biology", "physics", "chemistry", "neuroscience"]
            return domains
        return v

    @field_validator("enabled_experiment_types", mode="before")
    @classmethod
    def parse_experiment_types(cls, v):
        """Parse comma-separated experiment types string or handle empty values."""
        if v is None or v == "":
            # Return default if empty
            return ["computational", "data_analysis", "literature_synthesis"]
        if isinstance(v, str):
            # Handle comma-separated values
            types = [e.strip() for e in v.split(",") if e.strip()]
            if not types:
                return ["computational", "data_analysis", "literature_synthesis"]
            return types
        return v

    @model_validator(mode="after")
    def ensure_lists(self):
        """Ensure enabled_domains and enabled_experiment_types are always lists."""
        # Convert any remaining strings to lists (shouldn't happen with validators, but safety)
        if isinstance(self.enabled_domains, str):
            self.enabled_domains = [d.strip() for d in self.enabled_domains.split(",") if d.strip()]
        if isinstance(self.enabled_experiment_types, str):
            self.enabled_experiment_types = [e.strip() for e in self.enabled_experiment_types.split(",") if e.strip()]
        return self

    model_config = SettingsConfigDict(populate_by_name=True)


class DatabaseConfig(BaseSettings):
    """Database configuration."""

    url: str = Field(
        default="sqlite:///kosmos.db",
        description="Database URL",
        alias="DATABASE_URL"
    )
    echo: bool = Field(
        default=False,
        description="Enable SQL echo logging",
        alias="DATABASE_ECHO"
    )

    @property
    def is_sqlite(self) -> bool:
        """Check if using SQLite."""
        return self.url.startswith("sqlite")

    model_config = SettingsConfigDict(populate_by_name=True)


class RedisConfig(BaseSettings):
    """Redis cache configuration."""

    url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL",
        alias="REDIS_URL"
    )
    enabled: bool = Field(
        default=False,
        description="Enable Redis caching",
        alias="REDIS_ENABLED"
    )
    max_connections: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Maximum connection pool size",
        alias="REDIS_MAX_CONNECTIONS"
    )
    socket_timeout: int = Field(
        default=5,
        ge=1,
        le=60,
        description="Socket timeout in seconds",
        alias="REDIS_SOCKET_TIMEOUT"
    )
    socket_connect_timeout: int = Field(
        default=5,
        ge=1,
        le=60,
        description="Socket connect timeout in seconds",
        alias="REDIS_SOCKET_CONNECT_TIMEOUT"
    )
    retry_on_timeout: bool = Field(
        default=True,
        description="Retry on timeout",
        alias="REDIS_RETRY_ON_TIMEOUT"
    )
    decode_responses: bool = Field(
        default=True,
        description="Decode responses as UTF-8 strings",
        alias="REDIS_DECODE_RESPONSES"
    )
    default_ttl_seconds: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Default cache TTL in seconds (1 minute to 24 hours)",
        alias="REDIS_DEFAULT_TTL_SECONDS"
    )

    @property
    def is_available(self) -> bool:
        """Check if Redis is enabled and configured."""
        return self.enabled and self.url is not None

    model_config = SettingsConfigDict(populate_by_name=True)


class LoggingConfig(BaseSettings):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Log level",
        alias="LOG_LEVEL"
    )
    format: Literal["json", "text"] = Field(
        default="json",
        description="Log format",
        alias="LOG_FORMAT"
    )
    file: Optional[str] = Field(
        default="logs/kosmos.log",
        description="Log file path (None for stdout only)",
        alias="LOG_FILE"
    )
    debug_mode: bool = Field(
        default=False,
        description="Enable debug mode with verbose output",
        alias="DEBUG_MODE"
    )

    model_config = SettingsConfigDict(populate_by_name=True)


class LiteratureConfig(BaseSettings):
    """Literature API configuration."""

    semantic_scholar_api_key: Optional[str] = Field(
        default=None,
        description="Semantic Scholar API key (optional, increases rate limits)",
        alias="SEMANTIC_SCHOLAR_API_KEY"
    )
    pubmed_api_key: Optional[str] = Field(
        default=None,
        description="PubMed API key (optional, increases rate limits)",
        alias="PUBMED_API_KEY"
    )
    pubmed_email: Optional[str] = Field(
        default=None,
        description="Email for PubMed E-utilities (recommended)",
        alias="PUBMED_EMAIL"
    )
    cache_ttl_hours: int = Field(
        default=48,
        ge=1,
        le=168,
        description="Literature API cache TTL in hours (24-168)",
        alias="LITERATURE_CACHE_TTL_HOURS"
    )
    max_results_per_query: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum results per literature search query",
        alias="MAX_RESULTS_PER_QUERY"
    )
    pdf_download_timeout: int = Field(
        default=30,
        ge=5,
        le=120,
        description="PDF download timeout in seconds",
        alias="PDF_DOWNLOAD_TIMEOUT"
    )

    model_config = SettingsConfigDict(populate_by_name=True)


class VectorDBConfig(BaseSettings):
    """Vector database configuration."""

    type: Literal["chromadb", "pinecone", "weaviate"] = Field(
        default="chromadb",
        description="Vector database type",
        alias="VECTOR_DB_TYPE"
    )
    chroma_persist_directory: str = Field(
        default=".chroma_db",
        description="ChromaDB persistence directory",
        alias="CHROMA_PERSIST_DIRECTORY"
    )
    pinecone_api_key: Optional[str] = Field(
        default=None,
        description="Pinecone API key",
        alias="PINECONE_API_KEY"
    )
    pinecone_environment: Optional[str] = Field(
        default=None,
        description="Pinecone environment",
        alias="PINECONE_ENVIRONMENT"
    )
    pinecone_index_name: Optional[str] = Field(
        default="kosmos",
        description="Pinecone index name",
        alias="PINECONE_INDEX_NAME"
    )

    @model_validator(mode="after")
    def validate_pinecone_config(self):
        """Validate Pinecone configuration if selected."""
        if self.type == "pinecone":
            if not self.pinecone_api_key:
                raise ValueError("PINECONE_API_KEY required when using Pinecone")
            if not self.pinecone_environment:
                raise ValueError("PINECONE_ENVIRONMENT required when using Pinecone")
        return self

    model_config = SettingsConfigDict(populate_by_name=True)


class Neo4jConfig(BaseSettings):
    """Neo4j knowledge graph configuration."""

    uri: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j connection URI",
        alias="NEO4J_URI"
    )
    user: str = Field(
        default="neo4j",
        description="Neo4j username",
        alias="NEO4J_USER"
    )
    password: str = Field(
        default="kosmos-password",
        description="Neo4j password",
        alias="NEO4J_PASSWORD"
    )
    database: str = Field(
        default="neo4j",
        description="Neo4j database name",
        alias="NEO4J_DATABASE"
    )
    max_connection_lifetime: int = Field(
        default=3600,
        ge=60,
        description="Max connection lifetime in seconds",
        alias="NEO4J_MAX_CONNECTION_LIFETIME"
    )
    max_connection_pool_size: int = Field(
        default=50,
        ge=1,
        description="Max connection pool size",
        alias="NEO4J_MAX_CONNECTION_POOL_SIZE"
    )

    model_config = SettingsConfigDict(populate_by_name=True)


class SafetyConfig(BaseSettings):
    """Safety and security configuration."""

    enable_safety_checks: bool = Field(
        default=True,
        description="Enable code safety checks",
        alias="ENABLE_SAFETY_CHECKS"
    )
    max_experiment_execution_time: int = Field(
        default=300,
        ge=1,
        description="Max execution time for experiments (seconds)",
        alias="MAX_EXPERIMENT_EXECUTION_TIME"
    )
    max_memory_mb: int = Field(
        default=2048,
        ge=128,
        description="Maximum memory usage (MB)",
        alias="MAX_MEMORY_MB"
    )
    max_cpu_cores: Optional[float] = Field(
        default=None,
        ge=0.1,
        description="Maximum CPU cores to use (None = unlimited)",
        alias="MAX_CPU_CORES"
    )
    enable_sandboxing: bool = Field(
        default=True,
        description="Enable sandboxed code execution",
        alias="ENABLE_SANDBOXING"
    )
    require_human_approval: bool = Field(
        default=False,
        description="Require human approval for high-risk operations",
        alias="REQUIRE_HUMAN_APPROVAL"
    )

    # Ethical guidelines
    ethical_guidelines_path: Optional[str] = Field(
        default=None,
        description="Path to ethical guidelines JSON file",
        alias="ETHICAL_GUIDELINES_PATH"
    )

    # Result verification
    enable_result_verification: bool = Field(
        default=True,
        description="Enable result verification",
        alias="ENABLE_RESULT_VERIFICATION"
    )
    outlier_threshold: float = Field(
        default=3.0,
        ge=1.0,
        description="Z-score threshold for outlier detection",
        alias="OUTLIER_THRESHOLD"
    )

    # Reproducibility
    default_random_seed: int = Field(
        default=42,
        description="Default random seed for reproducibility",
        alias="DEFAULT_RANDOM_SEED"
    )
    capture_environment: bool = Field(
        default=True,
        description="Capture environment snapshots",
        alias="CAPTURE_ENVIRONMENT"
    )

    # Human oversight
    approval_mode: str = Field(
        default="blocking",
        description="Approval workflow mode (blocking/queue/automatic/disabled)",
        alias="APPROVAL_MODE"
    )
    auto_approve_low_risk: bool = Field(
        default=True,
        description="Automatically approve low-risk operations",
        alias="AUTO_APPROVE_LOW_RISK"
    )

    # Notifications
    notification_channel: str = Field(
        default="both",
        description="Notification channel (console/log/both)",
        alias="NOTIFICATION_CHANNEL"
    )
    notification_min_level: str = Field(
        default="info",
        description="Minimum notification level (debug/info/warning/error/critical)",
        alias="NOTIFICATION_MIN_LEVEL"
    )
    use_rich_formatting: bool = Field(
        default=True,
        description="Use rich formatting for console notifications",
        alias="USE_RICH_FORMATTING"
    )

    # Incident logging
    incident_log_path: str = Field(
        default="safety_incidents.jsonl",
        description="Path to safety incident log file",
        alias="INCIDENT_LOG_PATH"
    )
    audit_log_path: str = Field(
        default="human_review_audit.jsonl",
        description="Path to human review audit log",
        alias="AUDIT_LOG_PATH"
    )

    model_config = SettingsConfigDict(populate_by_name=True)


class PerformanceConfig(BaseSettings):
    """Performance and caching configuration."""

    enable_result_caching: bool = Field(
        default=True,
        description="Enable result caching",
        alias="ENABLE_RESULT_CACHING"
    )
    cache_ttl: int = Field(
        default=3600,
        ge=0,
        description="Cache TTL in seconds",
        alias="CACHE_TTL"
    )
    parallel_experiments: int = Field(
        default=0,
        ge=0,
        description="Number of parallel experiments (0 = sequential)",
        alias="PARALLEL_EXPERIMENTS"
    )

    # Concurrent operations configuration
    enable_concurrent_operations: bool = Field(
        default=False,
        description="Enable concurrent research operations",
        alias="ENABLE_CONCURRENT_OPERATIONS"
    )
    max_parallel_hypotheses: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum concurrent hypothesis evaluations",
        alias="MAX_PARALLEL_HYPOTHESES"
    )
    max_concurrent_experiments: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Maximum concurrent experiment executions",
        alias="MAX_CONCURRENT_EXPERIMENTS"
    )
    max_concurrent_llm_calls: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum concurrent LLM API calls",
        alias="MAX_CONCURRENT_LLM_CALLS"
    )
    llm_rate_limit_per_minute: int = Field(
        default=50,
        ge=1,
        le=200,
        description="LLM API rate limit per minute",
        alias="LLM_RATE_LIMIT_PER_MINUTE"
    )
    async_batch_timeout: int = Field(
        default=300,
        ge=10,
        le=3600,
        description="Timeout for async batch operations (seconds)",
        alias="ASYNC_BATCH_TIMEOUT"
    )

    model_config = SettingsConfigDict(populate_by_name=True)


class MonitoringConfig(BaseSettings):
    """Monitoring and metrics configuration."""

    enable_usage_stats: bool = Field(
        default=True,
        description="Enable usage statistics tracking",
        alias="ENABLE_USAGE_STATS"
    )
    metrics_export_interval: int = Field(
        default=60,
        ge=0,
        description="Metrics export interval in seconds (0 = disabled)",
        alias="METRICS_EXPORT_INTERVAL"
    )

    model_config = SettingsConfigDict(populate_by_name=True)


class DevelopmentConfig(BaseSettings):
    """Development settings."""

    hot_reload: bool = Field(
        default=False,
        description="Enable hot reload (development only)",
        alias="HOT_RELOAD"
    )
    log_api_requests: bool = Field(
        default=False,
        description="Log all API requests",
        alias="LOG_API_REQUESTS"
    )
    test_mode: bool = Field(
        default=False,
        description="Test mode (uses mocks)",
        alias="TEST_MODE"
    )

    model_config = SettingsConfigDict(populate_by_name=True)


class WorldModelConfig(BaseSettings):
    """World model configuration for persistent knowledge graphs."""

    enabled: bool = Field(
        default=True,
        description="Enable persistent knowledge graphs",
        alias="WORLD_MODEL_ENABLED"
    )

    mode: Literal["simple", "production"] = Field(
        default="simple",
        description="Storage mode: simple (Neo4j) or production (polyglot)",
        alias="WORLD_MODEL_MODE"
    )

    project: Optional[str] = Field(
        default=None,
        description="Default project namespace for multi-project support",
        alias="WORLD_MODEL_PROJECT"
    )

    auto_save_interval: int = Field(
        default=300,
        ge=0,
        description="Auto-export interval in seconds (0 = disabled)",
        alias="WORLD_MODEL_AUTO_SAVE_INTERVAL"
    )

    model_config = SettingsConfigDict(populate_by_name=True)


def _optional_openai_config() -> Optional[OpenAIConfig]:
    """Create OpenAIConfig only if OPENAI_API_KEY is set."""
    import os
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIConfig()
    return None


def _optional_anthropic_config() -> Optional[AnthropicConfig]:
    """Create AnthropicConfig only if configured."""
    import os
    # Anthropic config is created by default via claude field
    # This is for the anthropic alias field
    if os.getenv("ANTHROPIC_API_KEY"):
        return AnthropicConfig()
    return None


class KosmosConfig(BaseSettings):
    """
    Master configuration for Kosmos AI Scientist.

    Loads all configuration from environment variables with validation.

    Example:
        ```python
        from kosmos.config import get_config

        config = get_config()

        # Access configuration
        print(config.claude.model)  # Backward compatible
        print(config.research.max_iterations)
        print(config.database.url)

        # Multi-provider support
        print(config.llm_provider)  # "anthropic" or "openai"
        if config.llm_provider == "openai":
            print(config.openai.model)

        # Check Claude mode (backward compatible)
        if config.claude.is_cli_mode:
            print("Using Claude Code CLI")
        else:
            print("Using Anthropic API")
        ```
    """

    # LLM Provider Selection
    llm_provider: Literal["anthropic", "openai"] = Field(
        default="anthropic",
        description="LLM provider to use (anthropic or openai)",
        alias="LLM_PROVIDER"
    )

    # Component configurations
    claude: ClaudeConfig = Field(default_factory=ClaudeConfig)  # Backward compatibility
    anthropic: Optional[AnthropicConfig] = Field(default_factory=_optional_anthropic_config)  # New name (optional, defaults to claude)
    openai: Optional[OpenAIConfig] = Field(default_factory=_optional_openai_config)  # OpenAI provider config
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    literature: LiteratureConfig = Field(default_factory=LiteratureConfig)
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    development: DevelopmentConfig = Field(default_factory=DevelopmentConfig)
    world_model: WorldModelConfig = Field(default_factory=WorldModelConfig)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    @model_validator(mode="after")
    def validate_provider_config(self):
        """Validate that required API keys are set for the selected provider."""
        if self.llm_provider == "openai":
            if not self.openai or not self.openai.api_key:
                raise ValueError(
                    "OPENAI_API_KEY is required when LLM_PROVIDER=openai. "
                    "Please set OPENAI_API_KEY in your environment or .env file."
                )
        elif self.llm_provider == "anthropic":
            if not self.claude.api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic. "
                    "Please set ANTHROPIC_API_KEY in your environment or .env file."
                )
        return self

    def create_directories(self):
        """Create necessary directories if they don't exist."""
        # Create log directory
        if self.logging.file:
            log_dir = Path(self.logging.file).parent
            log_dir.mkdir(parents=True, exist_ok=True)

        # Create ChromaDB directory if using ChromaDB
        if self.vector_db.type == "chromadb":
            Path(self.vector_db.chroma_persist_directory).mkdir(parents=True, exist_ok=True)

    def validate_dependencies(self) -> List[str]:
        """
        Check if all required dependencies are available.

        Returns:
            List[str]: List of missing dependencies (empty if all present)
        """
        missing = []

        # Check LLM provider configuration
        if self.llm_provider == "anthropic":
            if not self.claude.api_key:
                missing.append("ANTHROPIC_API_KEY not set (required for anthropic provider)")
        elif self.llm_provider == "openai":
            if self.openai is None:
                missing.append("OpenAI configuration missing (OPENAI_API_KEY not set)")
            elif not self.openai.api_key:
                missing.append("OPENAI_API_KEY not set (required for openai provider)")

        # Check Pinecone if selected
        if self.vector_db.type == "pinecone" and not self.vector_db.pinecone_api_key:
            missing.append("PINECONE_API_KEY not set")

        return missing

    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary.

        Returns:
            dict: Configuration as dictionary
        """
        config_dict = {
            "llm_provider": self.llm_provider,
            "claude": self.claude.model_dump(),
            "research": self.research.model_dump(),
            "database": self.database.model_dump(),
            "redis": self.redis.model_dump(),
            "logging": self.logging.model_dump(),
            "literature": self.literature.model_dump(),
            "vector_db": self.vector_db.model_dump(),
            "neo4j": self.neo4j.model_dump(),
            "safety": self.safety.model_dump(),
            "performance": self.performance.model_dump(),
            "monitoring": self.monitoring.model_dump(),
            "development": self.development.model_dump(),
            "world_model": self.world_model.model_dump(),
        }

        # Add provider-specific configs if present
        if self.anthropic:
            config_dict["anthropic"] = self.anthropic.model_dump()
        if self.openai:
            config_dict["openai"] = self.openai.model_dump()

        return config_dict


# Singleton configuration instance
_config: Optional[KosmosConfig] = None


def get_config(reload: bool = False) -> KosmosConfig:
    """
    Get or create configuration singleton.

    Args:
        reload: If True, reload configuration from environment

    Returns:
        KosmosConfig: Configuration instance
    """
    global _config
    if _config is None or reload:
        _config = KosmosConfig()
        _config.create_directories()
    return _config


def reset_config():
    """Reset configuration singleton (useful for testing)."""
    global _config
    _config = None
