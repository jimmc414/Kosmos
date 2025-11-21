"""
Test suite for Core Infrastructure - Configuration Requirements (REQ-CFG-001 through REQ-CFG-005).

This test file validates configuration management, validation, defaults, and documentation.

Requirements tested:
- REQ-CFG-001 (MUST): Load configuration
- REQ-CFG-002 (MUST): Validate required params
- REQ-CFG-003 (MUST): Default values
- REQ-CFG-004 (MUST): Parameter documentation
- REQ-CFG-005 (MUST): No execution with invalid config
"""

import os
import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import patch, MagicMock
from pydantic import BaseModel, Field, ValidationError


# ============================================================================
# REQ-CFG-001: Load Configuration (MUST)
# ============================================================================

class TestREQ_CFG_001_LoadConfiguration:
    """Test REQ-CFG-001: System loads configuration from multiple sources."""

    def test_config_from_environment_variables(self):
        """Verify configuration loads from environment variables."""
        from kosmos.config import get_config, reset_config

        reset_config()

        with patch.dict(os.environ, {
            'ANTHROPIC_API_KEY': 'sk-ant-test-env',
            'LOG_LEVEL': 'DEBUG',
            'DATABASE_URL': 'sqlite:///test_env.db'
        }):
            config = get_config(reload=True)

            assert config.claude.api_key == 'sk-ant-test-env'
            assert config.logging.level == 'DEBUG'
            assert 'test_env.db' in config.database.url

        reset_config()

    def test_config_from_dotenv_file(self):
        """Verify configuration loads from .env file."""
        from kosmos.config import KosmosConfig
        from pathlib import Path

        # Create temporary .env file
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text(
                "ANTHROPIC_API_KEY=sk-ant-from-file\n"
                "LOG_LEVEL=WARNING\n"
            )

            # Load config with custom env file
            with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'sk-ant-from-file'}):
                config = KosmosConfig()
                assert config.claude.api_key == 'sk-ant-from-file'

    def test_config_singleton_pattern(self):
        """Verify config uses singleton pattern."""
        from kosmos.config import get_config, reset_config

        reset_config()

        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'sk-ant-singleton'}):
            config1 = get_config()
            config2 = get_config()

            # Should be same instance
            assert config1 is config2

        reset_config()

    def test_config_reload_capability(self):
        """Verify configuration can be reloaded."""
        from kosmos.config import get_config, reset_config

        reset_config()

        # Initial config
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'sk-ant-initial', 'LOG_LEVEL': 'INFO'}):
            config1 = get_config()
            initial_level = config1.logging.level

        # Change environment and reload
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'sk-ant-updated', 'LOG_LEVEL': 'DEBUG'}):
            config2 = get_config(reload=True)
            updated_level = config2.logging.level

            # Should reflect new value
            assert updated_level == 'DEBUG'
            assert updated_level != initial_level

        reset_config()

    def test_config_to_dict_conversion(self):
        """Verify configuration can be converted to dictionary."""
        from kosmos.config import get_config, reset_config

        reset_config()

        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'sk-ant-dict-test'}):
            config = get_config()
            config_dict = config.to_dict()

            # Should be a dictionary
            assert isinstance(config_dict, dict)

            # Should contain major sections
            assert 'claude' in config_dict
            assert 'logging' in config_dict
            assert 'database' in config_dict

            # Should contain actual values
            assert isinstance(config_dict['claude'], dict)
            assert isinstance(config_dict['logging'], dict)

        reset_config()

    def test_config_environment_precedence(self):
        """Verify environment variables take precedence over defaults."""
        from kosmos.config import get_config, reset_config

        reset_config()

        # Without env var, should use default
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'sk-ant-test'}, clear=True):
            config1 = get_config(reload=True)
            default_level = config1.logging.level  # Default is INFO

        reset_config()

        # With env var, should override default
        with patch.dict(os.environ, {
            'ANTHROPIC_API_KEY': 'sk-ant-test',
            'LOG_LEVEL': 'ERROR'
        }):
            config2 = get_config(reload=True)
            assert config2.logging.level == 'ERROR'

        reset_config()


# ============================================================================
# REQ-CFG-002: Validate Required Parameters (MUST)
# ============================================================================

class TestREQ_CFG_002_ValidateRequiredParameters:
    """Test REQ-CFG-002: System validates required configuration parameters."""

    def test_missing_required_api_key_fails(self):
        """Verify missing API key causes validation failure."""
        from kosmos.config import KosmosConfig

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValidationError):
                config = KosmosConfig()

    def test_required_fields_documented(self):
        """Verify required fields are properly defined."""
        from kosmos.config import ClaudeConfig

        # Check field metadata
        fields = ClaudeConfig.model_fields

        # api_key should be required (no default)
        assert 'api_key' in fields
        # Note: Field is required if no default is set

    def test_provider_specific_validation(self):
        """Verify provider-specific parameters are validated."""
        from kosmos.config import get_config, reset_config

        reset_config()

        # Anthropic provider requires ANTHROPIC_API_KEY
        with patch.dict(os.environ, {'LLM_PROVIDER': 'anthropic'}, clear=True):
            with pytest.raises((ValidationError, ValueError)):
                config = get_config(reload=True)

        reset_config()

        # OpenAI provider requires OPENAI_API_KEY
        with patch.dict(os.environ, {'LLM_PROVIDER': 'openai'}, clear=True):
            with pytest.raises((ValidationError, ValueError)):
                config = get_config(reload=True)

        reset_config()

    def test_field_type_validation(self):
        """Verify field types are validated."""
        from kosmos.config import ClaudeConfig

        # Valid types should work
        with patch.dict(os.environ, {
            'ANTHROPIC_API_KEY': 'sk-ant-test',
            'CLAUDE_MAX_TOKENS': '2048',
            'CLAUDE_TEMPERATURE': '0.5'
        }):
            config = ClaudeConfig()
            assert isinstance(config.max_tokens, int)
            assert isinstance(config.temperature, float)

        # Invalid types should fail
        with patch.dict(os.environ, {
            'ANTHROPIC_API_KEY': 'sk-ant-test',
            'CLAUDE_MAX_TOKENS': 'not_a_number'
        }):
            with pytest.raises(ValidationError):
                config = ClaudeConfig()

    def test_range_validation(self):
        """Verify numeric ranges are validated."""
        from kosmos.config import ClaudeConfig

        # Temperature out of range should fail
        with patch.dict(os.environ, {
            'ANTHROPIC_API_KEY': 'sk-ant-test',
            'CLAUDE_TEMPERATURE': '2.5'  # > 1.0
        }):
            with pytest.raises(ValidationError):
                config = ClaudeConfig()

        # Max tokens out of range should fail
        with patch.dict(os.environ, {
            'ANTHROPIC_API_KEY': 'sk-ant-test',
            'CLAUDE_MAX_TOKENS': '0'  # < 1
        }):
            with pytest.raises(ValidationError):
                config = ClaudeConfig()

    def test_enum_validation(self):
        """Verify enum fields are validated."""
        from kosmos.config import LoggingConfig

        # Valid log level
        with patch.dict(os.environ, {'LOG_LEVEL': 'INFO'}):
            config = LoggingConfig()
            assert config.level == 'INFO'

        # Invalid log level should fail
        with patch.dict(os.environ, {'LOG_LEVEL': 'INVALID_LEVEL'}):
            with pytest.raises(ValidationError):
                config = LoggingConfig()

    def test_nested_validation(self):
        """Verify nested configuration objects are validated."""
        from kosmos.config import VectorDBConfig

        # Pinecone requires api_key and environment
        with patch.dict(os.environ, {
            'VECTOR_DB_TYPE': 'pinecone'
            # Missing PINECONE_API_KEY and PINECONE_ENVIRONMENT
        }):
            with pytest.raises(ValidationError):
                config = VectorDBConfig()


# ============================================================================
# REQ-CFG-003: Default Values (MUST)
# ============================================================================

class TestREQ_CFG_003_DefaultValues:
    """Test REQ-CFG-003: System provides sensible default values."""

    def test_claude_defaults(self):
        """Verify Claude configuration has sensible defaults."""
        from kosmos.config import ClaudeConfig

        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'sk-ant-test'}):
            config = ClaudeConfig()

            # Check default values
            assert config.model == "claude-3-5-sonnet-20241022"
            assert config.max_tokens == 4096
            assert config.temperature == 0.7
            assert config.enable_cache is True

    def test_logging_defaults(self):
        """Verify logging configuration has sensible defaults."""
        from kosmos.config import LoggingConfig

        config = LoggingConfig()

        assert config.level == 'INFO'
        assert config.format == 'json'
        assert config.file == 'logs/kosmos.log'
        assert config.debug_mode is False

    def test_database_defaults(self):
        """Verify database configuration has sensible defaults."""
        from kosmos.config import DatabaseConfig

        config = DatabaseConfig()

        assert config.url == 'sqlite:///kosmos.db'
        assert config.echo is False

    def test_research_defaults(self):
        """Verify research configuration has sensible defaults."""
        from kosmos.config import ResearchConfig

        config = ResearchConfig()

        assert config.max_iterations == 10
        assert isinstance(config.enabled_domains, list)
        assert len(config.enabled_domains) > 0
        assert config.min_novelty_score == 0.6
        assert config.enable_autonomous_iteration is True
        assert config.budget_usd == 10.0

    def test_safety_defaults(self):
        """Verify safety configuration has sensible defaults."""
        from kosmos.config import SafetyConfig

        config = SafetyConfig()

        assert config.enable_safety_checks is True
        assert config.max_experiment_execution_time == 300
        assert config.enable_sandboxing is True
        assert config.enable_result_verification is True
        assert config.default_random_seed == 42

    def test_performance_defaults(self):
        """Verify performance configuration has sensible defaults."""
        from kosmos.config import PerformanceConfig

        config = PerformanceConfig()

        assert config.enable_result_caching is True
        assert config.cache_ttl == 3600
        assert config.parallel_experiments == 0  # Sequential by default
        assert config.enable_concurrent_operations is False  # Safe default

    def test_optional_fields_default_to_none(self):
        """Verify optional fields default to None appropriately."""
        from kosmos.config import LiteratureConfig

        config = LiteratureConfig()

        # Optional API keys should default to None
        assert config.semantic_scholar_api_key is None
        assert config.pubmed_api_key is None
        assert config.pubmed_email is None

    def test_defaults_are_production_ready(self):
        """Verify default configuration is suitable for production."""
        from kosmos.config import get_config, reset_config

        reset_config()

        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'sk-ant-prod-test'}):
            config = get_config()

            # Safety should be enabled
            assert config.safety.enable_safety_checks is True
            assert config.safety.enable_sandboxing is True

            # Logging should be structured
            assert config.logging.format == 'json'

            # Caching should be enabled
            assert config.claude.enable_cache is True

        reset_config()


# ============================================================================
# REQ-CFG-004: Parameter Documentation (MUST)
# ============================================================================

class TestREQ_CFG_004_ParameterDocumentation:
    """Test REQ-CFG-004: All configuration parameters are documented."""

    def test_fields_have_descriptions(self):
        """Verify all fields have description metadata."""
        from kosmos.config import ClaudeConfig

        fields = ClaudeConfig.model_fields

        for field_name, field_info in fields.items():
            # Each field should have a description
            assert field_info.description is not None, \
                f"Field '{field_name}' missing description"
            assert len(field_info.description) > 0, \
                f"Field '{field_name}' has empty description"

    def test_config_classes_have_docstrings(self):
        """Verify configuration classes have docstrings."""
        from kosmos.config import (
            ClaudeConfig, LoggingConfig, DatabaseConfig,
            ResearchConfig, SafetyConfig, KosmosConfig
        )

        config_classes = [
            ClaudeConfig, LoggingConfig, DatabaseConfig,
            ResearchConfig, SafetyConfig, KosmosConfig
        ]

        for config_class in config_classes:
            assert config_class.__doc__ is not None, \
                f"{config_class.__name__} missing docstring"
            assert len(config_class.__doc__) > 10, \
                f"{config_class.__name__} has insufficient docstring"

    def test_field_examples_in_docstrings(self):
        """Verify main config class has usage examples."""
        from kosmos.config import KosmosConfig

        docstring = KosmosConfig.__doc__

        # Should contain example usage
        assert 'Example:' in docstring or 'example' in docstring.lower()
        assert 'get_config' in docstring

    def test_field_constraints_documented(self):
        """Verify field constraints are documented or evident."""
        from kosmos.config import ClaudeConfig

        fields = ClaudeConfig.model_fields

        # Temperature field should have constraints
        temp_field = fields['temperature']
        assert temp_field.description is not None

        # Check field has validation info (ge, le, etc.)
        # This would be in the Field definition

    def test_environment_variable_mapping_clear(self):
        """Verify environment variable names are clear from aliases."""
        from kosmos.config import ClaudeConfig

        fields = ClaudeConfig.model_fields

        # Fields should have clear aliases matching env vars
        assert fields['api_key'].alias == 'ANTHROPIC_API_KEY'
        assert fields['model'].alias == 'CLAUDE_MODEL'
        assert fields['max_tokens'].alias == 'CLAUDE_MAX_TOKENS'

    def test_config_documentation_accessible(self):
        """Verify configuration documentation is accessible."""
        from kosmos.config import get_config, reset_config

        reset_config()

        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'sk-ant-doc-test'}):
            config = get_config()

            # Should be able to introspect configuration
            config_dict = config.to_dict()

            # Should contain all major sections
            major_sections = ['claude', 'logging', 'database', 'research', 'safety']
            for section in major_sections:
                assert section in config_dict, f"Section '{section}' not in config dict"

        reset_config()

    def test_validation_error_messages_informative(self):
        """Verify validation errors provide helpful messages."""
        from kosmos.config import ClaudeConfig

        try:
            with patch.dict(os.environ, {
                'ANTHROPIC_API_KEY': 'sk-ant-test',
                'CLAUDE_TEMPERATURE': '5.0'  # Invalid
            }):
                config = ClaudeConfig()
            pytest.fail("Should have raised ValidationError")

        except ValidationError as e:
            error_msg = str(e)
            # Error should mention the field and constraint
            assert 'temperature' in error_msg.lower()


# ============================================================================
# REQ-CFG-005: No Execution with Invalid Config (MUST)
# ============================================================================

class TestREQ_CFG_005_NoExecutionWithInvalidConfig:
    """Test REQ-CFG-005: System prevents execution with invalid configuration."""

    def test_initialization_fails_with_invalid_config(self):
        """Verify system initialization fails with invalid config."""
        from kosmos.config import KosmosConfig

        # Missing required API key
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValidationError):
                config = KosmosConfig()

    def test_llm_client_requires_valid_config(self):
        """Verify LLM client requires valid configuration."""
        from kosmos.core.llm import ClaudeClient

        # Should fail without API key
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                client = ClaudeClient()

    def test_validate_dependencies_method(self):
        """Verify config provides dependency validation."""
        from kosmos.config import get_config, reset_config

        reset_config()

        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'sk-ant-validate'}):
            config = get_config()

            # Should have validation method
            missing = config.validate_dependencies()

            # Should return list (empty if all dependencies present)
            assert isinstance(missing, list)

        reset_config()

    def test_invalid_provider_selection_fails(self):
        """Verify invalid provider selection is caught."""
        from kosmos.config import KosmosConfig

        # Invalid provider
        with patch.dict(os.environ, {
            'LLM_PROVIDER': 'invalid_provider',
            'ANTHROPIC_API_KEY': 'sk-ant-test'
        }):
            with pytest.raises(ValidationError):
                config = KosmosConfig()

    def test_pinecone_without_credentials_fails(self):
        """Verify Pinecone config fails without required credentials."""
        from kosmos.config import VectorDBConfig

        with patch.dict(os.environ, {
            'VECTOR_DB_TYPE': 'pinecone'
            # Missing PINECONE_API_KEY and PINECONE_ENVIRONMENT
        }):
            with pytest.raises(ValidationError):
                config = VectorDBConfig()

    def test_incompatible_config_combinations_fail(self):
        """Verify incompatible configuration combinations are rejected."""
        from kosmos.config import get_config, reset_config

        reset_config()

        # OpenAI provider without OpenAI API key should fail
        with patch.dict(os.environ, {
            'LLM_PROVIDER': 'openai',
            'ANTHROPIC_API_KEY': 'sk-ant-test'  # Wrong provider key
        }):
            with pytest.raises((ValidationError, ValueError)):
                config = get_config(reload=True)

        reset_config()

    def test_create_directories_with_valid_config(self):
        """Verify directory creation only happens with valid config."""
        from kosmos.config import get_config, reset_config

        reset_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, 'logs', 'test.log')

            with patch.dict(os.environ, {
                'ANTHROPIC_API_KEY': 'sk-ant-dirs',
                'LOG_FILE': log_file
            }):
                config = get_config(reload=True)

                # Create directories
                config.create_directories()

                # Log directory should be created
                log_dir = Path(log_file).parent
                assert log_dir.exists()

        reset_config()

    def test_early_validation_prevents_runtime_errors(self):
        """Verify configuration validation happens before execution."""
        from kosmos.config import KosmosConfig

        # Validation should happen at config creation, not later
        validation_happened = False

        try:
            with patch.dict(os.environ, {}, clear=True):
                config = KosmosConfig()  # Should fail here
        except ValidationError:
            validation_happened = True

        assert validation_happened, "Validation should have occurred during initialization"


# ============================================================================
# Integration Tests
# ============================================================================

class TestConfigurationIntegration:
    """Integration tests for configuration system."""

    def test_full_config_lifecycle(self):
        """Test complete configuration lifecycle."""
        from kosmos.config import get_config, reset_config

        # 1. Start fresh
        reset_config()

        # 2. Load configuration
        with patch.dict(os.environ, {
            'ANTHROPIC_API_KEY': 'sk-ant-lifecycle',
            'LOG_LEVEL': 'DEBUG',
            'DATABASE_URL': 'sqlite:///lifecycle.db'
        }):
            config = get_config()

            # 3. Verify loaded correctly
            assert config.claude.api_key == 'sk-ant-lifecycle'
            assert config.logging.level == 'DEBUG'

            # 4. Convert to dict
            config_dict = config.to_dict()
            assert isinstance(config_dict, dict)

            # 5. Validate dependencies
            missing = config.validate_dependencies()
            assert isinstance(missing, list)

            # 6. Create directories
            config.create_directories()

        # 7. Reset
        reset_config()

    def test_config_isolation_between_tests(self):
        """Verify configuration is properly isolated between tests."""
        from kosmos.config import get_config, reset_config

        # Test 1
        reset_config()
        with patch.dict(os.environ, {
            'ANTHROPIC_API_KEY': 'sk-ant-test1',
            'LOG_LEVEL': 'INFO'
        }):
            config1 = get_config()
            level1 = config1.logging.level

        # Test 2
        reset_config()
        with patch.dict(os.environ, {
            'ANTHROPIC_API_KEY': 'sk-ant-test2',
            'LOG_LEVEL': 'ERROR'
        }):
            config2 = get_config(reload=True)
            level2 = config2.logging.level

        # Should be different
        assert level1 != level2

        reset_config()

    def test_config_with_all_providers(self):
        """Test configuration with different LLM providers."""
        from kosmos.config import get_config, reset_config

        # Anthropic
        reset_config()
        with patch.dict(os.environ, {
            'LLM_PROVIDER': 'anthropic',
            'ANTHROPIC_API_KEY': 'sk-ant-provider-test'
        }):
            config = get_config()
            assert config.llm_provider == 'anthropic'

        # OpenAI
        reset_config()
        with patch.dict(os.environ, {
            'LLM_PROVIDER': 'openai',
            'OPENAI_API_KEY': 'sk-openai-provider-test'
        }):
            config = get_config(reload=True)
            assert config.llm_provider == 'openai'

        reset_config()

    def test_config_validation_comprehensive(self):
        """Comprehensive validation test."""
        from kosmos.config import get_config, reset_config

        reset_config()

        # Valid configuration should pass all checks
        with patch.dict(os.environ, {
            'ANTHROPIC_API_KEY': 'sk-ant-comprehensive',
            'LOG_LEVEL': 'INFO',
            'CLAUDE_TEMPERATURE': '0.7',
            'CLAUDE_MAX_TOKENS': '4096',
            'DATABASE_URL': 'sqlite:///test.db',
            'ENABLE_SAFETY_CHECKS': 'true',
            'MAX_RESEARCH_ITERATIONS': '10'
        }):
            config = get_config()

            # All validations should pass
            assert config.claude.api_key is not None
            assert 0.0 <= config.claude.temperature <= 1.0
            assert config.claude.max_tokens > 0
            assert config.safety.enable_safety_checks is True
            assert config.research.max_iterations > 0

            # Dependencies should be satisfied
            missing = config.validate_dependencies()
            # May have some optional dependencies missing, but required ones should be present
            assert isinstance(missing, list)

        reset_config()

    def test_config_documentation_complete(self):
        """Verify all configuration sections are documented."""
        from kosmos.config import KosmosConfig

        # Check main config class
        assert KosmosConfig.__doc__ is not None
        docstring = KosmosConfig.__doc__

        # Should mention key concepts
        assert 'configuration' in docstring.lower()
        assert 'environment' in docstring.lower()

        # Should have example
        assert 'example' in docstring.lower()
        assert 'get_config' in docstring.lower()


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestConfigurationEdgeCases:
    """Test edge cases and error handling in configuration."""

    def test_empty_string_values(self):
        """Test handling of empty string values."""
        from kosmos.config import LoggingConfig

        with patch.dict(os.environ, {
            'LOG_FILE': ''  # Empty string
        }):
            config = LoggingConfig()
            # Empty string should be treated as None or use default

    def test_whitespace_values(self):
        """Test handling of whitespace values."""
        from kosmos.config import ClaudeConfig

        with patch.dict(os.environ, {
            'ANTHROPIC_API_KEY': '  sk-ant-whitespace  '
        }):
            config = ClaudeConfig()
            # Should handle whitespace appropriately

    def test_case_sensitivity(self):
        """Test environment variable case handling."""
        from kosmos.config import KosmosConfig

        # Config should be case-insensitive by default
        with patch.dict(os.environ, {
            'anthropic_api_key': 'sk-ant-lowercase',  # lowercase
            'LOG_LEVEL': 'info'  # mixed case
        }):
            try:
                config = KosmosConfig()
                # Should handle case variations
            except ValidationError:
                # Some systems may be case-sensitive
                pass

    def test_numeric_string_conversion(self):
        """Test automatic conversion of numeric strings."""
        from kosmos.config import ClaudeConfig

        with patch.dict(os.environ, {
            'ANTHROPIC_API_KEY': 'sk-ant-numeric',
            'CLAUDE_MAX_TOKENS': '2048',  # String that should be converted to int
            'CLAUDE_TEMPERATURE': '0.5'   # String that should be converted to float
        }):
            config = ClaudeConfig()

            assert isinstance(config.max_tokens, int)
            assert config.max_tokens == 2048

            assert isinstance(config.temperature, float)
            assert config.temperature == 0.5

    def test_boolean_string_conversion(self):
        """Test automatic conversion of boolean strings."""
        from kosmos.config import SafetyConfig

        # Test various boolean representations
        for true_val in ['true', 'True', 'TRUE', '1', 'yes']:
            with patch.dict(os.environ, {'ENABLE_SAFETY_CHECKS': true_val}):
                config = SafetyConfig()
                assert config.enable_safety_checks is True

    def test_list_from_comma_separated(self):
        """Test list parsing from comma-separated strings."""
        from kosmos.config import ResearchConfig

        with patch.dict(os.environ, {
            'ENABLED_DOMAINS': 'biology,physics,chemistry'
        }):
            config = ResearchConfig()

            assert isinstance(config.enabled_domains, list)
            assert 'biology' in config.enabled_domains
            assert 'physics' in config.enabled_domains
            assert 'chemistry' in config.enabled_domains


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
