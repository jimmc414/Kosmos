# Issue #22 Analysis Report

**Issue:** `kosmos doctor requires ANTHROPIC_API_KEY even with OPENAI provider`
**Reporter:** User attempting to use Kosmos with Ollama as an OpenAI-compatible provider
**Version:** Kosmos v0.2.0
**Date:** 2025-11-26

---

## Executive Summary

The `kosmos doctor` command (and other configuration-dependent operations) fails with a validation error requiring `ANTHROPIC_API_KEY` even when the user has explicitly configured `LLM_PROVIDER=openai`. This is a **critical bug** that prevents users from using Kosmos with alternative LLM providers.

**Error Message:**
```
1 validation error for ClaudeConfig
ANTHROPIC_API_KEY
  Field required [type=missing, input_value={}, input_type=dict]
```

---

## Root Cause Analysis

### Primary Issue: Unconditional `ClaudeConfig` Instantiation

The root cause is in `/kosmos/config.py` at line 735:

```python
claude: ClaudeConfig = Field(default_factory=ClaudeConfig)  # Backward compatibility
```

This line **unconditionally** attempts to create a `ClaudeConfig` instance during `KosmosConfig` initialization, regardless of which `LLM_PROVIDER` is selected. Since `ClaudeConfig.api_key` is a **required field with no default value**, Pydantic validation fails immediately if `ANTHROPIC_API_KEY` is not set.

### Validation Timing Problem

The codebase has a proper provider-specific validator at lines 758-773:

```python
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
```

However, this validator **never executes** because field instantiation fails first. The validation flow is:

```
KosmosConfig initialization
    |
    v
Field default_factory calls (FAILS HERE)
    |-- claude: ClaudeConfig() --> ANTHROPIC_API_KEY missing --> ValidationError
    |-- anthropic: _optional_anthropic_config() --> (never reached)
    |-- openai: _optional_openai_config() --> (never reached)
    |
    v
model_validator: validate_provider_config() --> (never reached)
```

### Secondary Issue: Hardcoded API Key Check in Doctor Command

The `doctor` command in `/kosmos/cli/main.py` at line 257 has a hardcoded check:

```python
api_key_ok = bool(os.getenv("ANTHROPIC_API_KEY"))
checks.append(("Anthropic API Key", "Configured" if api_key_ok else "Not set", api_key_ok))
```

This check:
1. Does not respect the `LLM_PROVIDER` setting
2. Always reports Anthropic API key status, even when using OpenAI
3. Contributes to a confusing user experience

---

## Affected Files and Code Locations

### 1. `/kosmos/config.py`

| Line(s) | Component | Issue |
|---------|-----------|-------|
| 34-37 | `ClaudeConfig.api_key` | Required field with no default |
| 735 | `KosmosConfig.claude` | Unconditional `default_factory=ClaudeConfig` |
| 736 | `KosmosConfig.anthropic` | Uses conditional factory (correct pattern) |
| 737 | `KosmosConfig.openai` | Uses conditional factory (correct pattern) |
| 758-773 | `validate_provider_config()` | Never reached due to early validation failure |
| 679-694 | Helper functions | `_optional_openai_config()` and `_optional_anthropic_config()` exist but aren't used for `claude` field |

**Code showing the inconsistency:**

```python
# Line 735 - PROBLEMATIC (unconditional)
claude: ClaudeConfig = Field(default_factory=ClaudeConfig)

# Line 736 - CORRECT PATTERN (conditional)
anthropic: Optional[AnthropicConfig] = Field(default_factory=_optional_anthropic_config)

# Line 737 - CORRECT PATTERN (conditional)
openai: Optional[OpenAIConfig] = Field(default_factory=_optional_openai_config)
```

### 2. `/kosmos/cli/main.py`

| Line(s) | Component | Issue |
|---------|-----------|-------|
| 257-258 | API key check | Hardcoded to check only `ANTHROPIC_API_KEY` |
| 278 | `get_config()` | Fails before this point due to config validation |
| 323 | Error message | Recommends setting `ANTHROPIC_API_KEY` regardless of provider |

### 3. `/tests/requirements/core/test_req_configuration.py`

| Line(s) | Component | Issue |
|---------|-----------|-------|
| 178-196 | `test_provider_specific_validation()` | Tests expect validation failure but don't verify correct behavior |
| 191-194 | OpenAI test case | Currently fails for wrong reason (ClaudeConfig validation, not OpenAI validation) |

---

## Detailed Problem Description

### User Scenario

A user configures Kosmos to use Ollama (an OpenAI-compatible local LLM):

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=ollama  # dummy key for local
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_MODEL=llama3.1:70b
```

When they run `kosmos doctor`:

```bash
$ kosmos doctor
1 validation error for ClaudeConfig
ANTHROPIC_API_KEY
  Field required [type=missing, input_value={}, input_type=dict]
```

### Expected Behavior

- `kosmos doctor` should run diagnostics successfully
- The doctor should check for `OPENAI_API_KEY` when `LLM_PROVIDER=openai`
- `ANTHROPIC_API_KEY` should not be required when not using the Anthropic provider

### Actual Behavior

- Configuration loading fails immediately during Pydantic field initialization
- Error occurs before any provider-specific logic can execute
- User cannot use Kosmos with OpenAI-compatible providers without also setting unused Anthropic credentials

---

## Impact Assessment

| Aspect | Impact | Severity |
|--------|--------|----------|
| **Functionality** | Users cannot use OpenAI provider without dummy Anthropic key | Critical |
| **User Experience** | Confusing error message suggests Anthropic key is always required | High |
| **Documentation** | README and docs suggest OpenAI provider works, but it doesn't out-of-box | High |
| **Testing** | Tests pass for wrong reasons, masking the real issue | Medium |
| **Code Quality** | Inconsistent pattern between `claude` field and `anthropic`/`openai` fields | Medium |

---

## Recommended Changes

### Fix 1: Make `claude` Field Conditionally Optional (Primary Fix)

**File:** `/kosmos/config.py`

**Current (line 735):**
```python
claude: ClaudeConfig = Field(default_factory=ClaudeConfig)  # Backward compatibility
```

**Proposed:**
```python
claude: Optional[ClaudeConfig] = Field(default_factory=_optional_anthropic_config)
```

Or create a dedicated factory function:
```python
def _optional_claude_config() -> Optional[ClaudeConfig]:
    """Create ClaudeConfig only if ANTHROPIC_API_KEY is set."""
    import os
    if os.getenv("ANTHROPIC_API_KEY"):
        return ClaudeConfig()
    return None

# In KosmosConfig:
claude: Optional[ClaudeConfig] = Field(default_factory=_optional_claude_config)
```

### Fix 2: Update Provider Validator for Optional Claude Field

**File:** `/kosmos/config.py`

**Current (lines 767-772):**
```python
elif self.llm_provider == "anthropic":
    if not self.claude.api_key:
        raise ValueError(...)
```

**Proposed:**
```python
elif self.llm_provider == "anthropic":
    if not self.claude or not self.claude.api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic. "
            "Please set ANTHROPIC_API_KEY in your environment or .env file."
        )
```

### Fix 3: Update Doctor Command to Check Provider

**File:** `/kosmos/cli/main.py`

**Current (lines 256-258):**
```python
# Check API key
api_key_ok = bool(os.getenv("ANTHROPIC_API_KEY"))
checks.append(("Anthropic API Key", "Configured" if api_key_ok else "Not set", api_key_ok))
```

**Proposed:**
```python
# Check API key based on provider
llm_provider = os.getenv("LLM_PROVIDER", "anthropic")
if llm_provider == "openai":
    api_key_ok = bool(os.getenv("OPENAI_API_KEY"))
    checks.append(("OpenAI API Key", "Configured" if api_key_ok else "Not set", api_key_ok))
else:
    api_key_ok = bool(os.getenv("ANTHROPIC_API_KEY"))
    checks.append(("Anthropic API Key", "Configured" if api_key_ok else "Not set", api_key_ok))
```

### Fix 4: Update Error Recovery Message

**File:** `/kosmos/cli/main.py`

**Current (lines 322-324):**
```python
console.print("  2. Set ANTHROPIC_API_KEY environment variable")
```

**Proposed:**
```python
if llm_provider == "openai":
    console.print("  2. Set OPENAI_API_KEY environment variable")
else:
    console.print("  2. Set ANTHROPIC_API_KEY environment variable")
```

### Fix 5: Update Tests to Verify Correct Behavior

**File:** `/tests/requirements/core/test_req_configuration.py`

**Current test (lines 191-194):**
```python
# OpenAI provider requires OPENAI_API_KEY
with patch.dict(os.environ, {'LLM_PROVIDER': 'openai'}, clear=True):
    with pytest.raises((ValidationError, ValueError)):
        config = get_config(reload=True)
```

**Proposed additional test:**
```python
def test_openai_provider_does_not_require_anthropic_key(self):
    """Verify OpenAI provider doesn't require ANTHROPIC_API_KEY."""
    from kosmos.config import get_config, reset_config

    reset_config()

    # OpenAI provider should work without ANTHROPIC_API_KEY
    with patch.dict(os.environ, {
        'LLM_PROVIDER': 'openai',
        'OPENAI_API_KEY': 'sk-test-key',
        'OPENAI_MODEL': 'gpt-4'
    }, clear=True):
        config = get_config(reload=True)
        assert config.llm_provider == "openai"
        assert config.openai is not None
        assert config.openai.api_key == 'sk-test-key'

    reset_config()
```

---

## Backward Compatibility Considerations

### The `claude` Field

The `claude` field is marked for "backward compatibility" in the code comment. Making it optional requires careful handling:

1. **Existing code using `config.claude.api_key`** will break if `claude` is `None`
2. **Solution:** Add a property or method that provides safe access:

```python
@property
def active_llm_config(self):
    """Get the configuration for the active LLM provider."""
    if self.llm_provider == "openai":
        return self.openai
    return self.claude or self.anthropic
```

Or add null checks in code that accesses `config.claude`:

```python
# Instead of:
config.claude.api_key

# Use:
config.claude.api_key if config.claude else None
```

### Alternative: Keep `claude` Required but Allow Placeholder

Another approach is to allow a placeholder value for unused configurations:

```python
class ClaudeConfig(BaseSettings):
    api_key: str = Field(
        default="",  # Allow empty string as default
        description="Anthropic API key or empty if not using Anthropic",
        alias="ANTHROPIC_API_KEY"
    )
```

Then validate in `validate_provider_config()`:
```python
elif self.llm_provider == "anthropic":
    if not self.claude.api_key:  # Empty string is falsy
        raise ValueError(...)
```

**Recommendation:** The first approach (making `claude` optional) is cleaner and follows the existing pattern used for `anthropic` and `openai` fields.

---

## Testing Strategy

### Unit Tests Required

1. **Configuration loading with OpenAI provider only**
   - Set `LLM_PROVIDER=openai` and `OPENAI_API_KEY`
   - Verify config loads without error
   - Verify `config.claude` is `None` or has placeholder values

2. **Configuration loading with Anthropic provider only**
   - Set `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY`
   - Verify config loads without error
   - Verify `config.openai` is `None`

3. **Doctor command with OpenAI provider**
   - Verify correct API key check is performed
   - Verify appropriate diagnostic output

4. **Validation error messages**
   - Verify error clearly states which key is missing based on provider

### Integration Tests

1. **End-to-end with Ollama**
   - Configure OpenAI provider pointing to Ollama
   - Run `kosmos doctor`
   - Verify successful diagnostics

---

## Files to Modify Summary

| File | Changes Required | Priority |
|------|-----------------|----------|
| `/kosmos/config.py` | Make `claude` field optional, update validator | Critical |
| `/kosmos/cli/main.py` | Provider-aware API key check and error messages | High |
| `/tests/requirements/core/test_req_configuration.py` | Add tests for OpenAI-only configuration | High |
| `/docs/user/getting-started.md` | Clarify provider-specific requirements (if exists) | Medium |

---

## Conclusion

Issue #22 represents a fundamental flaw in the configuration validation logic that prevents users from using Kosmos with alternative LLM providers. The fix requires:

1. Making the `claude` configuration field conditionally optional (matching the pattern already used for `anthropic` and `openai` fields)
2. Updating the doctor command to check the appropriate API key based on the selected provider
3. Adding comprehensive tests to prevent regression

The recommended changes maintain backward compatibility while enabling the multi-provider functionality that the codebase was designed to support.

---

*Report generated: 2025-11-26*
*Analyzed by: Claude Code Analysis*
