"""
Robust JSON parsing utility for LLM responses.

Handles various formats that models (especially local models like Ollama)
may produce, including:
- Direct JSON
- JSON wrapped in markdown code blocks
- JSON with trailing commas
- JSON with single quotes
- Mixed text/JSON responses
"""

import json
import re
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class JSONParseError(Exception):
    """Exception raised when JSON parsing fails after all strategies."""

    def __init__(self, message: str, original_text: str, attempts: int = 0):
        self.message = message
        self.original_text = original_text
        self.attempts = attempts
        super().__init__(f"{message} (tried {attempts} strategies)")


# First-opening to last-closing spans used by the regex extraction strategies
_OBJECT_PATTERN = r'\{[\s\S]*\}'
_ARRAY_PATTERN = r'\[[\s\S]*\]'


def parse_json_response(
    response_text: str,
    schema: Optional[Dict[str, Any]] = None,
    strict: bool = False
) -> Dict[str, Any]:
    """
    Parse a JSON object from model response with multiple fallback strategies.

    Strategies tried in order:
    1. Direct JSON parse
    2. Extract from ```json code blocks (closed)
    2b. Extract from ```json code blocks (unclosed/truncated)
    3. Extract from ``` code blocks
    4. Extract JSON object using regex
    5. Clean common issues (trailing commas, single quotes)

    A strategy only succeeds when it yields an object (dict); a top-level
    array is treated as a failed strategy.

    Args:
        response_text: Raw response text from the model
        schema: Optional expected schema (for validation/hints)
        strict: If True, only try direct parse (no fallbacks)

    Returns:
        Dict[str, Any]: Parsed JSON object

    Raises:
        JSONParseError: If all parsing strategies fail
    """
    return _parse_with_strategies(response_text, _OBJECT_PATTERN, dict, strict)


def parse_json_array_response(
    response_text: str,
    strict: bool = False
) -> List[Any]:
    """
    Parse a JSON array from model response with the same fallback strategies
    as parse_json_response, extracting the first [...] span instead of {...}.

    Args:
        response_text: Raw response text from the model
        strict: If True, only try direct parse (no fallbacks)

    Returns:
        List[Any]: Parsed JSON array

    Raises:
        JSONParseError: If all parsing strategies fail
    """
    return _parse_with_strategies(response_text, _ARRAY_PATTERN, list, strict)


def _parse_with_strategies(
    response_text: str,
    pattern: str,
    expected_type: type,
    strict: bool
) -> Any:
    """Run the parse strategies, accepting only results of expected_type."""
    if not response_text or not str(response_text).strip():
        raise JSONParseError("Empty response", str(response_text or ""), 0)

    response_text = str(response_text)
    text = response_text.strip()
    attempts = 0

    def _loads(candidate: str) -> Any:
        value = json.loads(candidate)
        if not isinstance(value, expected_type):
            raise json.JSONDecodeError(
                f"Expected {expected_type.__name__}, got {type(value).__name__}",
                candidate,
                0
            )
        return value

    # Strategy 1: Direct parse
    attempts += 1
    try:
        return _loads(text)
    except json.JSONDecodeError:
        if strict:
            raise JSONParseError(
                f"JSON decode failed: {text[:100]}...",
                response_text,
                attempts
            )

    # Strategy 2: Extract from ```json code blocks
    attempts += 1
    json_block_match = re.search(r'```json\s*([\s\S]*?)\s*```', text)
    if json_block_match:
        try:
            return _loads(json_block_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Strategy 2b: Extract from unclosed ```json blocks (truncated responses)
    attempts += 1
    unclosed_json_match = re.search(r'```json\s*([\s\S]+)', text)
    if unclosed_json_match and not json_block_match:
        # Try to find complete JSON value within the unclosed block
        block_content = unclosed_json_match.group(1).strip()
        json_in_block = re.search(pattern, block_content)
        if json_in_block:
            try:
                return _loads(json_in_block.group(0))
            except json.JSONDecodeError:
                pass

    # Strategy 3: Extract from ``` code blocks (without json marker)
    attempts += 1
    code_block_match = re.search(r'```\s*([\s\S]*?)\s*```', text)
    if code_block_match:
        block_content = code_block_match.group(1).strip()
        # Skip if it looks like code (has common code markers)
        if not any(marker in block_content for marker in ['def ', 'class ', 'import ', 'function ']):
            try:
                return _loads(block_content)
            except json.JSONDecodeError:
                pass

    # Strategy 4: Extract JSON value using regex (first opening to last closing)
    attempts += 1
    json_obj_match = re.search(pattern, text)
    if json_obj_match:
        try:
            return _loads(json_obj_match.group(0))
        except json.JSONDecodeError:
            # Try with cleaning
            pass

    # Strategy 5: Clean common issues and retry
    attempts += 1
    cleaned = _clean_json_string(text)
    try:
        return _loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 5b: Try cleaning extracted JSON value
    if json_obj_match:
        attempts += 1
        cleaned_obj = _clean_json_string(json_obj_match.group(0))
        try:
            return _loads(cleaned_obj)
        except json.JSONDecodeError:
            pass

    # Strategy 5c: Try cleaning code block content
    if json_block_match:
        attempts += 1
        cleaned_block = _clean_json_string(json_block_match.group(1).strip())
        try:
            return _loads(cleaned_block)
        except json.JSONDecodeError:
            pass

    # All strategies failed
    logger.warning(f"JSON parsing failed after {attempts} attempts")
    logger.debug(f"Original text: {text[:500]}...")

    raise JSONParseError(
        f"Could not parse JSON from response",
        response_text,
        attempts
    )


def _clean_json_string(text: str) -> str:
    """
    Clean common JSON formatting issues.

    Handles:
    - Trailing commas before } or ]
    - Single quotes instead of double quotes
    - Unquoted keys
    - Extra whitespace

    Args:
        text: JSON-like string to clean

    Returns:
        Cleaned JSON string
    """
    if not text:
        return text

    # Remove leading/trailing whitespace
    text = text.strip()

    # Replace single quotes with double quotes (careful with apostrophes)
    # Only replace if it looks like JSON quotes (around keys/values)
    text = re.sub(r"(?<=[{,\[:])\s*'([^']*?)'\s*(?=[,}\]:])", r'"\1"', text)
    text = re.sub(r"(?<=[{,\[])\s*'([^']*?)'\s*(?=:)", r'"\1"', text)

    # Remove trailing commas before } or ]
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)

    # Remove any control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

    return text


def extract_json_value(text: str, key: str) -> Optional[str]:
    """
    Extract a specific key's value from potentially malformed JSON.

    Useful as a last resort when full JSON parsing fails.

    Args:
        text: Response text
        key: Key to extract

    Returns:
        Value if found, None otherwise
    """
    patterns = [
        rf'"{key}"\s*:\s*"([^"]*)"',      # "key": "value"
        rf'"{key}"\s*:\s*(\d+\.?\d*)',     # "key": number
        rf'"{key}"\s*:\s*(true|false)',    # "key": boolean
        rf'{key}\s*:\s*"([^"]*)"',         # key: "value" (unquoted key)
        rf'{key}\s*:\s*([^\n,}}]+)',       # key: value (loose)
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return None
