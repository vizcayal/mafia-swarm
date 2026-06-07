"""
agents/ai_client.py — Shared Anthropic client for La Mafia AI agents.

Usage:
    from agents.ai_client import call_claude, MODEL

All agents import from here so the model and key config live in one place.
"""

import os
import json
from pathlib import Path

# ── Load .env if present ──────────────────────────────────────────────────────
_env_path = Path(__file__).parent.parent / '.env'
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip())

# ── Model config ──────────────────────────────────────────────────────────────
MODEL = os.environ.get('MAFIA_MODEL', 'claude-opus-4-6')

# ── Client ────────────────────────────────────────────────────────────────────
try:
    import anthropic as _anthropic
    _client = _anthropic.Anthropic(api_key=os.environ.get('ANTHROPIC_API_KEY'))
except ImportError:
    _client = None


def call_claude(system: str, user: str, max_tokens: int = 4096) -> str:
    """
    Single-shot Claude call. Returns the text content of the response.
    Raises RuntimeError if the SDK is not installed or the key is missing.
    """
    if _client is None:
        raise RuntimeError(
            "anthropic SDK not installed. Run: pip install anthropic"
        )
    key = os.environ.get('ANTHROPIC_API_KEY', '')
    if not key or key.startswith('sk-ant-...'):
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Copy .env.example → .env and add your key."
        )

    response = _client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text


def parse_json_response(text: str) -> dict | list:
    """
    Extract JSON from a Claude response that may contain surrounding prose.
    Tries the full text first, then looks for a ```json block.
    """
    text = text.strip()
    # Try raw JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try fenced block
    for fence in ('```json', '```'):
        if fence in text:
            start = text.index(fence) + len(fence)
            end   = text.index('```', start)
            try:
                return json.loads(text[start:end].strip())
            except (json.JSONDecodeError, ValueError):
                pass
    raise ValueError(f"Could not parse JSON from Claude response:\n{text[:300]}")
