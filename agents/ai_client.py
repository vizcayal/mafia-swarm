"""
agents/ai_client.py — Shared AI client for La Mafia agents.

Supports two backends (controlled by MAFIA_BACKEND env var):
  1. "cli"  — Claude CLI (`claude -p`), uses your Claude Pro/Max subscription.
  2. "api"  — Anthropic Python SDK, requires ANTHROPIC_API_KEY.

Auto-detection: If ANTHROPIC_API_KEY is set and valid, defaults to "api".
Otherwise, falls back to "cli" (requires `claude` on PATH).

Usage:
    from agents.ai_client import call_claude, MODEL
"""

import os
import sys
import json
import shutil
import subprocess
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
MODEL = os.environ.get('MAFIA_MODEL', 'claude-sonnet-4-6')

# ── Backend detection ─────────────────────────────────────────────────────────
_api_key = os.environ.get('ANTHROPIC_API_KEY', '')
_has_valid_key = bool(_api_key) and not _api_key.startswith('sk-ant-...')

# Explicit override via env var, otherwise auto-detect
BACKEND = os.environ.get('MAFIA_BACKEND', '').lower()
if BACKEND not in ('cli', 'api'):
    BACKEND = 'api' if _has_valid_key else 'cli'

# ── API client (lazy) ─────────────────────────────────────────────────────────
_client = None
if BACKEND == 'api':
    try:
        import anthropic as _anthropic
        _client = _anthropic.Anthropic(api_key=_api_key)
    except ImportError:
        print("⚠️  anthropic SDK not installed, falling back to CLI backend.",
              file=sys.stderr)
        BACKEND = 'cli'


def _call_api(system: str, user: str, max_tokens: int) -> str:
    """Call Claude via the Anthropic Python SDK."""
    if _client is None:
        raise RuntimeError(
            "anthropic SDK not installed. Run: pip install anthropic"
        )
    if not _has_valid_key:
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


def _call_cli(system: str, user: str, max_tokens: int) -> str:
    """Call Claude via the `claude` CLI in print mode (-p)."""
    claude_bin = shutil.which('claude')
    if claude_bin is None:
        raise RuntimeError(
            "Claude CLI not found on PATH. Install it or set MAFIA_BACKEND=api "
            "with a valid ANTHROPIC_API_KEY."
        )

    cmd = [
        claude_bin,
        '-p',
        '--model', MODEL,
        '--system-prompt', system,
        '--output-format', 'json',
        '--no-session-persistence',
    ]

    # Strip ANTHROPIC_API_KEY from the subprocess environment so the CLI
    # uses OAuth / subscription auth instead of API credits
    cli_env = {k: v for k, v in os.environ.items() if k != 'ANTHROPIC_API_KEY'}

    result = subprocess.run(
        cmd,
        input=user,
        capture_output=True,
        encoding='utf-8',
        errors='replace',
        timeout=180,
        env=cli_env,
    )

    raw_output = (result.stdout or '').strip()

    # Parse the JSON envelope — the text lives in the "result" field
    try:
        envelope = json.loads(raw_output)
    except json.JSONDecodeError:
        envelope = None

    # The CLI returns exit code 1 for both hard errors and soft errors
    # (e.g. credit issues) but may still provide a JSON response
    if envelope:
        if envelope.get('is_error'):
            raise RuntimeError(
                f"Claude CLI error: {envelope.get('result', 'unknown error')}"
            )
        text = envelope.get('result', '')
        if text:
            return text

    if result.returncode != 0:
        stderr = (result.stderr or '').strip()
        raise RuntimeError(
            f"Claude CLI exited with code {result.returncode}:\n"
            f"stderr: {stderr}\nstdout: {raw_output[:500]}"
        )

    # Fallback: return raw output
    return raw_output


def call_claude(system: str, user: str, max_tokens: int = 4096) -> str:
    """
    Single-shot Claude call. Returns the text content of the response.

    Backend is selected via MAFIA_BACKEND env var:
      - "cli" → Claude CLI (claude -p), free with Pro/Max subscription
      - "api" → Anthropic SDK, requires ANTHROPIC_API_KEY

    Auto-detects if neither is set.
    """
    if BACKEND == 'api':
        return _call_api(system, user, max_tokens)
    else:
        return _call_cli(system, user, max_tokens)


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
