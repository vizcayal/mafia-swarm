"""
agents/ai_client.py — Shared AI client for La Mafia agents.

Supports three backends (controlled by MAFIA_BACKEND env var):
  1. "cli"    — Claude CLI (`claude -p`), uses your Claude Pro/Max subscription.
  2. "api"    — Anthropic Python SDK, requires ANTHROPIC_API_KEY.
  3. "gemini" — Google Gemini SDK, requires GEMINI_API_KEY (or GOOGLE_API_KEY).

Auto-detection priority: explicit MAFIA_BACKEND > GEMINI_API_KEY > ANTHROPIC_API_KEY > cli.

Default models per backend (override with MAFIA_MODEL):
  - api    → claude-sonnet-4-6
  - cli    → claude-sonnet-4-6
  - gemini → gemini-1.5-flash

Usage:
    from agents.ai_client import call_claude, MODEL
"""

import os
import sys
import json
import shutil
import subprocess
import time
from pathlib import Path

from agents.logger import get_logger

_log = get_logger('ai_client')

# ── Load .env if present ──────────────────────────────────────────────────────
_env_path = Path(__file__).parent.parent / '.env'
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip())

# ── Key detection ─────────────────────────────────────────────────────────────
_anthropic_key = os.environ.get('ANTHROPIC_API_KEY', '')
_has_anthropic = bool(_anthropic_key) and not _anthropic_key.startswith('sk-ant-...')

_gemini_key = os.environ.get('GEMINI_API_KEY', '') or os.environ.get('GOOGLE_API_KEY', '')
_has_gemini = bool(_gemini_key) and not _gemini_key.startswith('your-')

# ── Backend selection ─────────────────────────────────────────────────────────
BACKEND = os.environ.get('MAFIA_BACKEND', '').lower()
if BACKEND not in ('cli', 'api', 'gemini'):
    if _has_gemini:
        BACKEND = 'gemini'
    elif _has_anthropic:
        BACKEND = 'api'
    else:
        BACKEND = 'cli'

# ── Default model per backend ─────────────────────────────────────────────────
_DEFAULT_MODEL = {
    'api':    'claude-sonnet-4-6',
    'cli':    'claude-sonnet-4-6',
    'gemini': 'gemini-2.5-flash',
}
MODEL = os.environ.get('MAFIA_MODEL') or _DEFAULT_MODEL.get(BACKEND, 'claude-sonnet-4-6')

# ── Client init (lazy) ────────────────────────────────────────────────────────
_client = None
if BACKEND == 'api':
    try:
        import anthropic as _anthropic
        _client = _anthropic.Anthropic(api_key=_anthropic_key)
    except ImportError:
        _log.warning("anthropic SDK not installed, falling back to CLI backend.")
        BACKEND = 'cli'
        MODEL = os.environ.get('MAFIA_MODEL') or _DEFAULT_MODEL['cli']
elif BACKEND == 'gemini':
    try:
        import google.generativeai as _genai
        if not _has_gemini:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Get a free key at "
                "https://aistudio.google.com/apikey and add it to .env"
            )
        _genai.configure(api_key=_gemini_key)
        _client = _genai.GenerativeModel(MODEL)
    except ImportError:
        _log.warning(
            "google-generativeai not installed. "
            "Run: uv pip install google-generativeai"
        )
        raise

_log.info(f"Backend={BACKEND} | Model={MODEL}")


def _call_api(system: str, user: str, max_tokens: int) -> str:
    """Call Claude via the Anthropic Python SDK."""
    if _client is None:
        raise RuntimeError(
            "anthropic SDK not installed. Run: pip install anthropic"
        )
    if not _has_anthropic:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Copy .env.example → .env and add your key."
        )

    _log.debug(f"API call: model={MODEL}, max_tokens={max_tokens}")
    t0 = time.time()
    response = _client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    elapsed = time.time() - t0
    text = response.content[0].text
    usage = getattr(response, 'usage', None)
    _log.debug(f"API response ({elapsed:.1f}s): {len(text)} chars")
    if usage:
        _log.debug(f"  tokens: in={usage.input_tokens}, out={usage.output_tokens}")
    return text


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

    _log.debug(f"CLI call: model={MODEL}")
    _log.debug(f"CLI system prompt ({len(system)} chars): {system[:200]}...")
    _log.debug(f"CLI user prompt ({len(user)} chars): {user[:300]}...")

    # Strip ANTHROPIC_API_KEY from the subprocess environment so the CLI
    # uses OAuth / subscription auth instead of API credits
    cli_env = {k: v for k, v in os.environ.items() if k != 'ANTHROPIC_API_KEY'}

    t0 = time.time()
    result = subprocess.run(
        cmd,
        input=user,
        capture_output=True,
        encoding='utf-8',
        errors='replace',
        timeout=480,
        env=cli_env,
    )
    elapsed = time.time() - t0

    raw_output = (result.stdout or '').strip()
    _log.debug(f"CLI raw output ({elapsed:.1f}s, exit={result.returncode}): {len(raw_output)} chars")

    # Parse the JSON envelope — the text lives in the "result" field
    try:
        envelope = json.loads(raw_output)
    except json.JSONDecodeError:
        envelope = None

    # Log usage stats from the CLI envelope
    if envelope:
        usage = envelope.get('usage', {})
        cost = envelope.get('total_cost_usd', 0)
        _log.debug(f"  CLI usage: in={usage.get('input_tokens',0)}, out={usage.get('output_tokens',0)}, cost=${cost:.4f}")

    # The CLI returns exit code 1 for both hard errors and soft errors
    # (e.g. credit issues) but may still provide a JSON response
    if envelope:
        if envelope.get('is_error'):
            err_msg = envelope.get('result', 'unknown error')
            _log.error(f"CLI error: {err_msg}")
            raise RuntimeError(f"Claude CLI error: {err_msg}")
        text = envelope.get('result', '')
        if text:
            _log.debug(f"CLI response text ({len(text)} chars): {text[:300]}...")
            return text

    if result.returncode != 0:
        stderr = (result.stderr or '').strip()
        _log.error(f"CLI exit code {result.returncode}: stderr={stderr[:200]}")
        raise RuntimeError(
            f"Claude CLI exited with code {result.returncode}:\n"
            f"stderr: {stderr}\nstdout: {raw_output[:500]}"
        )

    # Fallback: return raw output
    return raw_output


def _call_gemini(system: str, user: str, max_tokens: int) -> str:
    """Call Google Gemini via the google-generativeai SDK.

    Gemini has no native `system` role on the basic SDK call, so we prepend
    the system prompt to the user message.
    """
    if _client is None:
        raise RuntimeError(
            "google-generativeai SDK not installed. "
            "Run: uv pip install google-generativeai"
        )
    if not _has_gemini:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Get a free key at "
            "https://aistudio.google.com/apikey and add it to .env"
        )

    _log.debug(f"Gemini call: model={MODEL}, max_tokens={max_tokens}")
    t0 = time.time()
    prompt = f"{system}\n\n---\n\n{user}"
    response = _client.generate_content(
        prompt,
        generation_config={
            "max_output_tokens": max_tokens,
            "temperature": 0.7,
        },
    )
    elapsed = time.time() - t0
    text = response.text or ''
    usage = getattr(response, 'usage_metadata', None)
    _log.debug(f"Gemini response ({elapsed:.1f}s): {len(text)} chars")
    if usage:
        _log.debug(
            f"  tokens: in={usage.prompt_token_count}, "
            f"out={usage.candidates_token_count}"
        )
    return text


def call_claude(system: str, user: str, max_tokens: int = 4096) -> str:
    """
    Single-shot LLM call. Returns the text content of the response.

    Backend is selected via MAFIA_BACKEND env var:
      - "cli"    → Claude CLI (claude -p), free with Pro/Max subscription
      - "api"    → Anthropic SDK, requires ANTHROPIC_API_KEY
      - "gemini" → Google Gemini SDK, requires GEMINI_API_KEY

    Auto-detects based on which API key is set; falls back to CLI.
    Name kept as `call_claude` for backward compatibility.
    """
    _log.info(f"Calling LLM ({BACKEND}): {len(user)} char prompt")
    try:
        if BACKEND == 'api':
            result = _call_api(system, user, max_tokens)
        elif BACKEND == 'gemini':
            result = _call_gemini(system, user, max_tokens)
        else:
            result = _call_cli(system, user, max_tokens)
        _log.info(f"LLM responded: {len(result)} chars")
        return result
    except Exception as e:
        _log.error(f"LLM call failed: {e}")
        raise


def parse_json_response(text: str) -> dict | list:
    """
    Extract JSON from an LLM response that may contain surrounding prose.
    Tries: full text → fenced ```json block → balanced-brace substring →
    truncated-JSON repair (close open strings/brackets).
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
            try:
                end = text.index('```', start)
                candidate = text[start:end].strip()
            except ValueError:
                candidate = text[start:].strip()
            try:
                return json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                pass
            # Try repairing the fenced candidate if it was truncated
            repaired = _repair_truncated_json(candidate)
            if repaired is not None:
                return repaired
    # Try slicing from first { or [ to the matching close
    for open_ch, close_ch in (('{', '}'), ('[', ']')):
        if open_ch in text:
            start = text.index(open_ch)
            candidate = text[start:]
            try:
                return json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                pass
            repaired = _repair_truncated_json(candidate)
            if repaired is not None:
                return repaired
    raise ValueError(f"Could not parse JSON from LLM response:\n{text[:300]}")


def _repair_truncated_json(text: str):
    """Best-effort repair of JSON that was cut off mid-stream.

    Strategy: track string/bracket state, then close any open string and
    append the matching closing brackets in reverse order.
    """
    stack = []
    in_string = False
    escape = False
    last_complete = 0  # index after last successfully parsed structural unit
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in '{[':
            stack.append('}' if ch == '{' else ']')
        elif ch in '}]':
            if stack and stack[-1] == ch:
                stack.pop()
                last_complete = i + 1
            else:
                return None  # malformed beyond repair
    if not stack and not in_string:
        # already balanced; original parse must have failed for another reason
        try:
            return json.loads(text[:last_complete] if last_complete else text)
        except json.JSONDecodeError:
            return None
    # Truncate any trailing partial token (e.g. half-written key) by cutting
    # back to the last comma or opening bracket before the unfinished region.
    repaired = text
    if in_string:
        repaired += '"'
    # Drop trailing comma + partial fragment after the last balanced char
    tail = repaired[last_complete:] if last_complete else repaired
    # Find last safe cut: position of last comma or bracket open in tail
    safe = max(tail.rfind(','), tail.rfind('{'), tail.rfind('['), tail.rfind(':'))
    if safe > 0:
        # Only trim if the segment after `safe` looks like an unfinished value
        segment = tail[safe + 1:].strip()
        if segment and not segment.endswith(('"', '}', ']', 'e', 'l')):
            repaired = (repaired[:last_complete] if last_complete else '') + tail[:safe]
    # Append closing brackets in reverse stack order
    repaired += ''.join(reversed(stack))
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return None
