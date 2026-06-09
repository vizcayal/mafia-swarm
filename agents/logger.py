"""
agents/logger.py — Centralized file + console logger for La Mafia.

Creates timestamped log files in logs/ directory.
All agents import from here for consistent logging.

Usage:
    from agents.logger import get_logger
    logger = get_logger("patron")
    logger.info("Starting batch 1")
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ── Log directory ─────────────────────────────────────────────────────────────
LOG_DIR = ROOT / 'logs'
LOG_DIR.mkdir(exist_ok=True)

# ── Session ID (shared across all loggers in one run) ─────────────────────────
_SESSION_ID = datetime.now().strftime('%Y%m%d_%H%M%S')
_SESSION_LOG = LOG_DIR / f'run_{_SESSION_ID}.log'

# ── Custom formatter ──────────────────────────────────────────────────────────
class MafiaFormatter(logging.Formatter):
    """Format: [HH:MM:SS] [AGENT] LEVEL — message"""
    def format(self, record):
        ts = datetime.now().strftime('%H:%M:%S')
        agent = getattr(record, 'agent', record.name)
        return f"[{ts}] [{agent:>12}] {record.levelname:<7} — {record.getMessage()}"


class CleanConsoleFormatter(logging.Formatter):
    """Console format with emoji icons per level."""
    ICONS = {
        'DEBUG':    '·',
        'INFO':     '·',
        'WARNING':  '⚠️ ',
        'ERROR':    '❌',
        'CRITICAL': '🔥',
    }

    def format(self, record):
        ts = datetime.now().strftime('%H:%M:%S')
        agent = getattr(record, 'agent', record.name)
        icon = self.ICONS.get(record.levelname, '·')
        return f"[{ts}] [{agent}] {icon} {record.getMessage()}"


# ── Shared file handler (one file per session) ────────────────────────────────
_file_handler = logging.FileHandler(_SESSION_LOG, encoding='utf-8')
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(MafiaFormatter())

# ── Console handler ───────────────────────────────────────────────────────────
_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(CleanConsoleFormatter())

# ── Registry ──────────────────────────────────────────────────────────────────
_loggers: dict[str, logging.Logger] = {}


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a named logger that writes to both console and the session log file.
    
    Args:
        name: Agent name (e.g. "patron", "contabile", "ai_client", "modelos")
    
    Returns:
        A configured Logger instance.
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(f'mafia.{name}')
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Avoid duplicate handlers on re-import
    if not logger.handlers:
        logger.addHandler(_file_handler)
        logger.addHandler(_console_handler)

    # Attach agent name for the formatter
    old_factory = logger.makeRecord

    def make_record_with_agent(*a, **kw):
        record = old_factory(*a, **kw)
        record.agent = name
        return record

    logger.makeRecord = make_record_with_agent

    _loggers[name] = logger
    return logger


def get_session_log_path() -> str:
    """Return the path to the current session's log file."""
    return str(_SESSION_LOG)


def log_separator(logger: logging.Logger, title: str = ''):
    """Log a visual separator line."""
    if title:
        logger.info(f"{'═' * 20} {title} {'═' * 20}")
    else:
        logger.info('═' * 62)
