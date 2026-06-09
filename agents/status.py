"""
agents/status.py — Centralized status tracker for La Mafia.
Writes agent execution states to logs/status.json.
"""

import os
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
STATUS_PATH = ROOT / 'logs' / 'status.json'

DEFAULT_STATUS = {
    "phase": "idle",
    "updated_at": "",
    "agents": {
        "patron": {"status": "idle", "message": "Not running"},
        "contabile": {"status": "idle", "message": "Not running"},
        "artigiano": {"status": "idle", "message": "Not running"},
        "selezionatore": {"status": "idle", "message": "Not running"},
        "modelos": {"status": "idle", "message": "Not running"},
        "ensemble": {"status": "idle", "message": "Not running"},
        "libro": {"status": "idle", "message": "Ready"}
    }
}

def update_status(phase=None, agents_update=None):
    """
    Updates the global status in logs/status.json.
    
    Args:
        phase: Optional new phase string (e.g., "bootstrap", "analyzing", "deciding", "dispatching", "evaluating", "completed", "idle").
        agents_update: Optional dict of agent_name -> {status, message} or similar updates.
                       e.g. {"patron": {"status": "thinking", "message": "Deciding proposals..."}}
    """
    # Ensure logs directory exists
    STATUS_PATH.parent.mkdir(exist_ok=True)
    
    # Load existing status or use default
    status = None
    if STATUS_PATH.exists():
        try:
            with open(STATUS_PATH, 'r', encoding='utf-8') as f:
                status = json.load(f)
        except Exception:
            pass
            
    if not status or "agents" not in status:
        status = json.loads(json.dumps(DEFAULT_STATUS))
        
    if phase is not None:
        status["phase"] = phase
        
    if agents_update:
        for agent, data in agents_update.items():
            if agent in status["agents"]:
                status["agents"][agent].update(data)
                
    status["updated_at"] = datetime.now().strftime('%H:%M:%S')
    
    try:
        # Simple atomic write
        temp_path = STATUS_PATH.with_suffix('.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(status, f, indent=2, ensure_ascii=False)
        if STATUS_PATH.exists():
            try:
                os.remove(STATUS_PATH)
            except Exception:
                pass
        os.rename(temp_path, STATUS_PATH)
    except Exception:
        # Fallback to direct write if rename fails due to locks
        try:
            with open(STATUS_PATH, 'w', encoding='utf-8') as f:
                json.dump(status, f, indent=2, ensure_ascii=False)
        except Exception:
            pass # Never crash the orchestrator for status updates
