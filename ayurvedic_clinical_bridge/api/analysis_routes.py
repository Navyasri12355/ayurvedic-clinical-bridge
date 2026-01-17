from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List
import subprocess
import threading
import uuid
import time
import os
import shutil
import json

router = APIRouter(prefix="/analysis", tags=["analysis"])

RESULT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "analysis_results"))
os.makedirs(RESULT_DIR, exist_ok=True)


class RunRequest(BaseModel):
    model: Optional[str] = "both"  # bilstm | transformer | both
    epochs: Optional[int] = 2
    tiny: Optional[bool] = False


runs_status = {}  # run_id -> status/info


def _run_script_async(run_id: str, model: str, epochs: int, tiny: bool):
    try:
        out_file = os.path.join(RESULT_DIR, f"sequence_tagging_comparison_{run_id}.json")
        cmd = ["python", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "compare_sequence_tagging.py")), "--model", model, "--epochs", str(epochs), "--out", out_file]
        if tiny:
            cmd.append("--tiny")
        runs_status[run_id]["started_at"] = time.time()
        runs_status[run_id]["status"] = "running"
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            # copy as latest
            latest = os.path.join(RESULT_DIR, "sequence_tagging_comparison.json")
            shutil.copy(out_file, latest)
            runs_status[run_id]["status"] = "completed"
            runs_status[run_id]["completed_at"] = time.time()
            runs_status[run_id]["output"] = out_file
        else:
            runs_status[run_id]["status"] = "failed"
            runs_status[run_id]["error"] = proc.stderr or proc.stdout
    except Exception as e:
        runs_status[run_id]["status"] = "failed"
        runs_status[run_id]["error"] = str(e)


@router.post("/sequence-tagging", status_code=202)
async def start_sequence_tagging(req: RunRequest):
    """Start a sequence tagging comparison run in background."""
    run_id = uuid.uuid4().hex[:8]
    runs_status[run_id] = {"status": "queued", "model": req.model, "epochs": req.epochs, "tiny": req.tiny}
    thread = threading.Thread(target=_run_script_async, args=(run_id, req.model, req.epochs, req.tiny), daemon=True)
    thread.start()
    return {"run_id": run_id, "status": "queued"}


@router.get("/sequence-tagging")
async def get_latest_sequence_tagging(run_id: Optional[str] = None):
    """Get results for a specific run_id, or the latest run if none provided."""
    if run_id:
        path = os.path.join(RESULT_DIR, f"sequence_tagging_comparison_{run_id}.json")
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    else:
        path = os.path.join(RESULT_DIR, "sequence_tagging_comparison.json")
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="No completed runs available")
    with open(path) as f:
        data = json.load(f)
    return data


@router.get("/sequence-tagging/runs")
async def list_runs() -> List[dict]:
    """List recent runs and statuses."""
    out = []
    for k, v in runs_status.items():
        out.append({"run_id": k, **v})
    return out
