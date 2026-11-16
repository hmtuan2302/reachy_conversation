import json
import logging
from pathlib import Path

from reachy_demo.ops.logging import configure_logging, trace_id


def test_jsonl_has_trace_extra_and_masks_secret(tmp_path: Path) -> None:
    path = configure_logging(tmp_path, "INFO", secrets=["sk-secret"])
    token = trace_id.set("t-1")
    try:
        logging.getLogger("x").info("key sk-secret", extra={"tool": "find_object"})
    finally:
        trace_id.reset(token)
    for h in logging.getLogger().handlers:
        h.flush()
    rec = json.loads(path.read_text().splitlines()[-1])
    assert rec["trace_id"] == "t-1" and rec["tool"] == "find_object"
    assert "sk-secret" not in json.dumps(rec) and "***" in rec["msg"]
