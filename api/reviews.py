"""Investigator review log: an append-only JSONL file on the API host.

Every decision is a new line (who, when, outcome, note, and the model score
the reviewer saw), so earlier decisions are never overwritten. That keeps an
audit trail and is the raw material for real labels: `retraining_label`
maps "confirmed_suspicious" to 1 and "cleared" to 0, while
"needs_more_info" is not a label yet. The file lives under the gitignored
`data/` directory by default (`FRAUD_APP_REVIEWS_PATH` overrides it).
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

OUTCOMES = ("confirmed_suspicious", "cleared", "needs_more_info")
RETRAINING_LABEL = {"confirmed_suspicious": 1, "cleared": 0}
MAX_NOTE_CHARS = 4000
MAX_REVIEWER_CHARS = 120


class ReviewStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def _read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records = []
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def append(
        self,
        domain: str,
        claim_key: str,
        outcome: str,
        note: str,
        reviewer: Optional[str],
        snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {list(OUTCOMES)}, got {outcome!r}")
        record = {
            "domain": domain,
            "claim_key": claim_key,
            "outcome": outcome,
            "retraining_label": RETRAINING_LABEL.get(outcome),
            "note": note[:MAX_NOTE_CHARS],
            "reviewer": (reviewer or "").strip()[:MAX_REVIEWER_CHARS] or None,
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **snapshot,
        }
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        return record

    def history(self, domain: str, claim_key: str) -> list[dict[str, Any]]:
        """All decisions for one claim, oldest first."""
        with self._lock:
            records = self._read()
        return [r for r in records if r["domain"] == domain and r["claim_key"] == claim_key]

    def latest(self, domain: str) -> list[dict[str, Any]]:
        """The current decision per claim in `domain`, newest first."""
        with self._lock:
            records = self._read()
        current: dict[str, dict[str, Any]] = {}
        for record in records:
            if record["domain"] == domain:
                current[record["claim_key"]] = record
        return sorted(current.values(), key=lambda r: r["recorded_at"], reverse=True)
