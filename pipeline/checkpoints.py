from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pipeline.dataset import atomic_json


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].to(device="cpu", dtype=torch.uint8))
    if torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all(
            [value.to(device="cpu", dtype=torch.uint8) for value in state["torch_cuda"]]
        )


class CheckpointManager:
    def __init__(self, output_dir: str | Path, *, mode="min", keep_numbered=3):
        if mode not in {"min", "max"}:
            raise ValueError("checkpoint mode must be min or max")
        self.directory = Path(output_dir) / "checkpoints"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.directory / "manifest.json"
        self.mode = mode
        self.keep_numbered = int(keep_numbered)
        self.manifest = self._load_manifest()

    def _load_manifest(self):
        if self.manifest_path.is_file():
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {"latest": None, "best": None, "numbered": []}

    def _save_payload(self, filename: str, payload: dict[str, Any]) -> Path:
        path = self.directory / filename
        temporary = path.with_suffix(path.suffix + ".partial")
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        return path

    def save(self, payload: dict[str, Any], *, metric: float, epoch: int) -> None:
        latest = self._save_payload("latest.ckpt", payload)
        numbered_name = f"epoch_{epoch:04d}.ckpt"
        self._save_payload(numbered_name, payload)
        numbered = [*self.manifest["numbered"], numbered_name]
        while len(numbered) > self.keep_numbered:
            removed = numbered.pop(0)
            (self.directory / removed).unlink(missing_ok=True)
        best = self.manifest["best"]
        better = best is None or (
            metric < best["metric"] if self.mode == "min" else metric > best["metric"]
        )
        if better:
            self._save_payload("best.ckpt", payload)
            best = {"path": "best.ckpt", "metric": metric, "epoch": epoch}
        updated = {
            "latest": {"path": latest.name, "metric": metric, "epoch": epoch},
            "best": best,
            "numbered": numbered,
        }
        atomic_json(self.manifest_path, updated)
        self.manifest = updated

    def load_latest(self, *, compatibility: dict[str, Any], map_location="cpu"):
        latest = self.manifest.get("latest")
        if latest is None:
            return None
        payload = torch.load(
            self.directory / latest["path"], map_location=map_location, weights_only=False
        )
        if payload.get("compatibility") != compatibility:
            raise ValueError("checkpoint model or dataset contract is incompatible")
        return payload
