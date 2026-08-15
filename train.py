from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from pipeline.checkpoints import CheckpointManager, capture_rng_state, restore_rng_state
from pipeline.contracts import load_training_config
from pipeline.dataset import atomic_json, build_dataset_identity
from policy.dataset.act_image_dataset import ACTImageDataset
from policy.policy.act_image_policy import ACTImagePolicy


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch, device):
    return {
        "obs": {key: value.to(device) for key, value in batch["obs"].items()},
        "action": batch["action"].to(device),
        "is_pad": batch["is_pad"].to(device),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an ACT policy")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_training_config(config_path)
    seed_everything(config.seed)
    device = torch.device(config.device)
    dataset_dir = Path(config.dataset_manifest).parent
    train_dataset = ACTImageDataset(
        {},
        str(dataset_dir),
        horizon=config.model.horizon,
        split="train",
        sample_stride=config.sample_stride,
    )
    validation_dataset = train_dataset.get_validation_dataset()
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=config.num_workers,
        persistent_workers=config.num_workers > 0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=validation_dataset.collate_fn,
        num_workers=config.num_workers,
        persistent_workers=config.num_workers > 0,
        pin_memory=device.type == "cuda",
    )
    model_arguments = {
        **config.model.arguments,
        "num_queries": config.model.horizon,
        "camera_names": train_dataset.camera_names,
        "robot_state_dim": 8,
        "action_dim": 8,
    }
    model = ACTImagePolicy(**model_arguments).to(device)
    normalizer = train_dataset.get_normalizer()
    model.set_normalizer(normalizer)
    model.normalizer.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
    )
    total_steps = max(1, len(train_loader) * config.epochs)
    if config.scheduler.type == "cosine":
        warmup = config.scheduler.warmup_steps
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: (
                (step + 1) / max(1, warmup)
                if step < warmup
                else 0.5
                * (
                    1
                    + np.cos(
                        np.pi
                        * (step - warmup)
                        / max(1, total_steps - warmup)
                    )
                )
            ),
        )
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    schema = json.loads((dataset_dir / "schema.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        Path(config.dataset_manifest).read_text(encoding="utf-8")
    )
    splits = json.loads((dataset_dir / "splits.json").read_text(encoding="utf-8"))
    if splits["split_version"] != config.split_version:
        raise ValueError("training split version differs from persisted splits")
    compatibility = {
        "model_type": config.model.type,
        "model_arguments": model_arguments,
        "dataset_schema_version": schema["schema_version"],
        "split_version": splits["split_version"],
        "camera_order": schema["camera_order"],
        "camera_shapes": {
            name: schema["features"][name]["shape"]
            for name in schema["camera_order"]
        },
        "action_convention": schema["action_convention"],
        "gripper_convention": schema["gripper_convention"],
        "sample_stride": config.sample_stride,
        "dataset_identity": build_dataset_identity(manifest, splits),
    }
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    portable_config = config.model_dump(mode="json")
    portable_config["dataset_manifest"] = os.path.relpath(
        config.dataset_manifest, Path.cwd()
    )
    (output_dir / "resolved_training.yaml").write_text(
        yaml.safe_dump(portable_config, sort_keys=False), encoding="utf-8"
    )
    environment = {
        package: importlib.metadata.version(package)
        for package in ("torch", "numpy", "h5py", "pydantic")
    }
    atomic_json(output_dir / "environment.json", environment)
    checkpoints = CheckpointManager(
        output_dir,
        mode=config.checkpoint.mode,
        keep_numbered=config.checkpoint.keep_numbered,
    )
    start_epoch = 0
    global_step = 0
    if config.resume:
        payload = checkpoints.load_latest(compatibility=compatibility, map_location=device)
        if payload is not None:
            model.load_state_dict(payload["model"])
            model.normalizer.load_state_dict(payload["normalizer"])
            model.normalizer.to(device)
            optimizer.load_state_dict(payload["optimizer"])
            scheduler.load_state_dict(payload["scheduler"])
            restore_rng_state(payload["rng"])
            start_epoch = payload["epoch"] + 1
            global_step = payload["global_step"]
    metrics_path = output_dir / "metrics.jsonl"
    for epoch in range(start_epoch, config.epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.compute_loss(move_batch(batch, device))["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1
            train_losses.append(float(loss.detach().cpu()))
        if (epoch + 1) % config.evaluation_interval != 0:
            continue
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in validation_loader:
                value = model.compute_loss(move_batch(batch, device))["loss"]
                val_losses.append(float(value.detach().cpu()))
        metrics = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": float(np.mean(val_losses)),
        }
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, sort_keys=True) + "\n")
        if (epoch + 1) % config.checkpoint.every_n_epochs == 0:
            checkpoints.save(
                {
                    "compatibility": compatibility,
                    "resolved_config": portable_config,
                    "model": model.state_dict(),
                    "normalizer": model.normalizer.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "rng": capture_rng_state(),
                },
                metric=metrics[config.checkpoint.metric],
                epoch=epoch,
            )


if __name__ == "__main__":
    main()
