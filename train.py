from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from stcafnet.data import FoldScaler, SalmonDataset, load_manifest, split_manifest
from stcafnet.metrics import regression_metrics
from stcafnet.model import STCAFNet
from stcafnet.utils import load_config, resolve_device, save_json, seed_everything


def make_model(config: dict) -> STCAFNet:
    model = config["model"]
    return STCAFNet(
        num_sensors=config["data"]["num_sensors"],
        feature_dim=model["feature_dim"],
        cbam_reduction=model["cbam_reduction"],
        enose_channels=tuple(model["enose_channels"]),
        lstm_hidden=model["lstm_hidden"],
        lstm_layers=model["lstm_layers"],
        lstm_dropout=model["lstm_dropout"],
        attention_heads=model["attention_heads"],
        fusion_dropout=model["fusion_dropout"],
        head_dropout=model["head_dropout"],
        pretrained=model["pretrained"],
    )


def make_loaders(config, train_frame, val_frame, scaler):
    data = config["data"]
    common = dict(
        image_root=data["image_root"],
        enose_root=data["enose_root"],
        scaler=scaler,
        image_size=data["image_size"],
        enose_length=data["enose_length"],
    )
    train_set = SalmonDataset(train_frame, training=True, **common)
    val_set = SalmonDataset(val_frame, training=False, **common)
    kwargs = dict(
        batch_size=config["training"]["batch_size"],
        num_workers=data["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    return (
        DataLoader(train_set, shuffle=True, drop_last=False, **kwargs),
        DataLoader(val_set, shuffle=False, drop_last=False, **kwargs),
    )


def optimizer_for_stage(model, config, warmup):
    training = config["training"]
    model.freeze_visual_backbone(warmup)
    if warmup:
        groups = [{"params": [p for p in model.parameters() if p.requires_grad],
                   "lr": training["warmup_lr"]}]
    else:
        backbone_ids = {id(p) for p in model.visual.backbone.parameters()}
        groups = [
            {"params": list(model.visual.backbone.parameters()),
             "lr": training["backbone_lr"]},
            {"params": [p for p in model.parameters() if id(p) not in backbone_ids],
             "lr": training["other_lr"]},
        ]
    return torch.optim.AdamW(groups, weight_decay=training["weight_decay"])


@torch.no_grad()
def validate(model, loader, device, scaler):
    model.eval()
    losses, predictions, targets = [], [], []
    for batch in loader:
        output = model(batch["image"].to(device), batch["enose"].to(device))
        target = batch["targets"].to(device)
        losses.append(model.uncertainty_weighted_loss(output.predictions, target).item())
        predictions.append(output.predictions.cpu().numpy())
        targets.append(target.cpu().numpy())
    predicted = scaler.inverse_labels(np.concatenate(predictions))
    actual = scaler.inverse_labels(np.concatenate(targets))
    return float(np.mean(losses)), regression_metrics(actual, predicted)


def run_stage(
    model,
    train_loader,
    val_loader,
    device,
    scaler,
    config,
    epochs,
    warmup,
    early_stopping,
):
    optimizer = optimizer_for_stage(model, config, warmup)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )
    best_state, best_loss, stale = None, float("inf"), 0
    patience = config["training"]["patience"]
    min_delta = config["training"]["min_delta"]
    for epoch in range(epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"{'warmup' if warmup else 'finetune'} {epoch + 1}/{epochs}")
        for batch in progress:
            optimizer.zero_grad(set_to_none=True)
            output = model(batch["image"].to(device), batch["enose"].to(device))
            loss = model.uncertainty_weighted_loss(
                output.predictions, batch["targets"].to(device)
            )
            loss.backward()
            optimizer.step()
            progress.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()
        val_loss, metrics = validate(model, val_loader, device, scaler)
        print(f"validation loss={val_loss:.6f} metrics={metrics}")
        if val_loss < best_loss - min_delta:
            best_loss, stale = val_loss, 0
            best_state = copy.deepcopy(model.state_dict())
        elif early_stopping:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--fold", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    fold = config["data"]["fold"] if args.fold is None else args.fold
    seed_everything(config["seed"])
    device = resolve_device(config["device"])
    frame = load_manifest(config["data"]["manifest"])
    test, folds = split_manifest(
        frame,
        config["data"]["test_fraction"],
        config["data"]["num_folds"],
        config["seed"],
    )
    train_frame, val_frame = folds[fold]
    scaler = FoldScaler().fit(train_frame, Path(config["data"]["enose_root"]))
    train_loader, val_loader = make_loaders(
        config, train_frame, val_frame, scaler
    )
    model = make_model(config).to(device)
    run_stage(
        model, train_loader, val_loader, device, scaler, config,
        config["training"]["warmup_epochs"], True, False
    )
    best_loss = run_stage(
        model, train_loader, val_loader, device, scaler, config,
        config["training"]["finetune_epochs"], False, True
    )
    output_dir = Path(config["output_dir"]) / f"fold_{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "scaler": scaler.state_dict(),
            "config": config,
            "fold": fold,
            "validation_loss": best_loss,
        },
        output_dir / "best.pt",
    )
    test.to_csv(output_dir / "test_manifest.csv", index=False)
    val_frame.to_csv(output_dir / "validation_manifest.csv", index=False)
    save_json({"validation_loss": best_loss}, output_dir / "training_summary.json")


if __name__ == "__main__":
    main()
