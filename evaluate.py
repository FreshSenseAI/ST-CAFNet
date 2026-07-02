from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from stcafnet.data import FoldScaler, SalmonDataset, load_manifest
from stcafnet.metrics import regression_metrics
from stcafnet.utils import resolve_device, save_json
from train import make_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="metrics.json")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    device = resolve_device(config["device"])
    model = make_model(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    scaler = FoldScaler.from_state_dict(checkpoint["scaler"])
    dataset = SalmonDataset(
        load_manifest(args.manifest),
        config["data"]["image_root"],
        config["data"]["enose_root"],
        scaler,
        config["data"]["image_size"],
        config["data"]["enose_length"],
        training=False,
    )
    loader = DataLoader(
        dataset, batch_size=config["training"]["batch_size"], shuffle=False
    )
    predictions, targets = [], []
    with torch.no_grad():
        for batch in loader:
            output = model(batch["image"].to(device), batch["enose"].to(device))
            predictions.append(output.predictions.cpu().numpy())
            targets.append(batch["targets"].numpy())
    actual = scaler.inverse_labels(np.concatenate(targets))
    predicted = scaler.inverse_labels(np.concatenate(predictions))
    metrics = regression_metrics(actual, predicted)
    save_json(metrics, args.output)
    print(metrics)


if __name__ == "__main__":
    main()

