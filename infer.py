from __future__ import annotations

import argparse

import torch

from stcafnet.data import FoldScaler, SalmonDataset, load_manifest
from stcafnet.utils import resolve_device
from train import make_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    device = resolve_device(config["device"])
    scaler = FoldScaler.from_state_dict(checkpoint["scaler"])
    model = make_model(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    dataset = SalmonDataset(
        load_manifest(args.manifest),
        config["data"]["image_root"],
        config["data"]["enose_root"],
        scaler,
        config["data"]["image_size"],
        config["data"]["enose_length"],
        training=False,
    )
    with torch.no_grad():
        for sample in dataset:
            result = model(
                sample["image"].unsqueeze(0).to(device),
                sample["enose"].unsqueeze(0).to(device),
            )
            values = scaler.inverse_labels(result.predictions.cpu().numpy())[0]
            print(
                f"{sample['sample_id']}: TVC={values[0]:.4f}, "
                f"TVB-N={values[1]:.4f}, TBARS={values[2]:.4f}"
            )


if __name__ == "__main__":
    main()

