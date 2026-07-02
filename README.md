# ST-CAFNet

Official PyTorch implementation of **ST-CAFNet: A visual-olfactory multimodal
cross-attention fusion network for freshness prediction of cold
plasma-treated salmon**.

ST-CAFNet predicts three freshness indicators from a paired RGB image and a
120-s, 10-sensor electronic-nose response sequence:

- TVC (log10 CFU/g)
- TVB-N (mg/100 g)
- TBARS (mg MDA/kg)

## Architecture

The implementation follows the manuscript:

1. EfficientNet-B0 visual backbone with CBAM on the final 1280 x 7 x 7 map.
2. Two 1D-CNN blocks (64 and 128 channels), followed by a two-layer
   bidirectional LSTM with 128 hidden units per direction.
3. Two symmetric cross-modal attention streams at a total dimension of 512
   (8 heads × 64 dimensions per head). Each global modality feature attends
   to the temporal or spatial token sequence from the opposite modality.
4. Dimension-wise vector-gated fusion and a
   `Linear(1024,512)-GELU-Dropout(0.1)-Linear(512,512)` feed-forward block.
5. Shared-private regression head for TVC, TVB-N and TBARS.
6. Homoscedastic uncertainty-weighted multi-task MSE.

## Installation

Python 3.10, PyTorch 2.2.2, and torchvision 0.17.2 are used in the manuscript.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## Data layout

The data are not included in this repository. Each E-nose file must be a
NumPy array with shape `[120, 10]`. Create `data/manifest.csv` using
`manifest_schema.csv` as the schema:

```text
data/
  images/
    CK_D0_001.jpg
  enose/
    CK_D0_001.npy
  manifest.csv
```

Each manifest row represents one independent sample unit and binds one RGB
image, one averaged E-nose record, and all three labels. `sample_id` must be
unique. The three repeated E-nose acquisitions described in the manuscript
must be averaged before creating the `.npy` file.

## Leakage-safe partitioning

Samples are stratified by the combination of treatment and storage day. The
independent test set contains 15% of each stratum. Five-fold cross-validation
is performed only on the remaining 85% development set.

```bash
python -m scripts.create_splits --manifest data/manifest.csv
```

E-nose min-max statistics and label z-score statistics are fitted on the
training portion of each fold only. Validation and test samples never
contribute to these statistics.

## Training

Edit paths and hardware settings in `configs/default.yaml`, then run:

```bash
python train.py --config configs/default.yaml --fold 0
```

Training has two stages:

- 20 warm-up epochs with the EfficientNet-B0 backbone frozen.
- Up to 80 end-to-end fine-tuning epochs with learning rates `1e-5` for the
  backbone and `1e-4` for other modules.

AdamW uses weight decay `1e-4`, cosine annealing, and early stopping with
patience 15. Repeat folds 0 through 4 and report independent-test metrics as
mean and standard deviation across folds.

The batch size is 16, as specified in Table S3. The random seed is fixed in
the configuration file so that data partitions and parameter initialization
can be reproduced.

## Evaluation and inference

```bash
python evaluate.py --checkpoint outputs/fold_0/best.pt \
  --manifest outputs/fold_0/test_manifest.csv \
  --output outputs/fold_0/test_metrics.json

python infer.py --checkpoint outputs/fold_0/best.pt \
  --manifest data/manifest.csv
```

`evaluate.py` reports R², RMSE, MAE and RPD after inverse-transforming each
target to its original unit.

## Verification

```bash
pip install pytest
pytest -q
```

The tests check the full multimodal forward pass, output dimensions,
uncertainty-weighted loss, and sample-level isolation of the hold-out and
cross-validation splits.

## License

MIT. The benchmark data may be subject to separate academic and
non-commercial-use terms.
