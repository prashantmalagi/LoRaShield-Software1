# Morse Correction V2 research package

This package is isolated from the live receiver, V1, V3, and the V1 experimental gate. It does not import production correction modules. Nothing in this package has been trained. The training command requires the explicit `--execute-training` flag and saves only to a caller-provided new path.

## Current implementation state

- Source-first synthetic examples with clean, substitution, burst, symbol insertion/deletion, letter/word boundary edit, and combined corruption labels. The trainer balances batches to include a clean identity case whenever partition counts permit.
- Hash-only exclusion guard supplied by a data custodian. The package does not read the old locked-test labels or V1/V3 manifests. The guard should include hashes for prior corpora and the previously evaluated locked set.
- Separate train, validation, and calibration JSONL files. The training CLI accepts only train and validation files; calibration remains outside gradients and checkpoint selection.
- A separate fresh locked-test source generator and vault writer/one-time reader. Preparation and training do not import it. No fresh locked data is created during implementation.
- A four-layer Transformer encoder with independent symbol-edit, insertion, boundary, clean-risk and candidate-conditioned correction-risk heads. Risk inputs include source context, actual candidate features, alignment, and normalized insert/delete/substitute counts.
- `risk_training.py` defines source-held-out OOF proposal generation. Base-model callbacks receive references only for base-fit train rows; heldout proposal inputs are reference-free. Candidate syntax/alignment is validated before labels are derived. Risk batches are materialized with candidate symbols, alignment, and boundary features plus verified fold provenance; ordinary caller-created batches are rejected. The risk optimizer is restricted to the risk MLP.
- `calibration.py` binds calibration cases to source IDs, calibration dataset fingerprint, and frozen model fingerprint. It temperature-scales risk logits and applies fixed criteria (zero altered-clean accepts, zero observed harmful accepts, P(improve) >= 0.95, P(harm) <= 0.01). If no accepted proposal meets these criteria, result is all-abstain.
- `provenance.py` verifies complete train/validation/calibration row contents, source-ID disjointness, and the manifest's per-partition JSONL fingerprints. Calibration accepts only a manifest object minted by this verifier and requires exact calibration source coverage.
- OOF proposals carry a private verified fold record, validated candidate syntax/alignment, and risk labels. The batch builder creates the model's candidate symbol/alignment/boundary tensors. Risk training rejects ordinary caller-created dictionaries and validates that each batch source is held out from that fold's base fit IDs.
- `checkpointing.py` commits immutable epoch directories containing model and optimizer state. A `CURRENT` pointer is atomically replaced only after all checkpoint files are written; existing epoch bundles cannot be overwritten. `--resume` restores optimizer/model/epoch state and checks full train/validation row-content fingerprints.
- `oof_runner.py` provides a source-grouped OOF fold-training and proposal-generation CLI. It requires a stage-one base checkpoint and explicit `--execute-training`; it writes a separate risk-head checkpoint only after all folds and risk updates complete.
- `calibration_runner.py` produces calibration cases from actual inference proposals/logits, requires a file-verified partition manifest and a checkpoint marked `risk_head_trained`, and writes an all-abstain report when the fixed safety criteria fail.

The boundary head uses deterministic noisy-symbol candidate augmentation during training. Training records candidate symbols, alignment, and boundary labels; validation also measures boundary quality on Stage-A model-predicted candidates. Gold-candidate validation loss remains available as an auxiliary diagnostic.

## Hash guard format

Provide a custodian-generated JSON file outside the repository with this shape:

```json
{
  "excluded_message_sha256": ["<lowercase sha256 of canonical source message>"] ,
  "excluded_word_sha256": ["<lowercase sha256 of each constituent source word>"]
}
```

Include the previously evaluated locked-set source and word hashes without exposing plaintext to the training process. Hashes of short words are not anonymization; protect the guard file and generate/use it under the project's data-handling policy. The package refuses to prepare data without an explicit guard argument.

## Local data preparation

Run from the repository root. The output directory must not exist already.

```powershell
python -m morse_correction_v2.prepare_data `
  --source-guard D:\research\morse_v2_source_guard.json `
  --output-dir D:\research\morse_v2_data `
  --train-sources 5000 --validation-sources 600 --calibration-sources 500
```

This writes `train.jsonl`, `validation.jsonl`, `calibration.jsonl`, and a source fingerprint manifest. It does not create or read locked-test labels.

## Future Google Colab GPU notebook

Use a Colab notebook with a GPU runtime selected. Required notebook checks:

1. Upload/clone the repository and the custodian-provided source guard to the runtime. Keep any future fresh locked-test bundle out of the runtime until model, gate, and report schema are frozen.
2. Confirm TensorFlow sees a GPU and print TensorFlow version, GPU model, and available memory. Colab GPU availability varies; do not assume a particular device.
3. Copy the guard securely into `/content` and keep prepared datasets/checkpoints in a new `/content` output directory or an explicitly versioned Drive directory.
4. Prepare data, then execute the explicit train command below. Do not pass calibration files to the trainer.
5. Save the command, source manifest, package versions, seeds, device, and output-weight SHA-256 with the run.

Notebook cells (the training cell is for a future approved run; it was not run while implementing this package):

```python
import tensorflow as tf
devices = tf.config.list_physical_devices("GPU")
print("TensorFlow:", tf.__version__)
print("GPUs:", devices)
assert devices, "Select Runtime > Change runtime type > GPU before training"
for device in devices:
    tf.config.experimental.set_memory_growth(device, True)
    details = tf.config.experimental.get_device_details(device)
    print("GPU details:", details)
    try:
        print("GPU memory:", tf.config.experimental.get_memory_info(device.name))
    except (ValueError, RuntimeError) as exc:
        print("GPU memory query unavailable:", exc)
```

```python
%cd /content/LoRaShield-Software-main
!python -m morse_correction_v2.prepare_data \
  --source-guard /content/morse_v2_source_guard.json \
  --output-dir /content/morse_v2_data \
  --train-sources 5000 --validation-sources 600 --calibration-sources 500
```

```python
%cd /content/LoRaShield-Software-main
!python -m morse_correction_v2.train \
  --train-data /content/morse_v2_data/train.jsonl \
  --validation-data /content/morse_v2_data/validation.jsonl \
  --output-checkpoint /content/morse_v2_outputs/v2_stage1_checkpoint \
  --epochs 20 --batch-size 32 --execute-training
```

To continue an interrupted run, use the same verified train/validation JSONL and checkpoint directory, and add `--resume`. The trainer checks their full content fingerprints against the committed checkpoint before restoring optimizer state. It appends a new immutable epoch checkpoint and never replaces existing epoch files.

After stage-one training, the OOF runner performs fold fitting, proposal generation, label derivation, feature batching, and risk-head training in one command. The output checkpoint path must be new:

```powershell
python -m morse_correction_v2.oof_runner `
  --train-data D:\research\morse_v2_data\train.jsonl `
  --manifest D:\research\morse_v2_data\manifest.json `
  --base-checkpoint D:\research\morse_v2_outputs\v2_stage1_checkpoint `
  --output-checkpoint D:\research\morse_v2_outputs\v2_risk_checkpoint `
  --folds 5 --fold-epochs 8 --risk-epochs 5 --batch-size 32 `
  --execute-training
```

After the risk checkpoint is created, calibration uses the verified manifest and actual frozen checkpoint weights:

```powershell
python -m morse_correction_v2.calibration_runner `
  --calibration-data D:\research\morse_v2_data\calibration.jsonl `
  --manifest D:\research\morse_v2_data\manifest.json `
  --model-checkpoint D:\research\morse_v2_outputs\v2_risk_checkpoint `
  --output-json D:\research\morse_v2_outputs\calibration_report.json
```

Calibration output does not mark the model automatic-correction-ready. If criteria fail, `selected_all_abstain` remains true. Do not use the locked-test partition in either runner.

The current stage-one trainer optimizes symbol, insertion, boundary, and clean-classification losses only. The OOF risk proposal and risk-head training APIs are implemented as separate research steps, but neither is run by the stage-one command. Calibration and a passing safety policy are also required before correction acceptance can be enabled. No automatic correction is enabled by this package.

For CPU-only package tests, run:

```powershell
python -m pytest tests/test_morse_correction_v2_data.py tests/test_morse_correction_v2_decoding.py tests/test_morse_correction_v2_gate.py tests/test_morse_correction_v2_inference.py tests/test_morse_correction_v2_metrics.py tests/test_morse_correction_v2_model.py tests/test_morse_correction_v2_partition_isolation.py tests/test_morse_correction_v2_risk_boundary.py tests/test_morse_correction_v2_checkpointing.py -q
```

This package remains research-only. Do not prepare data until the source guard has been created and independently verified by authorized custodians. Do not use calibration or locked-test examples in either base-model or risk-head gradients.
