# Morse Correction V2 Operational Handoff

**Current state: NO-GO for data preparation and training.** No data was prepared for this handoff. A source guard has not been supplied or independently verified. The commands below describe future authorized operation; none were run here.

## 1. Source guard custody and verification

- An authorized data custodian must produce the guard outside the training workspace. It must contain lowercase SHA-256 hashes for every excluded canonical source message and each source word, including the previously evaluated locked-set sources and all other excluded prior corpora.
- Hash canonical uppercase A-Z/0-9 messages with single ASCII spaces, using SHA-256 over UTF-8 bytes. Include provenance: source inventory, counts, canonicalization/hash tool versions, date, and guard digest. Keep the guard private; short-word hashes can reveal their source words.
- A second authorized custodian must independently recompute the message and word hash sets and compare their contents, counts, and aggregate digest with the provenance record.
- The package checks hash syntax and nonemptiness; it cannot establish guard completeness or custodian authority. Do not run preparation until the independent verification is complete. Never substitute arbitrary hashes.

Expected JSON shape:

```json
{
  "excluded_message_sha256": ["<64 lowercase hex characters>"],
  "excluded_word_sha256": ["<64 lowercase hex characters>"]
}
```

## 2. Data preparation command

Run from the repository root after source-guard verification. The output directory must not already exist; its parent directory must be writable.

```powershell
python -m morse_correction_v2.prepare_data `
  --source-guard D:\research\morse_v2_source_guard.json `
  --output-dir D:\research\morse_v2_data `
  --train-sources 5000 --validation-sources 600 --calibration-sources 500
```

The command creates 12 corruption variants per source in each partition and writes:

- `train.jsonl`
- `validation.jsonl`
- `calibration.jsonl`
- `manifest.json`

The manifest records partition source fingerprints, counts/seeds, and a `partition_content` entry for each JSONL file with its content SHA-256 and source IDs. Preparation does not create or read locked-test data.

## 3. Verify partition contents before use

From the repository root, verify the manifest against the actual three JSONL files:

```powershell
python -c "from morse_correction_v2.provenance import verify_manifest_files; m=verify_manifest_files(r'D:\research\morse_v2_data\manifest.json'); print(m.manifest_fingerprint, dict(m.source_ids), dict(m.content_fingerprints))"
```

This checks actual file-content fingerprints, declared source IDs, and source-ID disjointness among train, validation, and calibration. Stop on any mismatch. Do not edit a manifest to make a mismatch pass.

## 4. OOF fold training and risk-head command

After the approved stage-one checkpoint exists, the OOF runner performs source-grouped fold fitting, reference-free heldout proposal generation, candidate validation, label derivation, risk-tensor construction, and risk-head training. All fold variants remain grouped by source ID. Its output checkpoint directory must be new.

```powershell
python -m morse_correction_v2.oof_runner `
  --train-data D:\research\morse_v2_data\train.jsonl `
  --manifest D:\research\morse_v2_data\manifest.json `
  --base-checkpoint D:\research\morse_v2_outputs\v2_stage1_checkpoint `
  --output-checkpoint D:\research\morse_v2_outputs\v2_risk_checkpoint `
  --folds 5 --fold-epochs 8 --risk-epochs 5 --batch-size 32 `
  --execute-training
```

The OOF runner requires the explicit optimizer-update flag. It rejects non-train input and refuses an existing output checkpoint. A fold failure occurs before the risk checkpoint is written; fix the cause and rerun to recompute folds. Invalid per-example proposals are recorded and excluded from risk batches.

## 5. Google Colab GPU setup and memory checks

Select a GPU runtime, clone/upload the repository, and adjust `%cd` if the repository directory differs. Run this cell before any future approved training:

```python
import tensorflow as tf
devices = tf.config.list_physical_devices("GPU")
print("TensorFlow:", tf.__version__)
print("GPUs:", devices)
assert devices, "Select Runtime > Change runtime type > GPU before training"
for device in devices:
    tf.config.experimental.set_memory_growth(device, True)
    print("GPU details:", tf.config.experimental.get_device_details(device))
logical_devices = tf.config.list_logical_devices("GPU")
print("Logical GPUs:", logical_devices)
assert logical_devices, "TensorFlow did not initialize a logical GPU"
for device in logical_devices:
    try:
        print("TensorFlow allocator memory:",
              tf.config.experimental.get_memory_info(device.name))
    except (ValueError, RuntimeError) as exc:
        print("GPU memory query unavailable:", exc)
```

The memory query reports TensorFlow allocator usage/peak, not necessarily the device’s total VRAM. Keep the locked-test bundle out of Colab until final evaluation is approved.

## 6. Stage-one training and checkpoint resume

After data, manifest, and GPU checks pass and training is explicitly approved, the supported command is:

```powershell
python -m morse_correction_v2.train `
  --train-data D:\research\morse_v2_data\train.jsonl `
  --validation-data D:\research\morse_v2_data\validation.jsonl `
  --output-checkpoint D:\research\morse_v2_outputs\v2_stage1_checkpoint `
  --epochs 20 --batch-size 32 --execute-training
```

`--output-checkpoint` (also accepted as the legacy alias `--output-weights`) names a **new checkpoint directory**. The trainer refuses an existing run unless `--resume` is provided. It stores immutable epoch checkpoints with model and optimizer state and advances `CURRENT` atomically.

To resume the same run against the same train and validation contents:

```powershell
python -m morse_correction_v2.train `
  --train-data D:\research\morse_v2_data\train.jsonl `
  --validation-data D:\research\morse_v2_data\validation.jsonl `
  --output-checkpoint D:\research\morse_v2_outputs\v2_stage1_checkpoint `
  --epochs 20 --batch-size 32 --execute-training --resume
```

Resume verifies full train/validation row-content fingerprints and restores optimizer state. Each epoch checkpoint is immutable; do not rename an unrelated run to force resume.

Stage one trains symbol-edit, insertion, boundary, and clean-classification objectives. It does not train the risk head or make the artifact automatic-correction-ready.

## 7. Validation and calibration

- Stage-one validation is supplied only through `--validation-data`. The trainer uses it for checkpoint selection and records boundary metrics on model-predicted symbol candidates in checkpoint metadata. Calibration data is not accepted by this trainer.
- Run the prediction/calibration runner only with the risk checkpoint and the verified manifest:

```powershell
python -m morse_correction_v2.calibration_runner `
  --calibration-data D:\research\morse_v2_data\calibration.jsonl `
  --manifest D:\research\morse_v2_data\manifest.json `
  --model-checkpoint D:\research\morse_v2_outputs\v2_risk_checkpoint `
  --output-json D:\research\morse_v2_outputs\calibration_report.json
```

- The runner checks the actual calibration file contents, exact source coverage, and risk-trained checkpoint metadata. It generates proposals/logits from the checkpoint, labels valid candidates against calibration references, and binds the report to the verified manifest and actual model-weight fingerprint. Criteria are fixed: calibrated P(improve) >= 0.95, P(harm) <= 0.01, zero altered-clean accepts, and zero observed harmful accepts; otherwise it selects all-abstain. The report path must not exist.

## 8. Fresh locked-test approval

Keep locked labels inaccessible during source guard creation, preparation, training, OOF generation, validation, and calibration. A fresh locked set is a separate one-time process. Before creating or reading it, obtain explicit approval for the locked-set custodian workflow and freeze the model, correction policy, and report schema. The freeze manifest must bind the fresh dataset SHA-256 plus model, policy, and report-schema SHA-256 digests. The API `load_fresh_locked_once` refuses a second evaluation and validates those digests. Do not call it before approval; do not use locked labels for tuning.

## 9. Common errors and recovery

- **Guard missing/empty or not independently verified:** stop and return to the custodians; do not make a replacement guard.
- **Preparation output directory exists:** choose a new versioned output directory. The command refuses overwrite.
- **Partition manifest/content mismatch:** stop; preserve the files and investigate provenance. Do not edit fingerprints by hand.
- **Checkpoint path already exists:** use `--resume` only for that run with identical train/validation content; otherwise choose a new checkpoint directory.
- **Interrupted epoch promotion:** rerun the same training command with `--resume`. A fully promoted epoch with an interrupted `CURRENT` update is recovered automatically; incomplete staging files are ignored.
- **Resume fingerprint mismatch:** use the exact original JSONL contents or start a separately named run. Do not bypass the fingerprint check.
- **Invalid OOF candidate/alignment:** inspect `OOFGenerationResult.rejected`; fix the fold proposer, not the label or validator.
- **Calibration chooses all-abstain:** retain abstention and report that automatic correction is not suitable. Do not loosen thresholds after seeing locked-test results.

Complete V2 suite:

```powershell
python -m pytest tests/test_morse_correction_v2_data.py tests/test_morse_correction_v2_decoding.py tests/test_morse_correction_v2_gate.py tests/test_morse_correction_v2_inference.py tests/test_morse_correction_v2_metrics.py tests/test_morse_correction_v2_model.py tests/test_morse_correction_v2_partition_isolation.py tests/test_morse_correction_v2_risk_boundary.py tests/test_morse_correction_v2_checkpointing.py -q
```

Full project suite:

```powershell
python -m pytest tests -q
```

## GO / NO-GO

- [ ] **GO for data preparation:** only after custodians independently verify the complete source guard and the output path is new.
- [ ] **GO for stage-one training:** only after data/manifest verification and explicit training approval.
- [ ] **GO for risk training:** only after stage-one checkpoint review and explicit training approval; use the OOF runner command above.
- [ ] **GO for calibration:** only after risk checkpoint review and freeze of model/hyperparameters; use the calibration runner above.
- [ ] **GO for locked-test evaluation:** only after separate explicit approval and frozen artifact/report digests.

**Overall: NO-GO today.** The source guard is absent/unverified, and training has not been approved. OOF and calibration runners are implemented but have not been exercised with trained artifacts. No data, guard, model training, optimizer update, or locked-label access was performed for this handoff.
