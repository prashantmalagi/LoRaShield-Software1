"""Prediction-to-calibration runner; never loads locked-test partitions."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from .calibration import (RiskCalibrationCase, calibrate_risk_and_select_threshold,
                          derive_risk_label, fingerprint_model)
from .inference import propose
from .provenance import partition_content_fingerprint, verify_manifest_files
from .train import load_partition


def run_calibration(model, calibration_rows, verified_manifest, output_json, *,
                    checkpoint_metadata: dict, proposer=propose):
    """Generate model logits/candidates, bind them to verified calibration
    contents and model weights, select fixed safety thresholds, then atomically
    write a new report. Invalid proposals are abstentions and recorded.
    """
    from .provenance import VerifiedPartitionManifest
    if not isinstance(verified_manifest, VerifiedPartitionManifest) or \
       not verified_manifest.is_file_verified:
        raise ValueError("calibration runner requires a manifest verified from partition files")
    if not checkpoint_metadata.get("risk_head_trained", False):
        raise ValueError("calibration requires a trained candidate-conditioned risk head")
    if not calibration_rows or any(row.partition != "calibration" for row in calibration_rows):
        raise ValueError("calibration runner accepts calibration rows only")
    actual_fingerprint = partition_content_fingerprint(calibration_rows)
    if actual_fingerprint != verified_manifest.content_fingerprints["calibration"]:
        raise ValueError("loaded calibration rows do not match verified manifest content")
    model_fingerprint = fingerprint_model(model)
    cases, rejected = [], []
    for index, row in enumerate(calibration_rows):
        prediction = proposer(model, row.received_morse)
        if prediction.proposed_morse is None or prediction.risk_logits is None:
            rejected.append({"source_id": row.source_id, "row": index,
                             "reason": prediction.reason or "missing_proposal_or_risk_logits"})
            continue
        try:
            label = derive_risk_label(row.reference_morse, row.received_morse,
                                      prediction.proposed_morse)
        except (TypeError, ValueError) as exc:
            rejected.append({"source_id": row.source_id, "row": index,
                             "reason": f"invalid_candidate:{exc}"})
            continue
        cases.append(RiskCalibrationCase(
            row.source_id, "calibration", tuple(prediction.risk_logits), label,
            row.is_clean, prediction.proposed_morse != row.received_morse))
    present_sources = {row.source_id for row in cases}
    if present_sources != set(verified_manifest.source_ids["calibration"]):
        raise ValueError("valid calibration proposals do not cover every verified calibration source")
    result = calibrate_risk_and_select_threshold(
        cases, manifest=verified_manifest, model=model)
    report = {
        "spec_version": "morse-correction-v2-calibration-1",
        "partition": "calibration",
        "calibration_content_sha256": verified_manifest.content_fingerprints["calibration"],
        "partition_manifest_sha256": verified_manifest.manifest_fingerprint,
        "model_weights_sha256": model_fingerprint,
        "cases": len(cases), "rejected_proposals": rejected,
        "temperature": result.calibrator.temperature,
        "improve_threshold": result.improve_threshold,
        "harm_threshold": result.harm_threshold,
        "selected_all_abstain": result.selected_all_abstain,
        "automatic_correction_ready": False,
        "criteria": result.criteria,
        "provenance_digest": result.provenance_digest,
        "locked_test_loaded": False,
    }
    target = Path(output_json)
    if target.exists():
        raise FileExistsError("calibration report path exists; refusing overwrite")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent,
                                         prefix=f".{target.name}.", suffix=".tmp", delete=False) as stream:
            temp_name = stream.name
            json.dump(report, stream, sort_keys=True, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        if target.exists():
            raise FileExistsError("calibration report path appeared during write")
        os.replace(temp_name, target)
    except BaseException:
        if temp_name and Path(temp_name).exists():
            Path(temp_name).unlink()
        raise
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = verify_manifest_files(args.manifest)
    rows = load_partition(args.calibration_data, "calibration")
    from .oof_runner import restore_base_model
    model, metadata = restore_base_model(args.model_checkpoint)
    report = run_calibration(model, rows, manifest, args.output_json,
                             checkpoint_metadata=metadata)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
