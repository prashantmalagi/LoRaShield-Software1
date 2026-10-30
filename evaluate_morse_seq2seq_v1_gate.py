"""Calibration-only offline research evaluation for the V1 correction gate.

This command selects thresholds on the V1 calibration partition only. It never
loads or scores locked-test labels. Output is a research report, not a receiver
configuration. By default it uses the first 10 sources from the existing
calibration partition to keep CPU autoregressive inference practical; increase
``--calibration-source-count`` to cover more of that same partition.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import morse_seq2seq_v1_data as data
import morse_seq2seq_v1_model as v1
from correction.experimental_gate import GatePolicy, InferenceEvidence, decide
from evaluate_morse_seq2seq_v1 import (
    DEFAULT_MANIFEST_PATH, SEQ2SEQ_MODEL_PATH, SEQ2SEQ_METADATA_PATH,
    _load_v3_model, case_metrics, greedy_decode_batch,
)
from train_morse_seq2seq_v1 import make_partition_pairs
from evaluate_morse_correction_v3_corruption import predict_messages as predict_v3


def calibration_sources(manifest_path: Path) -> tuple[list[str], int]:
    """Reproduce calibration membership without generating V1 locked sources."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seed = int(manifest["planned_settings"]["root_seed"])
    exclusions = data.existing_source_corpora()
    used_messages = {" ".join(data._validate_source_text(m))
                     for messages in exclusions.values() for m in messages}
    used_words = {word for message in used_messages for word in message.split()}
    result = None
    for partition in ("train", "validation", "calibration"):
        rng = __import__("random").Random(seed + data.PARTITION_SEED_OFFSETS[partition])
        sources = []
        while len(sources) < data.PLANNED_PARTITION_COUNTS[partition]:
            words = ["".join(rng.choice(tuple(data.MORSE)) for _ in range(rng.randint(3, 6)))
                     for _ in range(rng.randint(2, 3))]
            if not any(len(data.MORSE[ch]) >= 3 for word in words for ch in word):
                continue
            message = " ".join(words)
            if message in used_messages or set(words) & used_words:
                continue
            data.encode_morse(data.encode_text(message))
            sources.append(message)
            used_messages.add(message)
            used_words.update(words)
        if partition == "calibration":
            result = sources
    expected = manifest["partition_audit"]["source_fingerprints_sha256"]["calibration"]
    if data.source_fingerprint(result or ()) != expected:
        raise ValueError("reproduced calibration partition fingerprint mismatch")
    return result or [], seed


def select_policy(rows):
    """Choose greatest coverage with zero clean changes and <=5% accepted harm."""
    choices = []
    for symbols in range(0, 5):
        for boundaries in range(0, 4):
            policy = GatePolicy(symbols, boundaries)
            accepted = harmful = clean_altered = 0
            for case, proposal, evidence in rows:
                decision = decide(case.received_morse, proposal, evidence, policy)
                if decision.accepted:
                    accepted += 1
                    before = case_metrics(case.reference_morse, case.received_morse)["combined_errors"]
                    after = case_metrics(case.reference_morse, decision.output_morse)["combined_errors"]
                    harmful += after > before
                    clean_altered += case.corruption_type == "clean" and decision.output_morse != case.reference_morse
            if accepted and clean_altered == 0 and harmful / accepted <= 0.05:
                choices.append((accepted, -harmful, -symbols, -boundaries, policy))
    if not choices:
        return GatePolicy(0, 0), {"selected": "abstain_all", "accepted": 0,
                                  "criteria": "zero accepted clean alterations and <=5% harmful accepted corrections"}
    selected = max(choices)[-1]
    return selected, {"selected": "calibration_policy", "max_symbol_edits": selected.max_symbol_edits,
                      "max_boundary_edits": selected.max_boundary_edits,
                      "criteria": "zero accepted clean alterations and <=5% harmful accepted corrections"}


def report(cases, v3_outputs, proposals, evidence, policy):
    systems = {name: [] for name in ("unchanged", "v3", "gated_v1")}
    accepted = abstentions = harmful = clean_false_accept = correction_success = 0
    for case, v3_out, proposal, ev in zip(cases, v3_outputs, proposals, evidence):
        decision = decide(case.received_morse, proposal, ev, policy)
        accepted += decision.accepted
        abstentions += not decision.accepted
        before = case_metrics(case.reference_morse, case.received_morse)["combined_errors"]
        after = case_metrics(case.reference_morse, decision.output_morse)["combined_errors"]
        harmful += decision.accepted and after > before
        clean_false_accept += case.corruption_type == "clean" and decision.accepted and decision.output_morse != case.reference_morse
        correction_success += case.corruption_type != "clean" and before > 0 and after < before
        systems["unchanged"].append(case.received_morse)
        systems["v3"].append(v3_out)
        systems["gated_v1"].append(decision.output_morse)
    result = {"denominators": {"all_cases": len(cases), "clean_cases": sum(c.corruption_type == "clean" for c in cases),
                                  "corrupted_cases": sum(c.corruption_type != "clean" for c in cases),
                                  "accepted": accepted, "abstentions": abstentions,
                                  "harmful_accepted_denominator": accepted,
                                  "clean_false_accept_denominator": sum(c.corruption_type == "clean" for c in cases),
                                  "corrupted_correction_rate_denominator": sum(c.corruption_type != "clean" and case_metrics(c.reference_morse, c.received_morse)["combined_errors"] > 0 for c in cases)},
              "clean_false_accepts": clean_false_accept, "harmful_accepted_corrections": harmful,
              "corrupted_case_corrections": correction_success,
              "systems": {}}
    for name, outputs in systems.items():
        rows = [case_metrics(case.reference_morse, out) for case, out in zip(cases, outputs)]
        sym_n = sum(r["symbol_errors"] for r in rows); sym_d = sum(r["symbol_denominator"] for r in rows)
        let_n = sum(r["letter_boundary_errors"] for r in rows); let_d = sum(r["letter_boundary_denominator"] for r in rows)
        word_n = sum(r["word_boundary_errors"] for r in rows); word_d = sum(r["word_boundary_denominator"] for r in rows)
        result["systems"][name] = {"messages": len(rows), "symbol_errors": sym_n, "symbol_denominator": sym_d,
            "symbol_error_rate": sym_n / max(sym_d, 1), "letter_boundary_errors": let_n,
            "letter_boundary_denominator": let_d, "word_boundary_errors": word_n,
            "word_boundary_denominator": word_d}
    result["denominators"]["clean_cases"] = sum(c.corruption_type == "clean" for c in cases)
    result["denominators"]["corrupted_cases"] = sum(c.corruption_type != "clean" for c in cases)
    result["rates"] = {
        "clean_false_accept_rate": clean_false_accept / max(result["denominators"]["clean_false_accept_denominator"], 1),
        "harmful_accepted_correction_rate": harmful / max(accepted, 1),
        "corrupted_case_correction_rate": correction_success / max(result["denominators"]["corrupted_correction_rate_denominator"], 1),
    }
    result["boundary_rates"] = {
        name: {"letter_boundary_error_rate": row["letter_boundary_errors"] / max(row["letter_boundary_denominator"], 1),
              "word_boundary_error_rate": row["word_boundary_errors"] / max(row["word_boundary_denominator"], 1)}
        for name, row in result["systems"].items()
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--candidate-model", type=Path, default=SEQ2SEQ_MODEL_PATH)
    parser.add_argument("--candidate-metadata", type=Path, default=Path(__file__).resolve().parent / "models" / "morse_seq2seq_v1_training_metadata.json")
    parser.add_argument("--calibration-source-count", type=int, default=10)
    args = parser.parse_args()
    if not args.manifest.is_file() or not args.candidate_model.is_file() or not args.candidate_metadata.is_file():
        raise SystemExit("completed source manifest and trained V1 candidate artifacts are required")
    metadata = json.loads(args.candidate_metadata.read_text(encoding="utf-8"))
    if metadata.get("model_file") != args.candidate_model.name:
        raise SystemExit("V1 metadata/model filename mismatch")
    calibration, seed = calibration_sources(args.manifest)
    # No V1 locked-test partition is generated, loaded, or scored here.
    if args.calibration_source_count < 1 or args.calibration_source_count > len(calibration):
        raise SystemExit(f"--calibration-source-count must be in [1, {len(calibration)}]")
    calibration = calibration[:args.calibration_source_count]
    cases = make_partition_pairs("calibration", calibration, seed=seed)
    candidate = v1.load_model(args.candidate_model)
    proposals_out = greedy_decode_batch(candidate, [c.received_morse for c in cases], max_output_tokens=128, batch_size=128)
    proposals = [row.output_morse for row in proposals_out]
    evidence = [InferenceEvidence(row.terminated_by_eos, row.termination_reason, row.output_tokens)
                for row in proposals_out]
    calibration_rows = list(zip(cases, proposals, evidence))
    policy, selection = select_policy(calibration_rows)
    v3 = _load_v3_model()
    v3_outputs = predict_v3(v3, [c.received_morse for c in cases])
    results = report(cases, v3_outputs, proposals, evidence, policy)
    print(json.dumps({"evaluation_partition": "calibration_only", "calibration_sources_used": len(calibration), "threshold_selection": selection,
        "policy_suitable_for_automatic_correction": selection["selected"] != "abstain_all",
        "available_evidence": ["emitted token sequence", "EOS termination", "termination reason", "proposal syntax", "symbol/boundary edit distances"],
        "probability_evidence": "Not exposed by V1 DecodeResult; decoder token probabilities are uncalibrated and cannot be gated without additional calibration.",
        "results": results,
        "limitations": ["Calibration criteria are sample constraints, not a guarantee of clean-message preservation.",
                        "Synthetic inputs and modeled corruptions only; no real radio-channel claim.",
                        "Thresholds are exploratory and must be independently validated before any operational use."]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
