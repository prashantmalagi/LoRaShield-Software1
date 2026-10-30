# Morse Correction V2: Research Design

**Status:** implementation approved. Isolated research package implemented; model training has not been run.  
**Scope:** isolated research model and offline evaluation. The receiver must continue to display received Morse by default. V3 and all current model artifacts remain unchanged.

## 1. Existing-model limitations

### Seq2Seq V1

V1 tokenizes dots, dashes, letter boundaries, and word boundaries as a variable-length sequence. It uses a small bidirectional GRU encoder and an autoregressive GRU decoder with attention. Inference is bounded greedy decoding and returns a proposal only after EOS; it exposes emitted tokens and termination status, but not a calibrated confidence estimate.

This design has several risks for correction:

- The decoder regenerates the whole message, so it can change a clean input or its separators even when no edit is needed.
- Autoregressive errors can accumulate, and greedy argmax does not account for uncertainty across complete output sequences.
- Token-level training loss does not directly express “preserve this input unless there is evidence that a specific correction helps.”
- Raw token probabilities, if exposed later, would not be calibrated confidence scores without separate calibration data and checks.
- The existing conservative gate evaluation on 10 calibration sources accepted no cases under its safety constraints. That sample is not evidence that a different gate will work.

### V3

V3 is a fixed-length, position-wise bidirectional GRU denoiser over dot, dash, and space characters. It predicts one character per input position and handles padding by retaining the original input where PAD/UNK is predicted. Prior evaluations found clean-message preservation, but that is empirical behavior on evaluated data, not a guarantee.

Its structural limits follow from its output contract:

- It cannot change sequence length, so it cannot directly repair inserted or deleted symbols.
- Since spaces are copied position by position, it is not designed to move, insert, or delete letter/word boundaries.
- Its training corruption model is primarily independent dot/dash substitutions at configured rates. It does not cover a broad channel model such as bursts, timing-driven segmentation errors, or symbol insertion/deletion.
- Per-position output scores do not by themselves establish calibrated correction confidence.

V2 should retain V3’s useful conservative behavior as a baseline while adding explicit support for symbol edits and boundary edits.

## 2. Proposed architecture

Use a small shared sequence encoder with two explicitly separated correction stages and a final risk/abstention head. Do not use an unconstrained free-form language decoder.

### Input representation

Represent a canonical received message as an ordered sequence of dot/dash symbols plus the observed letter and word boundaries between them. Retain source-position and boundary-position features. Reject malformed input before model inference. The input contract contains no reference label.

### Stage A: symbol correction

Run a monotonic edit transducer over the observed dot/dash stream. Its actions are `KEEP_DOT`, `KEEP_DASH`, `FLIP`, `DELETE`, `INSERT_DOT`, and `INSERT_DASH`; an action consumes or emits symbols according to a constrained transition table. Use an explicit maximum output length and maximum consecutive insertions. The transducer returns both the corrected symbol stream and an alignment from output symbols to input positions.

This stage may change the number and identity of symbols, but it must not make letter/word grouping decisions. It also exposes an identity path so an unedited message can pass through exactly.

### Stage B: boundary correction

Given the corrected symbol stream, its alignment, and the original boundary positions, predict a label for each gap between adjacent output symbols: `NONE`, `LETTER`, or `WORD`. Include start/end sentinels so decoding can enforce valid syntax. A constrained decoder disallows leading/trailing boundaries, adjacent separators, empty letters/words, or invalid token sequences. The boundary stage may move, insert, or remove separators without changing the corrected symbol stream.

Keep the two stages separately observable. This enables symbol-only and boundary-only ablations and makes it possible to tell whether a change came from dot/dash inference or segmentation.

### Shared encoder and size target

Initial candidate: a 4-layer Transformer encoder with width 128, four attention heads, feed-forward width 384, and a maximum sequence length of 256 events. Attach the edit-action, boundary, and risk heads to its contextual features. A small recurrent or finite-state edit decoder may be evaluated if the edit transducer cannot represent required alignments cleanly; that is an architecture experiment, not a reason to relax output constraints.

The encoder is approximately 0.66M parameters before heads, positional embeddings, and any edit-state decoder. Target total: about **0.7–1.2M parameters**, or roughly **3–5 MB** of FP32 weights. The implemented functional model currently builds with **796,433 parameters** (about 3.2 MB of FP32 weights); this is a model-size check, not a training result.

## 3. Preservation and abstention

The system returns a structured result containing the received Morse, candidate Morse (if valid), decision (`ACCEPT`/`ABSTAIN`), and reason codes. The caller-facing default remains the exact received Morse. The research gate returns the candidate only if every condition passes:

1. Original input and candidate are canonical and within configured length limits.
2. Both stages terminate normally and the combined candidate passes the constrained syntax decoder.
3. Edit counts and symbol/boundary edit budgets are within calibration-selected limits.
4. Calibrated risk estimates are above the acceptance threshold for the relevant edit classes, and the estimated probability of harm is below its threshold.
5. The clean-input preservation rule does not identify the example as clean/uncertain. Uncertain or out-of-distribution examples abstain.

If any check fails, return the original Morse byte-for-byte and record an abstention reason. An identity candidate is a pass-through, not a correction. Do not describe the learned risk score, raw logits, or a zero false-accept count as a guarantee of future clean-message preservation.

Train an auxiliary clean/noisy and correction-risk head, but do not let either head directly overwrite the input. At inference, define correction risk from whether the candidate improves, ties, or worsens the reference under the predeclared evaluation error measure. Fit any probability calibration map (for example, temperature scaling or isotonic regression) on calibration predictions only. Choose acceptance thresholds on that same calibration partition under preregistered constraints, such as zero accepted altered clean examples in calibration and a bounded harmful-accept rate. If no nonzero-coverage policy meets constraints, select `ABSTAIN_ALL` and report the model unsuitable for automatic correction.

## 4. Training examples and labels

Generate canonical source messages before corruption and assign every source and all its variants to exactly one partition. Store each pair with source ID, partition, corruption family, severity, random seed, received Morse, clean reference Morse, symbol alignment, and clean boundary positions.

Training corruption families should be separately tagged and sampled with bounded severities:

- Clean identity examples, with enough examples in every batch to train exact pass-through.
- Independent dot/dash substitutions and short burst substitutions.
- Symbol insertion and deletion, including mixtures with substitutions.
- Letter-boundary insertion, deletion, and one-symbol shifts.
- Word-boundary insertion, deletion, and one-letter shifts.
- Selected combined corruptions and channel-like mixtures, generated from documented parameters.

Synthetic corruption is a controlled approximation, not a substitute for a separately curated and labeled real-capture corpus. Never use V1/V3 locked-test examples to construct these examples.

### Supervision

- **Symbol target:** the clean dot/dash stream, plus monotonic edit alignment/actions derived by a deterministic minimum-edit procedure. Where several alignments tie, use a documented tie policy or marginalize over alignments rather than silently treating one as uniquely correct.
- **Boundary target:** the clean `NONE`/`LETTER`/`WORD` label at each gap of the clean symbol stream.
- **Identity target:** exact input-to-input target for clean examples; retain a clean flag and a clean-example exact-match metric.
- **Noise labels:** corruption family and configured severity for auxiliary diagnostics, not as a presumed available radio-time label.
- **Risk labels:** after inference on training/validation/calibration pairs, label a candidate as improved/tied/harmful against its reference using the predeclared combined metric. The reference is available only in offline training/evaluation, never at inference.

### Losses

Use a weighted multi-task objective, with component weights selected on validation only:

`L = λ_edit L_edit + λ_boundary L_boundary + λ_identity L_identity + λ_clean L_clean + λ_risk L_risk + λ_noise L_noise`

- `L_edit`: negative log likelihood for the monotonic edit transducer; marginalize ambiguous alignments when feasible.
- `L_boundary`: class-weighted cross entropy or a small CRF loss over gap labels, with constrained decoding at inference.
- `L_identity`: exact-target token loss on clean examples and on examples whose correct target equals the input. Track exact-match rate separately because token loss alone can hide a changed message.
- `L_clean`: binary cross entropy for clean/noisy detection, used only as one risk feature.
- `L_risk`: multi-class or ordinal loss for improvement/tie/harm labels. Train it from train examples; validate model selection on validation; calibrate and threshold only on calibration.
- `L_noise`: optional auxiliary corruption-family/severity classifier. It is diagnostic and is not assumed to know the real channel type at deployment.

Mask padded positions in every sequence loss. Do not allow a stronger corruption loss to swamp clean identity examples; report per-task and per-corruption losses, not just a single aggregate.

## 5. Data isolation and evaluation protocol

Partition by clean source before making any variants. Keep all variants of a source together. Prefer message and constituent-word disjointness across partitions, with counts, seeds, and SHA-256 fingerprints recorded in a versioned manifest.

1. **Train:** fit model parameters and learned risk features. No validation, calibration, or locked examples enter gradients.
2. **Validation:** choose architecture/checkpoint/epoch and loss weights. No threshold selection or calibration-map fit.
3. **Calibration:** after freezing model and hyperparameters, fit probability calibration and choose abstention thresholds under preregistered safety criteria. Do not use this partition for gradient updates or checkpoint selection.
4. **Fresh locked test:** create a new source-disjoint dataset for V2 with a new seed and manifest. A custodian or separate script stores labels apart from training and calibration paths. Freeze model, gate, and report template before a one-time evaluation. The evaluator must not expose locked labels to tuning code.

Exclude the previously evaluated locked-test corpus from all new V2 partitions. Do not open it to tune, select rules, inspect failure examples, or alter its artifacts. The new locked set is not to be generated or written as part of this design-only step; that belongs to an approved implementation phase.

Score unchanged input, V3, ungated V2, and gated V2 on exactly the same examples. Report denominators and abstentions, overall and by corruption family/severity. Primary measures:

- Clean false-accept rate: clean cases where the gate accepted a changed output / all clean cases.
- Clean exact-preservation rate and clean altered-message count.
- Corrupted-case correction rate: corrupted cases with lower predeclared combined errors after correction / corrupted cases with at least one input error; also report all-corrupted denominator.
- Harmful accepted corrections: accepted cases with higher combined error than unchanged input / accepted cases; include accepted denominator and severity.
- Dot/dash symbol edit distance and symbol error rate with numerator and denominator.
- Letter-boundary and word-boundary error counts/rates, each with numerator and denominator.
- Exact-message accuracy, proposal coverage, abstention rate, malformed-output rate, and risk-coverage curve.
- Confidence intervals clustered by source message, not by variant, because variants from one source are correlated.

Predeclare thresholds and criteria before opening the fresh locked set. If calibration selects abstain-all, or the locked test shows unacceptable harm/clean alterations, do not recommend automatic correction. A good result on a synthetic split is not a radio-channel guarantee.

## 6. Compute estimate for Google Colab

The proposed model is deliberately small. For batches of 32–64 and length at most 256, a **single 8–16 GB GPU** should be adequate; a T4-class GPU is a practical target, while a smaller available GPU can run with batch size 8–16 and gradient accumulation. Use mixed precision only after confirming numerical stability of the edit and boundary losses. A CPU can support unit/smoke tests and small overfit checks, but full multi-seed training and repeated evaluation will be considerably slower.

These are planning estimates, not Colab availability promises. Google states that GPU types and usage limits vary over time, resources are not guaranteed, and free-tier GPU access is restricted; paid tiers also remain subject to availability and usage limits. Confirm the actual runtime device and memory at the start of an approved run. [Google Colab FAQ: resource limits and GPU availability](https://research.google.com/colaboratory/faq.html)

## 7. Implementation plan and status

Implementation status after the pre-training audit: candidate-conditioned risk features, source-held-out OOF risk-proposal contracts, risk-only training guards, noisy boundary candidate augmentation, predicted-candidate boundary evaluation, provenance-bound calibration, fixed safety criteria, and subprocess dependency isolation are implemented in the isolated research package. These changes do not prepare data or execute optimization.

The risk head consumes original encoder context, candidate symbol features, source alignment, candidate boundary labels, and normalized edit features. Its labels use reference-relative combined symbol/boundary edit error: class 0 improve, 1 tie, 2 harm. The OOF runner groups all variants by source ID, trains a base model on each fit fold, and proposes using reference-free heldout inputs. It validates candidate syntax/alignment and materializes provenance-bound risk batches. The calibration runner verifies actual calibration files and model weights. Both remain untrained/unexecuted until explicitly authorized.

Boundary training can substitute deterministic noisy symbol candidates with explicit candidate-to-source alignment and projected boundary labels. Validation reports boundary accuracy and syntax validity after the model's own Stage-A edits. OOF runner interruption/failure paths are tested, but no fold model has been trained; boundary evaluation also requires trained weights. No data preparation or training was run during this audit.

Calibration requires a manifest verified against actual train/validation/calibration files and computes the fingerprint from actual model weights. The temperature fit uses calibration predictions only. Thresholds are predeclared at P(improve) >= 0.95 and P(harm) <= 0.01, with zero altered-clean accepts and zero observed harmful accepts; if none qualify, the result is all-abstain. No calibration runner has been executed with trained weights.

The source guard remains a separate readiness blocker: the package validates syntax/nonemptiness but cannot prove that a custodian's hashes are complete. No source guard was created, no data was prepared, and locked-test labels were not accessed. No fresh locked-test dataset was created or read.

Integration follow-up: OOF proposals now require canonical candidate Morse plus symbol alignment, retain rejected-candidate reasons, derive labels from the proposal/reference pair, and materialize risk batches with candidate symbols, alignment, and boundary labels. Risk updates require internally minted fold provenance and restore original trainability flags. Calibration provenance is minted only after partition content and source-ID checks. Checkpoint bundles preserve model and optimizer state in immutable epoch directories with atomic `CURRENT` advancement; resume verifies complete train/validation row fingerprints.

1. Before data preparation, obtain the custodian-generated hash guard for prior source corpora and the previously evaluated locked set. The repository does not contain this guard; preparation refuses an empty guard.
2. Run data-generation smoke checks, then prepare train, validation, and calibration files. Review the generated manifest and verify all source IDs/words remain disjoint.
3. Obtain a legitimate nonempty source guard through independent custodian verification before data preparation. OOF and calibration runners are implemented but require approved trained artifacts before execution.
4. After explicit training approval, execute the calibration runner against the risk-trained checkpoint and verified calibration manifest; the runner selects thresholds only from calibration predictions and reports abstain-all when criteria fail.
5. Add the offline comparison report for unchanged input, V3, ungated V2, and gated V2 on exactly the same cases.
6. Only after the user authorizes training, run a CPU data/model smoke test, then a Colab GPU resource/overfit smoke test, and then the stage-one training command in the package README. Weights must use a new versioned output path.
7. Generate the fresh locked bundle separately, after training and calibration are frozen. Open it once through the locked-test vault using a freeze manifest; never pass it into preparation, training, calibration, or threshold selection.

The package implementation approval is recorded. Training and any access to fresh locked-test labels remain separate approval gates.
