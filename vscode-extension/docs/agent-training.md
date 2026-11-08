# RAG Doctor investigator: training and evaluation plan

## Current implementation

The investigator uses an already-trained, code-capable model supplied by a local Ollama server. RAG Doctor adds a RAG-specific diagnosis policy, read-only workspace tools, evidence citation checks, and deterministic verifiers. No foundation model weights are bundled, and the investigator does not fine-tune itself from a user's workspace.

The existing `MLTraceDiagnoser` is a small supervised telemetry classifier. Its current `fit_and_evaluate` reports metrics on the same traces it just fit, so those numbers must not be presented as held-out accuracy or evidence of generalization.

## Why not train on the current benchmark now

The available FinanceBench cases are answer examples, not verified root-cause/fix records. A test case that misses an answer does not by itself tell us whether retrieval, chunking, generation, source coverage, or the test label is wrong. Training on guessed causes would teach the model to repeat those guesses. A small set of examples also cannot enumerate every failure in unfamiliar RAG code.

## What to collect before fine-tuning

Each training record should be reviewed and contain:

- a scrubbed code and runtime evidence bundle, with framework and pipeline stage;
- a confirmed diagnosis or an explicit `unknown` label;
- the exact evidence supporting the diagnosis, including valid file/line or trace references;
- a proposed change and a control/variant run;
- before/after answer correctness, retrieval support, latency, and token cost;
- reviewer approval that the fix caused the measured improvement.

Collection must be explicit and local. Do not use customer prompts, answers, or source documents as training examples by default. If examples are later exported for training, require a separate user action and secret/PII review.

## How to evaluate generalization

Keep whole applications, repositories, and framework versions out of training and use them only for held-out evaluation. Do not randomly split nearly identical traces from one app across train and test. Measure:

1. root-cause precision and recall, including the rate of false diagnoses;
2. percentage of evidence citations that point to real code or captured data;
3. percentage of proposed fixes accepted by a reviewer;
4. percentage of fixes that improve the held-out evaluation suite without unacceptable latency/token regressions;
5. calibration: whether `confirmed`, `probable`, and `insufficient_evidence` match observed correctness.

Fine-tuning should be an optional later optimization to the local model. The tool-using investigator and deterministic verification remain necessary for issues not represented in the training set.
