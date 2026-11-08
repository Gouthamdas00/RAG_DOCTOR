# RAG Doctor for VS Code (local development build)

RAG Doctor adds an Activity Bar view and commands to scan a workspace or connect a local RAG application. The Python analyzer starts only after the user chooses **Scan workspace** or confirms **Connect app**. It binds to `127.0.0.1` and receives only the open workspace folders as allowed roots. Workspace scans read bounded code, dataset, and log files locally. The analyzer keeps only the most recent captured run per workspace in process memory. Stopping the analyzer clears those captures; it does not write them to a database.

## Start the extension

1. Install the dependencies from the project root with `python -m pip install -r requirements.txt` (or use the project's `.venv`).
2. Open this `vscode-extension` folder in VS Code and launch an Extension Development Host with **F5**.
3. Open the finance/RAG application folder in the Development Host.
4. Click the RAG Doctor Activity Bar icon. It scans the workspace automatically; choose **Run with Doctor** for an opt-in runtime evaluation.

Opening the trusted RAG Doctor view now scans automatically. Findings are collapsed into clickable sections so only the category the engineer opens is shown. **Review RAG code with local AI** is an explicit, read-only workspace audit that can run even when there is no capture or labeled dataset. It asks the configured local code model to trace bounded source/config files and report only code-supported risks; without a run it cannot claim runtime answer quality or declare the system healthy. **Run with Doctor** is an opt-in workflow that asks for the app start command and launches it with temporary local Python and Node.js observers. The Python observer covers `requests`, HTTPX (sync and async), standard-library `urllib`, and `aiohttp` calls to Google, OpenAI, Anthropic, and Ollama inference endpoints. The Node.js observer preloads into Node processes through `NODE_OPTIONS` and observes supported provider calls through Node's built-in `fetch` and core `http`/`https` request transports; it uses core HTTP for its local analyzer beacon when the app's Node version has no global `fetch`. Both observers are activated only for the launched process tree, send captured run data only to the local analyzer at `127.0.0.1`, and do not edit application source. When local labeled JSON/JSONL cases are available, RAG Doctor discovers a FastAPI OpenAPI query endpoint and offers to replay those cases. Without labeled cases, it still starts the app for manual capture: submit a prompt in the app, then click **Scan workspace** to inspect the latest run and choose **Investigate latest run with local AI**. Manual capture does not replay or score answers automatically. Python `requests` and HTTPX streaming plus Node `fetch`/`http`/`https` response bodies are observed without reading from the application's response stream; Python `urllib` and `aiohttp` are observed when the application fully reads the response body. The observer extracts explicitly labeled source blocks and common `Context:`/document headings as heuristic prompt-context evidence; this does not prove which retriever results were used or their ranks. Custom retrieval formats may show an observability gap until an adapter is added.

The replay can invoke the app's model provider for each case, so the extension asks before sending test queries. Before launch it checks whether the target API port is already occupied; stop an existing app process and retry so RAG Doctor can instrument the process it starts. It waits for a content-free local beacon confirming that a bundled observer loaded, then clears the previous capture before each case. Evaluation distinguishes an explicitly missing labeled source from an answer that is merely not quoted verbatim; calculated or paraphrased cases are marked for review unless runtime evidence establishes a stage failure. Diagnosis reports link to relevant scanned code lines and show detected numeric settings when available. Captured prompt, response, and passages are held in the local analyzer's RAM for the current process. The app itself still follows its configured provider/network behavior. After another **Run with Doctor** in the same VS Code extension session, RAG Doctor compares paired case outcomes and a bounded source/config fingerprint with the prior run. The comparison baseline stores only metrics and a hash in extension memory, not prompts, answers, or passages; the comparison is evidence of change, not proof that an edit caused it.

The evaluation report also has **Replay labeled cases against current app**. After applying a recommendation and confirming that the app reloaded it, this replays the same request settings for the captured labeled cases, then compares paired reference-support outcomes, labeled-source presence, latency, and source fingerprints with the initial run. Each replay uses a fresh session ID when the API supports one. An expandable per-case view shows before/after answer text locally (up to 4,000 characters per answer) and up to ten retrieved document IDs, chunk IDs, ranks, and scores, so engineers can review answer changes and retrieval shifts that a score alone misses. Passage text is not duplicated in this comparison. The extension asks before the replay because it calls the configured app/model and may use API quota; answers are not sent to RAG Doctor's remote service or saved by the extension. For an AI-proposed code fix, **Track this exact fix** stores only its workspace-relative path, line, case ID, and replacement hash in extension memory. Track it before applying the fix, or afterward if the exact suggested replacement was applied unchanged; the replay checks whether that code block is present in the file. This source check and the outcome comparison do not prove the change caused an improvement. Tracking data is cleared when the extension unloads. Without labeled cases, RAG Doctor cannot measure answer improvement.

The **Latest RAG run** section also has **Investigate latest run with local AI**, so an unlabeled app run can be investigated without creating a benchmark record. It uses the same bounded, read-only agent and local endpoint. It starts from the run's code/log locations and can inventory, search, and read relevant workspace files even when names are unfamiliar. If prompt/answer/passage capture is disabled, the agent only receives the available metadata and code/log evidence and must report that content-level conclusions are unavailable. The agent cannot execute commands or edit files. AI actions send selected bounded workspace evidence to the configured loopback model only after the engineer clicks the action; no remote inference endpoint is accepted.

Each captured evaluation case has **Investigate with local AI** too. Both actions invoke the configured local Ollama chat endpoint (default `http://127.0.0.1:11434/api/chat`). Install and start Ollama, install a model in that local runtime, then set `ragDoctor.localModelName` to its installed name (for example, `qwen3:4b`). RAG Doctor sends the selected run evidence and bounded workspace excerpts only to the loopback endpoint. The endpoint is restricted to loopback HTTP, environment-configured proxies are bypassed, and redirects are disabled. With passage text excluded, the investigator still receives chunk IDs, ranks, and finite dense/BM25/rerank scores; included question, prompt, answer, and passage content is scrubbed again before local-model use. If reasoning mode returns no final answer content, RAG Doctor retries once with reasoning disabled and records that fallback. The agent returns evidence citations, a proposed code change, and a verification plan; an exact replacement block is shown only when the quoted current code matches the saved source. Python replacements are syntax checked with Python's parser. CommonJS JavaScript replacements are parsed in memory with Node without executing the workspace code or writing a temporary file; stale or invalid replacements are suppressed. ES modules, JSX, and TypeScript are explicitly left unchecked. Unverified conclusions remain hypotheses. Explicit arithmetic in the answer is checked separately with decimal arithmetic.

When the agent proposes a supported `top_k` experiment, **Run verified experiment** asks for confirmation, then replays every eligible labeled case once with the captured baseline value and once with the candidate value. Each variant gets a fresh chat session, and baseline-first/candidate-first order alternates across cases to reduce history and cache-warmup bias. It compares strict reference-supported outcomes, labeled-document presence, and app-reported latency, and counts only cases with both diagnoses in the paired totals. The replay consumes the app model's normal quota and never changes the app default. This validates only that parameter on those cases; other code changes still require a fresh run after the engineer applies them.

RAG Doctor does not ship or train a foundation model. The current agent uses an installed code-capable model plus RAG-specific investigation instructions, bounded workspace tools, and deterministic checks. This lets it reason about code patterns and failures absent from a labeled dataset; it does not guarantee discovery of every possible failure. The separate Random Forest telemetry classifier is not the agent and its current same-sample metrics do not establish generalization. Fine-tuning should wait until there is a varied, human-reviewed dataset of verified causes and fixes, and must be evaluated on held-out applications. Verification remains necessary because a plausible AI suggestion is not proof that a change improves a RAG system.

From the RAG Doctor project root, test an installed model against synthetic chunking, prompt-context, and captured ranking defects that are not part of the FinanceBench labels. The ranking case also checks that the model returns a grounded fix without relying on the deterministic fallback. Start Ollama and set the model name in PowerShell before running:

```powershell
$env:RAG_DOCTOR_TEST_MODEL_NAME = "qwen3:4b"
$env:RAG_DOCTOR_LOCAL_MODEL_TIMEOUT = "300"
.venv\Scripts\python.exe -m unittest backend.tests.test_agentic_model_live -v
```

These opt-in checks evaluate the configured model on temporary synthetic workspaces and do not read the finance application. The checks require the model itself to confirm the grounded finding; a deterministic fallback does not count as model-quality evidence. They are skipped when the model name is unset or the local endpoint is unreachable, and fail if the model responds but does not support the diagnosis. The model call timeout defaults to 300 seconds and can be set with `RAG_DOCTOR_LOCAL_MODEL_TIMEOUT` (30–900 seconds). Reasoning mode is enabled by default and can be disabled in RAG Doctor settings; if a response has no final content, the investigator retries once with reasoning disabled. Set `RAG_DOCTOR_TEST_MODEL_ENDPOINT` only to a local Ollama-compatible `/api/chat` endpoint if it is not using the default port.

The deterministic checks can be exercised without Ollama with `python -m unittest backend.tests.test_agentic_static_verifier backend.tests.test_agentic_investigator backend.tests.test_runtime_observer -v`. Extension observer coverage can be exercised with `npm test` from this folder. These offline checks cover `UNKNOWN`/timeout fallback, cross-file retrieval-to-prompt traces, weak substring/any-number evaluation matchers, evidence/source validation, Python HTTP-client adapters, and Node.js `fetch`/`http`/`https` interception.

The investigator is hybrid: a local language model traces unfamiliar code, while deterministic checks can take over when they prove a narrow defect from source and run evidence. Agent tools can inventory, search, and read bounded Python, JavaScript/TypeScript, Java, Go, Rust, C#, PHP, Ruby, Kotlin, Scala, and common configuration files; deterministic setting and dataflow checks remain strongest for Python and simple JavaScript/TypeScript. Current Python checks cover invalid chunk-step loops, prompt builders that drop context, retrieval-to-generation flows across same-file code or straightforward imported functions, benchmark matchers that accept a gold substring or any answer number without binding it to the requested metric and unit, and captured high-is-better scores ranked opposite to a source similarity sort. Runtime ranking findings require an explicit higher-is-better statement and at least two ranked finite scores; the proposed sort change is syntax checked and cited against both the run summary and source line. Source excerpts prioritize relevant function bodies rather than only the file header. Empty source files are handled safely during inventory and excerpt loading. If the local model times out or returns invalid output, a matching deterministic finding is still returned with the model failure identified. A verified finding can also replace an `UNKNOWN` or ungrounded model result; it is labeled as a probable issue and includes source evidence. These checks do not cover every RAG failure, dynamic imports, arbitrary cross-file data flow, or every language. Fine-tuning remains deferred until there is a varied, human-reviewed dataset and an evaluation split by application to measure generalization.

The VSIX stages the analyzer source, local dashboard, and `requirements.txt` under its `runtime/` directory, so it no longer needs a separate RAG Doctor checkout to find the backend. Python and the listed packages must still be installed locally; RAG Doctor does not download dependencies. During development it uses the project `.venv` when available. For an installed extension, set `ragDoctor.pythonPath` to a Python environment with the packages from `runtime/requirements.txt` installed. `ragDoctor.backendPath` can override the bundled analyzer with a local RAG Doctor project root. The local analyzer defaults to port `8765` and binds only to loopback.

## What a workspace scan reads

- Python files: AST inspection of simple numeric RAG settings.
- JavaScript/TypeScript files: literal setting matches for `chunk_size`, `chunk_overlap`, `top_k`, and related settings.
- `.txt`, `.md`, `.csv`, and `.json` files: bounded document-shape summaries; document contents are not returned to the view.
- `.log`, `.jsonl`, and `.ndjson` files, plus JSON files whose names contain `rag`, `trace`, or `run`: bounded search for common event fields and OpenTelemetry JSON span batches. GenAI retrieval/generation spans are grouped by trace ID; provider, model, top-k, token counts, latency, candidate ranks, and IDs are normalized without relying on application variable names.
- Network-call patterns in code: shown as **potential** transfer locations. Static scanning cannot prove a transfer occurred at runtime.

The scan skips hidden folders, virtual environments, dependency trees, build output, and files larger than configured limits. By default it excludes question, prompt, response, and passage text from the normalized run and diagnosis; only bounded retrieval/model metadata is retained. The scanner still reads local log bytes to parse their structure. Enable `ragDoctor.includeRunContent` only when content-level diagnosis is approved. Any included content is scrubbed for common secrets and PII patterns before retention or display.

## Connect an application

Run **RAG Doctor: Connect App** or click **Connect app** in the view. RAG Doctor reads only common dependency manifests to identify Python/Node.js and known framework dependencies. After confirmation, it installs the bundled standard-library-only `rag_doctor_probe.py` or `rag_doctor_probe.js` helper at the selected workspace root and starts the local analyzer. It does not edit or execute application source code.

The connection guide opens with an example to adapt to the application's variable names. Add the capture call where both retrieved documents and the final answer are available. This application call is required because VS Code cannot generically inspect another process's in-memory RAG transaction. By default, the connector sends ranks, IDs, scores, and selected runtime metadata to loopback; it omits question, prompt, response, and passage text. Set `include_retrieved_text=True` (Python) or `includeRetrievedText: true` (JavaScript) to send retrieved passages to the local analyzer while still excluding question, prompt, and response. `include_content=True` is a separate opt-in that includes all of those fields. Captures remain in analyzer RAM until the local analyzer exits. Workspace Trust must be granted before scanning or connecting.

## Capture a live run from the application

Use **RAG Doctor: Connect App** to install the bundled standard-library helper into the selected application workspace and open a language-specific example. Adapt the variable names and add one capture call after retrieval and generation. The app must make this call because RAG Doctor cannot generically inspect another process's in-memory state. The connector sends the run over loopback to the analyzer; it does not send it to a hosted RAG Doctor service.

```python
from pathlib import Path
from rag_doctor_probe import RAGDoctorProbe

probe = RAGDoctorProbe("http://127.0.0.1:8765")
probe.capture(
    question=user_question,
    prompt=assembled_prompt,
    retrieved_chunks=[
        {
            "text": item.page_content,
            "chunk_id": item.metadata.get("id"),
            "doc_id": item.metadata.get("source"),
            "final_rank": rank,
            "dense_score": item.metadata.get("score"),
        }
        for rank, item in enumerate(retrieved_documents, start=1)
    ],
    llm_answer=model_response,
    workspace_path=str(Path(__file__).resolve().parent),
    include_content=True,  # Explicitly include question, prompt, passages, and answer locally.
    expected_answer=expected_answer,  # Supply only for a labeled/evaluation run.
    top_k_final=top_k,
)
```

For a Node.js application, use the installed `rag_doctor_probe.js` helper and call it after generation. Set `includeContent: true` only for an explicitly approved content-level run:

```js
const { captureRun } = require("./rag_doctor_probe");

await captureRun({
  question,
  prompt,
  retrievedChunks: documents.map((doc, index) => ({
    text: doc.pageContent,
    final_rank: index + 1,
    score: doc.score,
  })),
  answer: modelResponse,
  workspacePath: process.cwd(),
  includeContent: true, // Explicitly include question, prompt, passages, and answer locally.
  expectedAnswer, // Include only for a labeled/evaluation run.
  topKFinal: topK,
});
```

Pass `oracle_passed` when your app can rerun the same question with only the relevant source passage. That evidence lets RAG Doctor distinguish context noise from generator failure. For richer retrieval/reranking diagnosis, include chunk ranks and scores. `data_transfer_events` can carry sanitized local telemetry your app already records; RAG Doctor does not intercept process network traffic.

## Diagnosis limits

Without an expected answer, source passage, or oracle result, the view reports evidence limits instead of claiming factual correctness. Batch replay currently requires a discoverable FastAPI JSON query route. Runtime capture supports Python `requests`, HTTPX, `urllib`, and `aiohttp`, plus Node.js built-in `fetch` and core `http`/`https`, for supported Google, OpenAI, Anthropic, and Ollama model endpoints. Python `requests`/HTTPX and Node.js response streams are observed without consuming the application's response; for Python `urllib` and `aiohttp`, capture occurs when the full response body is read. Other custom SDK transports, providers, application frameworks, and retrieval prompt formats may need an adapter. Log parsing supports common JSON/JSONL shapes. Current static setting analysis is strongest for Python and simple JavaScript/TypeScript literals. The extension does not modify source files automatically.
