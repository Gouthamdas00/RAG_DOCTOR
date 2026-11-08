# RAG Doctor

RAG Doctor is a local-first diagnostic tool for Retrieval-Augmented Generation (RAG) applications. Its VS Code extension inspects workspace code and supported local runtime evidence, then points engineers to likely failure locations and suggests specific fixes. It does not automatically edit application code.

## What it does

- Scans supported workspace source, configuration, dataset structure, and logs for RAG-related evidence.
- Optionally observes an application started with **Run with Doctor** and captures supported model requests locally.
- Uses a local Ollama-compatible model for read-only investigation of a captured run and workspace code.
- Reports evidence, confidence, relevant file and line, a suggested replacement when it can verify one, and a way to check the result.
- Keeps captured run data in the local analyzer process memory; it is not sent to a hosted RAG Doctor service.

RAG Doctor cannot generically see another process's in-memory retriever results. If its observer does not support an app's client or framework, retrieval details may be unavailable. The report should identify that as an observability gap rather than claim retrieval failed. Findings are evidence-limited and suggestions require engineer review.

## Current evaluation evidence

The checked-in synthetic fault campaign contains 700 seeded fault variants and 100 healthy controls. Its deterministic checks detected the seeded patterns in that campaign, with no false positives on those controls; only 9 runs used the live Qwen model. These results test known rule patterns and are **not a real-world accuracy claim**. RAG Doctor still needs a human-labeled, held-out benchmark across independent RAG applications before reporting production accuracy.

## Get started

1. Install Python and VS Code.
2. Install the project dependencies from the repository root:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Open `vscode-extension` in VS Code and press **F5** to launch the Extension Development Host.
4. Open your RAG application workspace in that host and select the RAG Doctor Activity Bar icon.
5. Choose **Scan workspace** for static inspection, or **Run with Doctor** to start the app with the local observer.
6. To run **Investigate with local AI**, start Ollama, install a local model, and set `ragDoctor.localModelName` in VS Code settings (for example, `qwen3:4b`).

See the [VS Code extension guide](vscode-extension/README.md) for supported observation paths, privacy behavior, app connection details, and evaluation workflow.

## Build the VSIX

From `vscode-extension`:

```powershell
npm run vscode:prepublish
npx vsce package
```

This stages the local analyzer into the extension package and creates a `.vsix` file. Generated runtime files and packaged VSIX files are excluded from Git.

## Development checks

```powershell
python -m unittest discover -s backend/tests
```

From `vscode-extension`:

```powershell
npm test
```

The local-model quality checks are optional and require a running compatible local model. See [LOCAL_MODEL_TESTING.md](LOCAL_MODEL_TESTING.md).

## Repository layout

- `backend/` — local diagnostic API, RAG analysis, runtime evidence handling, and tests.
- `frontend/` — local analyzer interface.
- `vscode-extension/` — VS Code extension, observers, connector helpers, and extension tests.
- `rag_doctor_probe.py` and `rag_doctor_probe.js` — standalone local connector helpers.

## Privacy and safety

The analyzer binds to loopback and restricts workspace reads to selected workspace roots. Runtime capture can include sensitive prompts, answers, or retrieved text when content capture is enabled; review the extension settings and the [privacy details](vscode-extension/README.md) before using it with sensitive data. The local AI investigator reads bounded workspace evidence and cannot execute workspace code or apply suggested edits.
