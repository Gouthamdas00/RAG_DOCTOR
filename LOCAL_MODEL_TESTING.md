# Run the local-model quality checks

The agentic quality suite uses a local Ollama model against synthetic workspaces. It does not read or send your finance app's files to the model.

Open PowerShell in the RAG Doctor project folder. Install Ollama and pull the model you want to test, for example:

```powershell
ollama pull qwen3:4b
```

Run the live-model checks, including the runtime ranking case that requires the model to agree with captured score evidence:

```powershell
.\scripts\Test-RagDoctorLocalModel.ps1 -ModelName qwen3:4b -TimeoutSeconds 300
```

The script checks the configured loopback server's `/api/tags` endpoint for the named model, so the Ollama CLI does not need to be on PowerShell's `PATH` when the local server is reachable. It falls back to `ollama list` if model discovery through HTTP is unavailable and the CLI is installed. It uses the project's `.venv` when present, configures the timeout for the test process, and restores prior environment values afterward. The tests run against the staged backend inside the current VSIX when available and print the analyzer path; the source backend is used only before the first staging build. They do not download models automatically. Use `-Endpoint` only with an HTTP loopback Ollama-compatible chat endpoint.

The offline suite verifies deterministic Python chunk-step, prompt-field, TypeScript prompt-template, and runtime ranking patterns. It also simulates an AI that invents a `topK` limit and verifies that the source-grounded prompt finding overrides it. Verified findings and their cited call sites are included as bounded source excerpts for the model. If the model disagrees with a deterministic or runtime finding, it gets one focused review turn using a compact evidence packet; its response still must pass source and evidence checks, and the verified fallback remains in place when it does not. The live suite contains five model-quality cases: an opaque prompt builder that drops retrieved context, a chunk loop with a zero step, an unsafe benchmark acceptance matcher, a model-generated tuning claim against a verified TypeScript prompt issue, and a ranking inversion grounded in captured scores. These cases require the model itself to confirm the finding; deterministic fallback does not count as a model-quality pass. If the configured model name is set but the local endpoint cannot be reached, the affected check is skipped with an explicit reason. An unreachable model therefore cannot produce a green model-quality result. The offline tests separately verify that source-backed fallback remains available. These checks do not establish accuracy on every RAG framework or workload; use labeled cases from the target app for that evaluation.
