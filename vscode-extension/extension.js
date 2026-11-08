const vscode = require('vscode');

const http = require('http');
const path = require('path');
const fs = require('fs');
const crypto = require('crypto');
const vm = require('vm');
const { spawn } = require('child_process');
const { compareSnapshots, snapshotEvaluation } = require('./evaluation-comparison');
const { fingerprintWorkspace } = require('./workspace-fingerprint');

class RagDoctorViewProvider {
  constructor(context) {
    this.context = context;
    this.output = vscode.window.createOutputChannel('RAG Doctor');
    this.backendProcess = null;
    this.ownsBackend = false;
    this.backendRoots = [];
    this.previousSuiteSnapshots = new Map();
    this.latestDiagnoses = new Map();
    this.pendingFixVerifications = new Map();
  }

  resolveWebviewView(view) {
    view.webview.options = { enableScripts: true };
    this.attach(view.webview);
    if (vscode.workspace.isTrusted && this.getWorkspaceFolders().length) void this.scan(view.webview);
  }

  attach(webview) {
    webview.html = this.getHtml(webview);
    const disposable = webview.onDidReceiveMessage(async (message) => {
      if (message && message.type === 'scan') await this.scan(webview);
      if (message && message.type === 'connect') await this.connectApp();
      if (message && message.type === 'runWithDoctor') await this.runWithDoctor();
      if (message && message.type === 'openLocation') await this.openLocation(message);
      if (message && message.type === 'investigateLatestRun') await this.investigateLatestRun(webview, message.workspacePath, message.index);
      if (message && message.type === 'investigateWorkspace') await this.investigateLatestRun(webview, message.workspacePath, message.index, true);
    });
    this.context.subscriptions.push(disposable);
  }

  getWorkspaceFolders() {
    return (vscode.workspace.workspaceFolders || []).filter((folder) => folder.uri.scheme === 'file').map((folder) => ({
      name: folder.name,
      path: folder.uri.fsPath,
      uri: folder.uri,
    }));
  }

  getSettings() {
    const cfg = vscode.workspace.getConfiguration('ragDoctor');
    const configuredRoot = cfg.get('backendPath', '').trim();
    const extensionRoot = this.context.extensionPath;
    const developmentRoot = path.resolve(extensionRoot, '..');
    const bundledRoot = path.join(extensionRoot, 'runtime');
    const bundledServer = path.join(bundledRoot, 'backend', 'app', 'api', 'server.py');
    const root = configuredRoot || (fs.existsSync(bundledServer) ? bundledRoot : developmentRoot);
    const pythonRoots = [path.resolve(root), developmentRoot];
    const venvPython = pythonRoots.map((candidate) => process.platform === 'win32'
      ? path.join(candidate, '.venv', 'Scripts', 'python.exe')
      : path.join(candidate, '.venv', 'bin', 'python')).find((candidate) => fs.existsSync(candidate));
    return {
      root: path.resolve(root),
      python: cfg.get('pythonPath', '').trim() || venvPython || (process.platform === 'win32' ? 'python' : 'python3'),
      port: Math.max(1024, Math.min(65535, Number(cfg.get('backendPort', 8765)) || 8765)),
      localModelEndpoint: cfg.get('localModelEndpoint', 'http://127.0.0.1:11434/api/chat').trim(),
      localModelName: cfg.get('localModelName', '').trim(),
      localModelThinking: cfg.get('localModelThinking', true) !== false,
    };
  }

  async ensureBackend(roots) {
    const settings = this.getSettings();
    const normalizedRoots = roots.map((root) => path.resolve(root)).sort();
    if (this.ownsBackend && this.backendProcess && this.backendProcess.exitCode === null && JSON.stringify(normalizedRoots) !== JSON.stringify(this.backendRoots)) {
      const previous = this.backendProcess;
      previous.kill();
      await Promise.race([
        new Promise((resolve) => previous.once('exit', resolve)),
        new Promise((resolve) => setTimeout(resolve, 2500)),
      ]);
      this.backendProcess = null;
      this.ownsBackend = false;
    }
    if (await this.request(settings.port, 'GET', '/api/health').then((x) => x.status === 200).catch(() => false)) {
      return settings;
    }

    const serverPath = path.join(settings.root, 'backend', 'app', 'api', 'server.py');
    if (!fs.existsSync(serverPath)) {
      throw new Error(`RAG Doctor backend was not found at ${serverPath}. Set ragDoctor.backendPath to the RAG Doctor project folder.`);
    }
    if (!this.backendProcess || this.backendProcess.exitCode !== null) {
      const python = settings.python;
      this.output.appendLine(`Starting local analyzer on 127.0.0.1:${settings.port}`);
      this.backendProcess = spawn(python, [
        '-m', 'uvicorn', 'backend.app.api.server:app',
        '--host', '127.0.0.1', '--port', String(settings.port), '--no-access-log',
      ], {
        cwd: settings.root,
        env: {
          ...process.env,
          PYTHONUNBUFFERED: '1',
          RAG_DOCTOR_ALLOWED_ROOTS: JSON.stringify(roots),
        },
        windowsHide: true,
        stdio: ['ignore', 'pipe', 'pipe'],
      });
      this.ownsBackend = true;
      this.backendRoots = normalizedRoots;
      this.backendProcess.stdout.on('data', (data) => this.output.append(data.toString()));
      this.backendProcess.stderr.on('data', (data) => this.output.append(data.toString()));
      this.backendProcess.on('error', (error) => this.output.appendLine(`Backend process error: ${error.message}`));
    }

    const deadline = Date.now() + 15000;
    while (Date.now() < deadline) {
      if (await this.request(settings.port, 'GET', '/api/health').then((x) => x.status === 200).catch(() => false)) {
        return settings;
      }
      await new Promise((resolve) => setTimeout(resolve, 300));
    }
    throw new Error(`Could not start the local RAG Doctor service. Choose “RAG Doctor: Show Output” for startup details.`);
  }

  request(port, method, route, body, timeoutMs = 10000) {
    return new Promise((resolve, reject) => {
      const payload = body === undefined ? null : Buffer.from(JSON.stringify(body));
      const req = http.request({
        hostname: '127.0.0.1', port, path: route, method,
        headers: payload ? { 'Content-Type': 'application/json', 'Content-Length': payload.length } : {},
        timeout: timeoutMs,
      }, (res) => {
        const chunks = [];
        res.on('data', (chunk) => chunks.push(chunk));
        res.on('end', () => {
          const text = Buffer.concat(chunks).toString('utf8');
          let data;
          try { data = text ? JSON.parse(text) : {}; } catch { data = { detail: text }; }
          resolve({ status: res.statusCode || 0, data });
        });
      });
      req.on('error', reject);
      req.on('timeout', () => req.destroy(new Error('Local analyzer request timed out.')));
      if (payload) req.write(payload);
      req.end();
    });
  }

  async scan(webview) {
    if (!vscode.workspace.isTrusted) {
      webview.postMessage({ type: 'error', message: 'Trust this workspace in VS Code before scanning. RAG Doctor reads local workspace files.' });
      return;
    }
    const folders = this.getWorkspaceFolders();
    if (!folders.length) {
      webview.postMessage({ type: 'error', message: 'Open a local folder or workspace, then scan again.' });
      return;
    }
    webview.postMessage({ type: 'loading' });
    try {
      const settings = await this.ensureBackend(folders.map((f) => f.path));
      const results = [];
      const includeContent = vscode.workspace.getConfiguration('ragDoctor').get('includeRunContent', false) === true;
      for (const folder of folders) {
        const route = `/api/analyze-workspace?workspace_path=${encodeURIComponent(folder.path)}&include_content=${includeContent}`;
        const response = await this.request(settings.port, 'GET', route);
        if (response.status !== 200) {
          const detail = response.data && response.data.detail ? response.data.detail : `HTTP ${response.status}`;
          throw new Error(`${folder.name}: ${detail}`);
        }
        const codeScan = response.data && response.data.code_scan || {};
        const workspaceCodeReferences = [
          ...(codeScan.static_warnings || []).slice(0, 12).map((item) => ({ file: item.file, line: item.line, label: item.root_cause || 'static finding' })),
          ...(codeScan.pipeline_locations || []).slice(0, 20).map((item) => ({ file: item.file, line: item.line, label: item.component || 'RAG pipeline' })),
        ].filter((item) => typeof item.file === 'string' && Number.isInteger(Number(item.line)));
        this.latestDiagnoses.set(folder.path, {
          diagnosis: response.data && response.data.diagnosis || null,
          workspaceCodeReferences,
        });
        results.push({ name: folder.name, path: folder.path, data: response.data });
      }
      webview.postMessage({ type: 'result', results });
    } catch (error) {
      webview.postMessage({ type: 'error', message: error.message || String(error) });
    }
  }

  async runWithDoctor() {
    if (!vscode.workspace.isTrusted) {
      vscode.window.showWarningMessage('Trust this workspace before running an app under RAG Doctor.');
      return;
    }
    const folders = this.getWorkspaceFolders();
    if (!folders.length) {
      vscode.window.showWarningMessage('Open the local RAG application workspace first.');
      return;
    }
    let folder = folders[0];
    if (folders.length > 1) {
      const selected = await vscode.window.showQuickPick(folders.map((item) => ({ label: item.name, description: item.path, item })), { placeHolder: 'Choose the RAG application workspace' });
      if (!selected) return;
      folder = selected.item;
    }
    const command = await vscode.window.showInputBox({
      title: 'Run with RAG Doctor',
      prompt: 'Enter the command you normally use to start this RAG app. RAG Doctor will launch it with a temporary, local runtime observer; it will not edit app source.',
      placeHolder: '.\\.venv\\Scripts\\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000',
      ignoreFocusOut: true,
      validateInput: (value) => value.trim() ? null : 'Enter the app start command.',
    });
    if (!command) return;

    let settings;
    let scan;
    try {
      settings = await this.ensureBackend(folders.map((item) => item.path));
      const scanResponse = await this.request(settings.port, 'GET', `/api/analyze-workspace?workspace_path=${encodeURIComponent(folder.path)}`);
      if (scanResponse.status !== 200) throw new Error(scanResponse.data.detail || `Workspace scan failed (HTTP ${scanResponse.status}).`);
      scan = scanResponse.data;
    } catch (error) {
      vscode.window.showErrorMessage(error.message || 'Could not prepare the local RAG Doctor analyzer.');
      return;
    }

    const evaluationCases = scan.code_scan && scan.code_scan.evaluation_cases || [];
    const manualCaptureOnly = evaluationCases.length === 0;
    const pythonObserverPath = path.join(this.context.extensionPath, 'agent', 'python', 'sitecustomize.py');
    const nodeObserverPath = path.join(this.context.extensionPath, 'agent', 'node', 'rag-doctor-preload.cjs');
    const hasPythonObserver = fs.existsSync(pythonObserverPath);
    const hasNodeObserver = fs.existsSync(nodeObserverPath);
    if (!hasPythonObserver && !hasNodeObserver) {
      vscode.window.showErrorMessage('The bundled Python and Node.js runtime observers are missing from this RAG Doctor installation.');
      return;
    }
    const observerDescription = [hasPythonObserver && 'Python', hasNodeObserver && 'Node.js'].filter(Boolean).join(' and ');
    const confirmed = await vscode.window.showWarningMessage(
      manualCaptureOnly
        ? `RAG Doctor will start the app with temporary local ${observerDescription} observation. Submit a prompt in your app, then scan the workspace to inspect the captured run. No labeled cases were found, so RAG Doctor will not replay requests or score answers automatically.`
        : `RAG Doctor will start the app and replay ${evaluationCases.length} local test case(s). This invokes your app's configured model and may use its API quota. Prompt, retrieved context, and response will be captured in local analyzer memory for this run.`,
      { modal: true }, manualCaptureOnly ? 'Start app' : 'Start and evaluate', 'Cancel',
    );
    if (confirmed !== (manualCaptureOnly ? 'Start app' : 'Start and evaluate')) return;

    const portList = this._candidatePorts(command);
    const alreadyRunning = await this._findFastApi([portList[0]]);
    if (alreadyRunning) {
      vscode.window.showWarningMessage(
        `A FastAPI app is already answering on port ${alreadyRunning.port}. Stop that app first, then run this command again. RAG Doctor must launch the app process itself to attach its temporary observer.`,
        { modal: true }, 'OK',
      );
      return;
    }

    try {
      const reset = await this.request(settings.port, 'POST', '/api/runtime-agent-reset', { workspace_path: folder.path });
      if (reset.status !== 200) throw new Error(reset.data.detail || `HTTP ${reset.status}`);
    } catch (error) {
      vscode.window.showErrorMessage(`Could not reset RAG Doctor's in-memory runtime capture: ${error.message}`);
      return;
    }

    const agentPath = path.join(this.context.extensionPath, 'agent', 'python');
    const oldPythonPath = process.env.PYTHONPATH || '';
    const oldNodeOptions = process.env.NODE_OPTIONS || '';
    const nodePreload = hasNodeObserver
      ? `--require="${nodeObserverPath.replace(/\\/g, '/')}"`
      : '';
    const terminal = vscode.window.createTerminal({
      name: 'RAG Doctor — Observed App',
      cwd: folder.path,
      env: {
        ...(hasPythonObserver ? { PYTHONPATH: [agentPath, oldPythonPath].filter(Boolean).join(path.delimiter) } : {}),
        ...(nodePreload ? { NODE_OPTIONS: [nodePreload, oldNodeOptions].filter(Boolean).join(' ') } : {}),
        RAG_DOCTOR_CAPTURE: '1',
        RAG_DOCTOR_PORT: String(settings.port),
        RAG_DOCTOR_WORKSPACE: folder.path,
      },
    });
    terminal.show(true);
    terminal.sendText(command, true);
    this.output.appendLine(`Launched app with local runtime observation in ${folder.path}`);

    const agent = await this._waitForRuntimeAgent(settings.port, folder.path, 12000);
    if (!agent || agent.status !== 'ready') {
      const detail = agent && agent.detail ? ` ${agent.detail}` : '';
      vscode.window.showWarningMessage(
        `The app was launched, but RAG Doctor's runtime observer did not attach.${detail} Confirm that the launched app uses Python with sitecustomize enabled, or Node.js with global fetch and the injected NODE_OPTIONS.`,
      );
      return;
    }

    if (manualCaptureOnly) {
      const action = await vscode.window.showInformationMessage(
        `The app is running with local ${observerDescription} observation. Submit a prompt in the app; then click Scan workspace in RAG Doctor to load the latest run and investigate it with your local AI model. The capture stays in analyzer memory.`,
        'Open RAG Doctor',
      );
      if (action === 'Open RAG Doctor') await vscode.commands.executeCommand('ragDoctor.openStudio');
      return;
    }

    // Bind replay to the command's requested port. Falling back to common ports
    // can discover a different FastAPI service and make it look like this app
    // started successfully when the launched command has not started at all.
    const targetPort = portList[0];
    const api = await this._waitForFastApi([targetPort], 60000);
    if (!api) {
      vscode.window.showWarningMessage('The app was launched with the observer, but RAG Doctor could not find a FastAPI OpenAPI endpoint on the expected local ports. Supported Python model clients and Node.js global fetch calls can still be captured for a manual run; automated replay needs a discoverable FastAPI query route. After submitting a prompt, scan the workspace to inspect the latest capture.');
      return;
    }

    const operation = this._selectQueryOperation(api.spec);
    if (!operation) {
      vscode.window.showWarningMessage('FastAPI is running, but RAG Doctor could not identify a POST endpoint with a query/question field. The app is running with the observer; automated evaluation is not available for this API shape yet.');
      return;
    }

    const results = [];
    const evaluationRunId = crypto.randomBytes(8).toString('hex');
    for (const testCase of evaluationCases.slice(0, 100)) {
      try {
        await this.request(settings.port, 'POST', '/api/captured-run-reset', { workspace_path: folder.path });
      } catch { /* Evaluation endpoint will report missing capture if reset is unavailable. */ }
      const requestBody = this._buildCaseRequest(operation.schema, testCase, evaluationRunId);
      if (!requestBody) {
        results.push({ testCase, error: 'The detected API requires fields RAG Doctor cannot safely infer.' });
        continue;
      }
      try {
        const response = await this.request(api.port, 'POST', operation.route, requestBody, 120000);
        if (response.status < 200 || response.status >= 300) {
          results.push({ testCase, error: `App endpoint returned HTTP ${response.status}.` });
          continue;
        }
        const evaluated = await this.request(settings.port, 'POST', '/api/evaluate-captured-run', {
          workspace_path: folder.path,
          expected_answer: testCase.expected_answer,
          case_id: testCase.case_id,
          gold_document_id: testCase.gold_document_id,
          question_type: testCase.question_type,
          top_k_final: Number.isInteger(requestBody.top_k) ? requestBody.top_k : (Number.isInteger(requestBody.top_k_final) ? requestBody.top_k_final : undefined),
        });
        results.push(evaluated.status === 200
          ? {
            testCase,
            appResponse: response.data,
            diagnosis: evaluated.data.diagnosis,
            replay: { port: api.port, route: operation.route, requestBody },
          }
          : { testCase, error: evaluated.status === 404
            ? 'The app answered, but no supported model API call was intercepted for this test. The observer covers Python requests, HTTPX, urllib, and aiohttp calls plus Node.js built-in fetch to supported Google, OpenAI, Anthropic, and Ollama endpoints; Node http/https clients, other custom transports, and providers may need an adapter.'
            : evaluated.data.detail || 'No supported model-runtime capture was received for this case.' });
      } catch (error) {
        results.push({ testCase, error: error.message || 'The test request failed.' });
      }
    }
    const sourceFingerprint = await fingerprintWorkspace(folder.path);
    const currentSnapshot = snapshotEvaluation(results, sourceFingerprint);
    const previousSnapshot = this.previousSuiteSnapshots.get(folder.path);
    const comparison = previousSnapshot
      ? { firstRun: false, metrics: compareSnapshots(previousSnapshot, currentSnapshot), sourceFingerprint }
      : { firstRun: true, baselineSaved: currentSnapshot.cases.length > 0, sourceFingerprint };
    if (currentSnapshot.cases.length) this.previousSuiteSnapshots.set(folder.path, currentSnapshot);
    this.showEvaluationReport(folder, results, comparison);
  }

  _candidatePorts(command) {
    const match = command.match(/(?:--port(?:=|\s+)|:)(\d{2,5})\b/i);
    const values = [match ? Number(match[1]) : 8000, 8000, 8080, 5000, 3000, 5001];
    return [...new Set(values.filter((port) => port >= 1024 && port <= 65535))];
  }

  async _waitForFastApi(ports, timeoutMs) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      for (const port of ports) {
        try {
          const response = await this.request(port, 'GET', '/openapi.json', undefined, 1500);
          if (response.status === 200 && response.data && response.data.paths) return { port, spec: response.data };
        } catch { /* The server may still be starting. */ }
      }
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
    return null;
  }

  async _findFastApi(ports) {
    for (const port of ports) {
      try {
        const response = await this.request(port, 'GET', '/openapi.json', undefined, 1200);
        if (response.status === 200 && response.data && response.data.paths) return { port, spec: response.data };
      } catch { /* No app is listening on this port. */ }
    }
    return null;
  }

  async _waitForRuntimeAgent(port, workspacePath, timeoutMs) {
    const deadline = Date.now() + timeoutMs;
    const route = `/api/runtime-agent-status?workspace_path=${encodeURIComponent(workspacePath)}`;
    while (Date.now() < deadline) {
      try {
        const response = await this.request(port, 'GET', route, undefined, 1500);
        if (response.status === 200 && response.data && response.data.status !== 'not_ready') return response.data;
      } catch { /* Wait for sitecustomize to send the local beacon. */ }
      await new Promise((resolve) => setTimeout(resolve, 300));
    }
    return null;
  }

  _selectQueryOperation(spec) {
    const candidates = [];
    for (const [route, methods] of Object.entries(spec.paths || {})) {
      const post = methods && methods.post;
      if (!post) continue;
      let schema = post.requestBody && post.requestBody.content && post.requestBody.content['application/json'] && post.requestBody.content['application/json'].schema;
      if (schema && schema.$ref) {
        const name = schema.$ref.split('/').pop();
        schema = spec.components && spec.components.schemas && spec.components.schemas[name];
      }
      const properties = schema && schema.properties || {};
      const questionKey = ['query', 'question', 'user_query', 'prompt', 'input'].find((key) => properties[key]);
      if (!questionKey) continue;
      const score = (/query|chat|ask/i.test(route) ? 2 : 0) + (questionKey === 'query' || questionKey === 'question' ? 1 : 0);
      candidates.push({ route, schema, questionKey, score });
    }
    candidates.sort((a, b) => b.score - a.score);
    return candidates[0] || null;
  }

  _buildCaseRequest(schema, testCase, runId = crypto.randomBytes(8).toString('hex')) {
    const body = {};
    body[this._selectQuestionKey(schema)] = testCase.question;
    for (const [key, field] of Object.entries(schema.properties || {})) {
      if (key === this._selectQuestionKey(schema)) continue;
      if (key === 'session_id') body[key] = `rag-doctor-${runId}-${testCase.case_id}`;
      else if (Object.prototype.hasOwnProperty.call(field, 'default')) body[key] = field.default;
      else if (key === 'top_k') body[key] = 5;
      else if ((schema.required || []).includes(key)) return null;
    }
    return body;
  }

  _selectQuestionKey(schema) {
    return ['query', 'question', 'user_query', 'prompt', 'input'].find((key) => schema.properties && schema.properties[key]);
  }

  _renderRunComparison(comparison, escape) {
    if (!comparison) return '';
    const fingerprint = comparison.sourceFingerprint || {};
    const coverage = `Fingerprint covers ${fingerprint.files || 0} source/config file(s) and ${fingerprint.totalBytes || 0} byte(s)${fingerprint.complete ? '' : ' (partial due to scan limits or inaccessible files)'}.`;
    if (comparison.firstRun) {
      const baseline = comparison.baselineSaved
        ? 'This run becomes a metrics-only baseline in extension memory for comparison with your next run of this workspace.'
        : 'No captured cases were available, so RAG Doctor did not save a comparison baseline.';
      return `<section class="case"><h2>Repeatable run baseline</h2><p>This is the first completed suite in this VS Code session. ${escape(baseline)} ${escape(coverage)}</p></section>`;
    }
    const result = comparison.metrics || {};
    const paired = Number(result.pairedCount) || 0;
    const score = (before, after) => paired ? `${before}/${paired} → ${after}/${paired}` : 'unavailable; no paired cases';
    const latency = (value) => Number.isFinite(value) ? `${value.toFixed(1)} ms` : 'unavailable';
    const sourceDescription = {
      changed: 'Tracked source/config files changed between runs.',
      unchanged: 'No tracked source/config change was detected between runs.',
      partial: 'Source comparison is partial; files outside the fingerprint limits may have changed.',
      unavailable: 'A complete source comparison was unavailable.',
    }[result.sourceChange] || 'Source change status unavailable.';
    const cases = (result.cases || []).map((item) => `<li>${escape(item.caseId)}: ${escape(item.beforeCause)} → ${escape(item.afterCause)}${item.supportChanged ? ' · answer-support status changed' : ''}${item.goldChanged ? ' · labeled-document presence changed' : ''}</li>`).join('');
    return `<section class="case"><h2>Repeatable run comparison</h2><p>${paired} case(s) paired across the previous ${result.beforeCapturedCases}/${result.beforeTotalCases} and current ${result.afterCapturedCases}/${result.afterTotalCases} captured suites.</p><p>Reference-supported outcomes (current evaluator): ${escape(score(result.beforeSupported, result.afterSupported))}.</p><p>Labeled source-document presence: ${escape(score(result.beforeGold, result.afterGold))}.</p><p>Mean app-reported latency: ${escape(latency(result.beforeLatencyMs))} → ${escape(latency(result.afterLatencyMs))}.</p><p>${escape(sourceDescription)} ${escape(coverage)}</p>${cases ? `<details><summary>Per-case changes</summary><ul>${cases}</ul></details>` : ''}<p>This comparison measures the same labeled case IDs; it does not prove a code change caused the outcome. Model/provider variance may affect answers.</p></section>`;
  }

  showEvaluationReport(folder, results, comparison = null) {
    const panel = vscode.window.createWebviewPanel('ragDoctorEvaluation', 'RAG Doctor — Evaluation', vscode.ViewColumn.Beside, { enableScripts: true });
    const nonce = crypto.randomBytes(16).toString('base64');
    const escape = (value) => String(value ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    const diagnosed = results.filter((item) => item.diagnosis);
    const noIssue = diagnosed.filter((item) => item.diagnosis.root_cause === 'NO_FAILURE').length;
    const comparisonBlock = this._renderRunComparison(comparison, escape);
    const replayableCount = results.filter((item) => item && item.testCase && item.diagnosis && item.replay).length;
    const suiteVerification = replayableCount
      ? `<section class="case"><h2>Verify a code change</h2><p>After applying a suggested fix and confirming the app has reloaded it, replay the same ${replayableCount} labeled case(s) against the current app. RAG Doctor compares paired outcomes and latency with this baseline; it does not edit source. Track a suggested exact replacement either before applying it or after applying the unchanged replacement.</p><button class="verify" data-verify-suite="1">Replay labeled cases against current app</button><pre id="suite-verification-output" hidden></pre><div id="suite-verification-answers" hidden></div></section>`
      : '';
    const rows = results.map((item) => {
      const test = item.testCase;
      const diagnosis = item.diagnosis;
      const summary = diagnosis
        ? `${diagnosis.root_cause || 'INSUFFICIENT_EVIDENCE'} · ${diagnosis.finding_label || diagnosis.status} · ${diagnosis.evidence_level} evidence`
        : `Not evaluated · ${item.error || 'No result'}`;
      const codeRefs = diagnosis ? (diagnosis.code_references || []).map((ref) => `<li><button class="loc" data-open="1" data-file="${escape(ref.file)}" data-line="${Number(ref.line) || 1}">Open ${escape(ref.file)}:${Number(ref.line) || 1}</button><p>${escape(ref.label)}</p><pre>${escape(ref.code_snippet || '')}</pre></li>`).join('') : '';
      const arithmeticBlock = diagnosis && diagnosis.arithmetic_checks && diagnosis.arithmetic_checks.length
        ? `<section><h3>Deterministic arithmetic verification</h3><ul>${diagnosis.arithmetic_checks.map((check) => `<li><code>${escape(check.expression)}</code> computes to <b>${escape(check.computed)}</b>; the response states <b>${escape(check.reported)}</b>.</li>`).join('')}</ul></section>`
        : '';
      const details = diagnosis ? `<p>${escape(diagnosis.explanation)}</p><p><b>Suggested next step:</b> ${escape(diagnosis.recommended_fix)}</p>${arithmeticBlock}${codeRefs ? `<section><h3>Relevant code and experiments</h3><ol>${codeRefs}</ol><p>These locations are static matches to the diagnosed or uncertain pipeline stage. Confirm the highlighted code is used by this run before editing it.</p></section>` : '<p>No matching code location was found. RAG Doctor will not invent a file or line.</p>'}<button class="investigate" data-investigate="${results.indexOf(item)}">Investigate with local AI</button><pre class="investigation" id="investigation-${results.indexOf(item)}" hidden></pre><button class="verify" data-track-fix="${results.indexOf(item)}" hidden>Track this exact fix</button><pre id="fix-tracking-${results.indexOf(item)}" hidden></pre><details><summary>Run evidence</summary><p><b>Expected:</b> ${escape(test.expected_answer)}</p><p><b>Actual:</b> ${escape(diagnosis.llm_answer)}</p><p><b>Prompt:</b><pre>${escape(diagnosis.prompt)}</pre></p><p><b>Retrieved passages:</b>${(diagnosis.retrieved_chunks || []).map((chunk) => `<pre>#${escape(chunk.final_rank)} ${escape(chunk.doc_id || '')}\n${escape(chunk.text)}</pre>`).join('')}</p><p><b>Evidence:</b> ${escape((diagnosis.evidence || []).join(' · '))}</p></details>` : `<p>${escape(item.error || '')}</p>`;
      const verifyButton = diagnosis && item.replay && Number.isInteger(item.replay.requestBody.top_k)
        ? `<button class="verify" data-verify="${results.indexOf(item)}" hidden>Run verified experiment</button>`
        : '';
      return `<details class="case"><summary>${escape(test.case_id)} · ${escape(summary)}</summary><p><b>Question:</b> ${escape(test.question)}</p>${details}${verifyButton}</details>`;
    }).join('');
    panel.webview.html = `<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'nonce-${nonce}';"><style>body{font:14px var(--vscode-font-family);color:var(--vscode-foreground);background:var(--vscode-editor-background);padding:22px}.case{border:1px solid var(--vscode-panel-border);padding:12px;border-radius:6px;margin:10px 0}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:var(--vscode-textCodeBlock-background);padding:10px}.loc{color:var(--vscode-textLink-foreground);background:transparent;border:0;padding:0;text-decoration:underline;cursor:pointer}.case p{line-height:1.5}.investigate,.verify{background:var(--vscode-button-background);color:var(--vscode-button-foreground);border:0;border-radius:4px;padding:7px 10px;margin:6px 8px 6px 0;cursor:pointer}.investigation{margin-top:10px;border-left:3px solid var(--vscode-focusBorder)}</style></head><body><h1>RAG Doctor evaluation</h1><p>Workspace: ${escape(folder.name)} · ${diagnosed.length}/${results.length} cases captured · ${noIssue} with no obvious issue</p><p>Expand a case, then choose <b>Investigate with local AI</b> to let a local model inspect the captured run and read-only workspace evidence. The investigator cannot edit files or run commands.</p>${comparisonBlock}${suiteVerification}${rows}<script nonce="${nonce}">const vscode=acquireVsCodeApi();document.addEventListener('click',event=>{const button=event.target.closest('[data-open]');if(button){vscode.postMessage({type:'openLocation',file:button.dataset.file,line:Number(button.dataset.line)});return;}const suite=event.target.closest('[data-verify-suite]');if(suite){suite.disabled=true;suite.textContent='Replaying labeled cases…';const output=document.getElementById('suite-verification-output');output.hidden=false;output.textContent='Preparing paired local runs…';vscode.postMessage({type:'verifyCurrentSuite'});return;}const track=event.target.closest('[data-track-fix]');if(track){track.disabled=true;track.textContent='Tracking…';vscode.postMessage({type:'trackFix',index:Number(track.dataset.trackFix)});return;}const verify=event.target.closest('[data-verify]');if(verify){verify.disabled=true;verify.textContent='Running verification…';vscode.postMessage({type:'verifyExperiment',index:Number(verify.dataset.verify)});return;}const investigate=event.target.closest('[data-investigate]');if(investigate){const output=document.getElementById('investigation-'+investigate.dataset.investigate);output.hidden=false;output.textContent='Investigating with the configured local model…';investigate.disabled=true;vscode.postMessage({type:'investigateCase',index:Number(investigate.dataset.investigate)});}});window.addEventListener('message',event=>{const message=event.data||{};if(message.type==='suiteVerificationProgress'||message.type==='suiteVerificationResult'){const output=document.getElementById('suite-verification-output');const answers=document.getElementById('suite-verification-answers');const suite=document.querySelector('[data-verify-suite]');if(output){output.hidden=false;output.textContent=message.text||'No verification result.';}if(answers&&message.type==='suiteVerificationResult'){answers.replaceChildren();for(const pair of message.answerComparisons||[]){const details=document.createElement('details');const summary=document.createElement('summary');summary.textContent=pair.case_id+' · '+(pair.changed===null?'answer unavailable':pair.changed?'answer changed':'answer text identical')+(pair.retrieval_changed?' · retrieved set changed':'');details.appendChild(summary);for(const [label,value] of [['Before',pair.before],['After',pair.after]]){const heading=document.createElement('h3');heading.textContent=label;const answer=document.createElement('pre');answer.textContent=value+(pair.truncated?String.fromCharCode(10)+'…answer text truncated for display':'');details.appendChild(heading);details.appendChild(answer);}for(const [label,value] of [['Retrieved metadata before',pair.retrieval_before],['Retrieved metadata after',pair.retrieval_after]]){const heading=document.createElement('h3');heading.textContent=label;const retrieval=document.createElement('pre');retrieval.textContent=value&&value.length?JSON.stringify(value,null,2):'No retrieval metadata captured';details.appendChild(heading);details.appendChild(retrieval);}answers.appendChild(details);}answers.hidden=!answers.childElementCount;}if(suite&&message.type==='suiteVerificationResult'){suite.disabled=false;suite.textContent='Replay labeled cases against current app';}return;}if(message.type==='fixTrackingResult'){const output=document.getElementById('fix-tracking-'+message.index);const track=document.querySelector('[data-track-fix="'+message.index+'"]');if(output){output.hidden=false;output.textContent=message.text||'No tracking result.';}if(track){track.disabled=false;track.textContent=message.tracked?'Tracked for the next replay':'Track this exact fix';}return;}const output=document.getElementById('investigation-'+message.index);const button=document.querySelector('[data-investigate="'+message.index+'"]');const verify=document.querySelector('[data-verify="'+message.index+'"]');const track=document.querySelector('[data-track-fix="'+message.index+'"]');if(message.type==='investigationResult'){if(output){output.hidden=false;output.textContent=message.text||'No investigation result.';}if(button)button.disabled=false;if(verify){verify.hidden=!message.canVerify;if(message.canVerify)verify.textContent='Run top_k '+message.experiment.baseline+' → '+message.experiment.candidate+' across suite';}if(track)track.hidden=!message.canTrackFix;return;}if(message.type==='verificationProgress'){if(output){output.hidden=false;output.textContent=message.text||'Replaying cases…';}return;}if(message.type==='verificationResult'){if(output){output.hidden=false;output.textContent=message.text||'No verification result.';}if(verify){verify.disabled=false;verify.textContent='Run verified experiment';}}});</script></body></html>`;
    const messageSubscription = panel.webview.onDidReceiveMessage((message) => {
      if (message && message.type === 'openLocation') void this.openLocation({ ...message, workspace: folder.path });
      if (message && message.type === 'investigateCase') void this.investigateEvaluationCase(panel, folder, results, message.index);
      if (message && message.type === 'verifyExperiment') void this.verifyEvaluationSuite(panel, folder, results, message.index);
      if (message && message.type === 'verifyCurrentSuite') void this.verifyCurrentSuite(panel, folder, results, comparison && comparison.sourceFingerprint);
      if (message && message.type === 'trackFix') void this.trackFixForVerification(panel, folder, results, message.index);
    });
    panel.onDidDispose(() => messageSubscription.dispose());
    this.context.subscriptions.push(panel);
  }

  async investigateEvaluationCase(panel, folder, results, index) {
    const item = results[Number(index)];
    if (!item || !item.diagnosis) return;
    const settings = this.getSettings();
    if (!settings.localModelName) {
      const modelName = await vscode.window.showInputBox({
        title: 'Local RAG Doctor model',
        prompt: 'Enter the model name available in your local Ollama server. RAG Doctor sends this case and selected workspace excerpts only to the loopback model endpoint.',
        placeHolder: 'qwen3:4b',
        ignoreFocusOut: true,
        validateInput: (value) => value.trim() ? null : 'Enter an installed local model name.',
      });
      if (!modelName) {
        panel.webview.postMessage({ type: 'investigationResult', index, text: 'Investigation canceled. Set ragDoctor.localModelName to use a local Ollama model.' });
        return;
      }
      settings.localModelName = modelName.trim();
    }
    try {
      const backend = await this.ensureBackend(this.getWorkspaceFolders().map((item) => item.path));
      const response = await this.request(backend.port, 'POST', '/api/agentic-investigate', {
        workspace_path: folder.path,
        case_id: item.testCase.case_id,
        question: item.testCase.question,
        expected_answer: item.testCase.expected_answer,
        gold_document_id: item.testCase.gold_document_id,
        question_type: item.testCase.question_type,
        diagnosis: item.diagnosis,
        local_model_endpoint: settings.localModelEndpoint,
        local_model_name: settings.localModelName,
        local_model_thinking: settings.localModelThinking,
      }, 240000);
      if (response.status !== 200) throw new Error(response.data.detail || `Local investigator returned HTTP ${response.status}.`);
      item.investigation = this.validateProposedFix(folder.path, response.data.investigation);
      const fix = item.investigation && item.investigation.fix;
      const canTrackFix = Boolean(fix && fix.file && Number.isInteger(fix.line) && fix.source_matches && fix.replacement_code && fix.syntax_status !== 'invalid' && fix.syntax_status !== 'source_changed' && fix.syntax_status !== 'source_unverifiable');
      const experiment = item.investigation && item.investigation.experiment;
      const canVerify = Boolean(
        experiment
        && experiment.parameter === 'top_k'
        && item.replay
        && Number.isInteger(item.replay.requestBody.top_k)
        && item.replay.requestBody.top_k === experiment.baseline
        && Number.isInteger(experiment.candidate)
        && experiment.candidate >= 1
        && experiment.candidate <= 100
      );
      panel.webview.postMessage({
        type: 'investigationResult',
        index,
        text: this.formatInvestigation(item.investigation),
        canVerify,
        canTrackFix,
        experiment: canVerify ? experiment : null,
      });
    } catch (error) {
      panel.webview.postMessage({ type: 'investigationResult', index, text: `Investigation failed: ${error.message || String(error)}\n\nStart the local Ollama server, install a model, and configure ragDoctor.localModelName. Workspace evidence is not sent to a remote model.` });
    }
  }

  async investigateLatestRun(webview, workspacePath, index, workspaceAudit = false) {
    index = Number(index);
    if (!Number.isInteger(index) || index < 0) return;
    const folder = this.getWorkspaceFolders().find((item) => item.path === workspacePath);
    if (!vscode.workspace.isTrusted || !folder) {
      webview.postMessage({ type: 'latestInvestigationResult', index, workspaceAudit, text: 'The workspace is no longer trusted or open. Reopen the workspace and scan again.' });
      return;
    }
    const scanEvidence = this.latestDiagnoses.get(folder.path) || {};
    const diagnosis = scanEvidence.diagnosis || {};
    if (!workspaceAudit && !Object.keys(diagnosis).length) {
      webview.postMessage({ type: 'latestInvestigationResult', index, workspaceAudit, text: 'No captured run is available. Use the workspace AI audit for a code-only review, or capture a run and scan again.' });
      return;
    }

    const settings = this.getSettings();
    if (!settings.localModelName) {
      const modelName = await vscode.window.showInputBox({
        title: 'Local RAG Doctor model',
        prompt: workspaceAudit
          ? 'Enter a model installed in your local Ollama server. The investigator sends bounded workspace source/config/log excerpts only to this local endpoint.'
          : 'Enter a model installed in your local Ollama server. The investigator sends the captured run and selected workspace excerpts only to this local endpoint.',
        placeHolder: 'qwen3:4b',
        ignoreFocusOut: true,
        validateInput: (value) => value.trim() ? null : 'Enter an installed local model name.',
      });
      if (!modelName) {
        webview.postMessage({ type: 'latestInvestigationResult', index, workspaceAudit, text: 'Investigation canceled. Set ragDoctor.localModelName to use a local Ollama model.' });
        return;
      }
      settings.localModelName = modelName.trim();
    }
    webview.postMessage({ type: 'latestInvestigationProgress', index, workspaceAudit, text: workspaceAudit ? 'Reviewing RAG source/configuration with your local model…' : 'Investigating the latest captured run with your local model…' });
    try {
      const backend = await this.ensureBackend(this.getWorkspaceFolders().map((item) => item.path));
      const response = await this.request(backend.port, 'POST', '/api/agentic-investigate', {
        workspace_path: folder.path,
        case_id: diagnosis.case_id || null,
        question: diagnosis.question || '',
        expected_answer: diagnosis.expected_answer || '',
        gold_document_id: diagnosis.gold_document_id || null,
        question_type: diagnosis.question_type || null,
        diagnosis: {
          ...diagnosis,
          workspace_code_references: scanEvidence.workspaceCodeReferences || [],
        },
        local_model_endpoint: settings.localModelEndpoint,
        local_model_name: settings.localModelName,
        local_model_thinking: settings.localModelThinking,
      }, 240000);
      if (response.status !== 200) throw new Error(response.data.detail || `Local investigator returned HTTP ${response.status}.`);
      webview.postMessage({
        type: 'latestInvestigationResult',
        index,
        workspaceAudit,
        text: this.formatInvestigation(this.validateProposedFix(folder.path, response.data.investigation)),
      });
    } catch (error) {
      webview.postMessage({
        type: 'latestInvestigationResult',
        index,
        workspaceAudit,
        text: `Investigation failed: ${error.message || String(error)}\n\nStart the local Ollama server, install a model, and configure ragDoctor.localModelName. Workspace evidence is not sent to a remote model.`,
      });
    }
  }

  validateProposedFix(workspacePath, investigation) {
    if (!investigation || !investigation.fix || !investigation.fix.source_matches || !investigation.fix.replacement_code) return investigation;
    const fix = investigation.fix;
    const relative = String(fix.file || '');
    const extension = path.extname(relative).toLowerCase();
    const suppressStaleFix = () => {
      fix.source_matches = false;
      fix.replacement_code = '';
      fix.syntax_status = 'source_changed';
      return investigation;
    };
    const suppressUnverifiableFix = () => {
      fix.source_matches = false;
      fix.replacement_code = '';
      fix.syntax_status = 'source_unverifiable';
      return investigation;
    };
    try {
      const workspaceRoot = fs.realpathSync(workspacePath);
      const targetPath = path.resolve(workspaceRoot, relative);
      const targetRealPath = fs.realpathSync(targetPath);
      const relativeTarget = path.relative(workspaceRoot, targetRealPath);
      if (!relativeTarget || relativeTarget === '..' || relativeTarget.startsWith(`..${path.sep}`) || path.isAbsolute(relativeTarget)) return suppressStaleFix();
      const stat = fs.statSync(targetRealPath);
      if (!stat.isFile() || stat.size > 1_000_000) return suppressUnverifiableFix();
      const source = fs.readFileSync(targetRealPath, 'utf8');
      const sourceLines = source.split(/\r\n|\n|\r/);
      const start = Number(fix.line);
      const end = Number(fix.end_line || fix.line);
      if (!Number.isInteger(start) || !Number.isInteger(end) || start < 1 || end < start || end > sourceLines.length) return suppressStaleFix();
      const currentBlock = sourceLines.slice(start - 1, end).join('\n').replace(/\r\n/g, '\n').trim();
      const expectedBlock = String(fix.current_code || '').replace(/\r\n/g, '\n').trim();
      if (!expectedBlock) return suppressStaleFix();
      if (/\[REDACTED_[A-Z0-9_]+\]/.test(expectedBlock)) return suppressUnverifiableFix();
      if (currentBlock !== expectedBlock) return suppressStaleFix();

      // Source matching is rechecked for every language so edits made while
      // the model was responding cannot leave a stale copy-ready fix visible.
      // Syntax parsing below is intentionally limited to Node script syntax.
      if (!['.js', '.cjs'].includes(extension)) return investigation;
      if (extension === '.js') {
        // Node's script parser cannot validate ESM or JSX. Leave those fixes
        // available but explicitly unchecked instead of mislabeling them.
        let directory = path.dirname(targetRealPath);
        while (directory === workspaceRoot || directory.startsWith(`${workspaceRoot}${path.sep}`)) {
          const manifestPath = path.join(directory, 'package.json');
          if (fs.existsSync(manifestPath)) {
            let manifest;
            try {
              const manifestRealPath = fs.realpathSync(manifestPath);
              const relativeManifest = path.relative(workspaceRoot, manifestRealPath);
              if (!relativeManifest || relativeManifest === '..' || relativeManifest.startsWith(`..${path.sep}`) || path.isAbsolute(relativeManifest) || fs.statSync(manifestRealPath).size > 128_000) return investigation;
              manifest = JSON.parse(fs.readFileSync(manifestRealPath, 'utf8'));
            } catch {
              return investigation;
            }
            if (manifest.type === 'module') return investigation;
            break;
          }
          if (directory === workspaceRoot) break;
          directory = path.dirname(directory);
        }
        if (/^\s*(?:import|export)\s/m.test(source) || /(?:return|=>)\s*(?:\(\s*)?</.test(source)) return investigation;
      }

      const replacement = String(fix.replacement_code).replace(/\r\n/g, '\n');
      const patchedSource = [
        ...sourceLines.slice(0, start - 1),
        ...replacement.split('\n'),
        ...sourceLines.slice(end),
      ].join('\n');
      try {
        new vm.Script(patchedSource, { filename: 'RAG Doctor proposed JavaScript' });
        fix.syntax_status = 'valid';
        fix.syntax_language = 'JavaScript';
      } catch {
        fix.syntax_status = 'invalid';
        fix.syntax_language = 'JavaScript';
        fix.replacement_code = '';
      }
    } catch {
      return suppressUnverifiableFix();
    }
    return investigation;
  }

  formatInvestigation(result) {
    const staticVerifier = result.static_verifier;
    const runtimeVerifier = result.runtime_verifier;
    const coverage = result.evidence_coverage || {};
    const verifierFallback = Boolean((staticVerifier && staticVerifier.fallback_applied) || (runtimeVerifier && runtimeVerifier.fallback_applied));
    const deterministicFinding = Boolean(
      (staticVerifier && staticVerifier.finding_detected)
      || (runtimeVerifier && runtimeVerifier.finding_detected)
    );
    const label = verifierFallback || deterministicFinding ? 'Hybrid investigation' : 'AI investigation';
    const lines = [
      `${label}: ${String(result.status || 'insufficient_evidence').toUpperCase()} · ${result.root_cause || 'UNKNOWN'}`,
      '',
      result.summary || 'No supported root cause was returned.',
    ];
    if (Object.keys(coverage).length) {
      const captured = [];
      if (coverage.prompt_captured) captured.push('model prompt');
      if (coverage.answer_captured) captured.push('model answer');
      if (coverage.retrieved_candidates_captured) captured.push('retriever candidates');
      if (coverage.prompt_context_captured) captured.push('context extracted from prompt');
      lines.push('', `Evidence captured: ${captured.length ? captured.join(', ') : 'no prompt, answer, or retriever output'}.`);
      if (coverage.retrieval_status_observed) {
        lines.push(coverage.retrieval_observed
          ? `Retriever status was explicitly observed${Number.isInteger(coverage.retrieval_count) ? ` (${coverage.retrieval_count} candidate(s))` : ''}.`
          : 'Runtime telemetry explicitly reports that retrieval returned no candidates.');
      } else if (!coverage.retrieved_candidates_captured) {
        lines.push('Retriever candidates and ranks were not captured; an empty prompt or missing telemetry does not prove retrieval returned zero results.');
      }
    }
    const verifierDisposition = result.model_confirmed_deterministic_check
      ? 'the model confirmed the cause; the source-verified patch replaced its different or unverified edit'
      : 'used because the model did not confirm the source-grounded finding';
    if (staticVerifier) {
      if (staticVerifier.finding_detected) {
        const disposition = staticVerifier.fallback_applied
          ? verifierDisposition
          : 'cross-checked against the model result';
        lines.push('', `Static verifier: ${staticVerifier.finding_root_cause} in ${staticVerifier.finding_file || 'workspace code'}; ${disposition}.`);
      } else if (staticVerifier.checked) {
        lines.push('', 'Static verifier: no supported deterministic pattern matched; the agentic result is based on the model and cited evidence.');
        const trace = staticVerifier.trace || {};
        const scanned = Number.isFinite(trace.python_files_scanned) ? trace.python_files_scanned : null;
        const parsed = Number.isFinite(trace.python_files_parsed) ? trace.python_files_parsed : null;
        const traceParts = [];
        if (scanned !== null) traceParts.push(`Python files parsed ${parsed ?? 0}/${scanned}`);
        if (Number.isFinite(trace.chunk_step_candidate_count)) traceParts.push(`chunk-step candidates ${trace.chunk_step_candidate_count}`);
        if (Number.isFinite(trace.dropped_context_return_count)) traceParts.push(`prompt builders dropping context ${trace.dropped_context_return_count}`);
        if (trace.matched_stage) traceParts.push(`matched stage ${trace.matched_stage}`);
        if (trace.inventory_error) traceParts.push('workspace inventory reported an error');
        if (traceParts.length) lines.push(`Static audit trace: ${traceParts.join(' · ')}.`);
      }
    }
    if (runtimeVerifier && runtimeVerifier.checked) {
      if (runtimeVerifier.finding_detected) {
        const disposition = runtimeVerifier.fallback_applied
          ? verifierDisposition
          : 'cross-checked against the model result';
        lines.push('', `Runtime verifier: ${runtimeVerifier.finding_root_cause} in ${runtimeVerifier.finding_file || 'workspace code'}; ${disposition}.`);
      } else {
        const hasRankingEvidence = Boolean(coverage.retrieved_candidates_captured || coverage.retrieval_status_observed);
        lines.push('', hasRankingEvidence
          ? 'Runtime verifier: no supported score-order defect matched the available retrieval evidence.'
          : 'Runtime verifier: ranking could not be checked because retriever candidates or explicit retrieval status were not captured.');
      }
    }
    if (result.model_error) {
      lines.push('', `Local model did not complete: ${result.model_error}. Any displayed verifier finding is based on deterministic local evidence checks.`);
    }
    if (result.model_assessment_before_verification) {
      const assessment = result.model_assessment_before_verification;
      const assessmentNote = result.model_confirmed_deterministic_check
        ? 'The model confirmed the cause, but its proposed edit did not match the source-verified patch shown below.'
        : 'The verified finding below is used because the model result did not match the source-grounded check.';
      lines.push(
        '',
        `Model assessment before verification: ${String(assessment.root_cause || 'UNKNOWN')} · ${String(assessment.status || 'insufficient_evidence')}. ${String(assessment.summary || 'No summary was returned')} ${assessmentNote}`,
      );
    }
    if (result.agent_trace) {
      const trace = result.agent_trace;
      const provided = Array.isArray(trace.source_files_provided) ? trace.source_files_provided : [];
      const opened = Array.isArray(trace.source_files_opened) ? trace.source_files_opened : [];
      const tools = Array.isArray(trace.tools_used) ? trace.tools_used : [];
      const paths = (items) => {
        const shown = items.slice(0, 4).map((item) => String(item).replace(/[\r\n]/g, '').slice(0, 100));
        return shown.length ? `${shown.join(', ')}${items.length > shown.length ? `, +${items.length - shown.length} more` : ''}` : 'none';
      };
      const reasoning = trace.thinking_fallback_used
        ? 'retried without reasoning after the model returned no final answer'
        : (trace.thinking_enabled === false ? 'reasoning disabled' : 'reasoning enabled');
      const verifierReview = trace.verifier_review_retry_used ? '; one focused review checked the verifier finding' : '';
      lines.push('', `Agent inspection: ${Number(trace.model_turns) || 0} local model turn(s); ${Number(trace.read_only_tool_calls) || 0} additional read-only tool call(s)${tools.length ? ` (${tools.join(', ')})` : ''}; ${reasoning}${verifierReview}.`);
      lines.push(`Source excerpts supplied to the investigator: ${provided.length ? paths(provided) : 'none'}; additional files opened: ${opened.length ? paths(opened) : 'none'}.`);
    }
    if (result.evidence && result.evidence.length) {
      lines.push('', 'Evidence:');
      for (const item of result.evidence) {
        lines.push(`- ${item.verified ? 'Verified' : 'Unverified'} · ${item.source}: “${item.quote}”`);
      }
    }
    if (result.deterministic_checks && result.deterministic_checks.length) {
      lines.push('', 'Deterministic checks:');
      for (const item of result.deterministic_checks) {
        lines.push(`- ${item.expression} → computed ${item.computed}, response says ${item.reported}`);
      }
    }
    if (result.fix) {
      lines.push('', 'Suggested code change:');
      if (result.fix.file) lines.push(`${result.fix.file}:${result.fix.line}${result.fix.end_line && result.fix.end_line !== result.fix.line ? `-${result.fix.end_line}` : ''}`);
      if (result.fix.current_code) lines.push(`Current source block:\n${result.fix.current_code}`);
      if (result.fix.source_matches && result.fix.replacement_code) lines.push(`Replace with:\n${result.fix.replacement_code}`);
      if (result.fix.syntax_status === 'valid') lines.push(`${result.fix.syntax_language || 'Python'} syntax check passed for the proposed replacement; runtime behavior still needs the repeatable suite check.`);
      else if (result.fix.syntax_status === 'invalid') lines.push(`${result.fix.syntax_language || 'Source'} syntax check failed. The copy-ready replacement was suppressed.`);
      else if (result.fix.syntax_status === 'source_changed') lines.push('Workspace source changed since analysis. The stale replacement was suppressed; investigate again before applying a fix.');
      else if (result.fix.syntax_status === 'source_unverifiable') lines.push('Workspace source could not be rechecked. The copy-ready replacement was suppressed; confirm file access and investigate again.');
      else if (result.fix.source_matches && result.fix.replacement_code) lines.push('Syntax was not checked automatically for this file type or module syntax.');
      if (result.fix.proposed_change) lines.push(`Change: ${result.fix.proposed_change}`);
      if (result.fix.why) lines.push(`Why: ${result.fix.why}`);
      if (result.fix.file && !result.fix.source_matches) lines.push('RAG Doctor verified that this file and line exist, but the model did not provide an exact matching source block. Treat this as guidance, not a copy-ready replacement.');
    } else {
      lines.push('', 'No code edit is proposed because the evidence does not support a specific change.');
    }
    if (result.experiment) {
      lines.push('', `Proposed experiment: top_k ${result.experiment.baseline} → ${result.experiment.candidate}.`);
      if (result.experiment.rationale) lines.push(result.experiment.rationale);
    }
    lines.push('', `Verification: ${result.verification || 'Rerun the same case and compare the evidence.'}`);
    if (result.fix) {
      lines.push('Repeatable check: if you choose to apply the suggestion, confirm the app has reloaded it, then use “Replay labeled cases against current app” in the evaluation report. RAG Doctor compares paired outcomes, latency, and source fingerprints. You can track the exact replacement before applying it, or afterward if you applied it unchanged; the replay also checks whether that code block is present. RAG Doctor never applies the suggestion. Without labeled cases, it cannot measure answer improvement.');
    }
    lines.push('RAG Doctor did not edit files or execute workspace code.');
    return lines.join('\n');
  }

  trackFixForVerification(panel, folder, results, index) {
    const item = results[Number(index)];
    if (!item || !item.investigation || !item.investigation.fix) {
      panel.webview.postMessage({ type: 'fixTrackingResult', index, tracked: false, text: 'No copy-ready recommendation is available to track.' });
      return;
    }
    const originalFix = item.investigation.fix;
    const checked = this.validateProposedFix(folder.path, {
      ...item.investigation,
      fix: { ...originalFix },
    });
    const checkedFix = checked && checked.fix;
    const validFix = (fix) => Boolean(
      fix && fix.file && Number.isInteger(fix.line) && fix.source_matches && fix.replacement_code
      && fix.syntax_status !== 'invalid' && fix.syntax_status !== 'source_changed' && fix.syntax_status !== 'source_unverifiable'
    );
    let fix = validFix(checkedFix) ? checkedFix : null;
    if (!fix && validFix(originalFix)) {
      const replacement = String(originalFix.replacement_code).replace(/\r\n/g, '\n').trim();
      const appliedTracker = {
        caseId: String(item.testCase && item.testCase.case_id || ''),
        file: String(originalFix.file).replace(/\\/g, '/'),
        line: Number(originalFix.line),
        replacementLines: replacement.split('\n').length,
        replacementHash: crypto.createHash('sha256').update(replacement).digest('hex'),
      };
      if (this._trackedFixStatus(folder.path, appliedTracker).status === 'present') fix = originalFix;
    }
    if (!fix) {
      panel.webview.postMessage({ type: 'fixTrackingResult', index, tracked: false, text: 'The current file no longer matches the original source or the exact suggested replacement. Track before editing, or apply the suggested replacement unchanged and try again.' });
      return;
    }
    const replacement = String(fix.replacement_code).replace(/\r\n/g, '\n').trim();
    const tracker = {
      caseId: String(item.testCase && item.testCase.case_id || ''),
      file: String(fix.file).replace(/\\/g, '/'),
      line: Number(fix.line),
      replacementLines: replacement.split('\n').length,
      replacementHash: crypto.createHash('sha256').update(replacement).digest('hex'),
    };
    const pending = this.pendingFixVerifications.get(folder.path) || [];
    const existing = pending.findIndex((entry) => entry.caseId === tracker.caseId && entry.file === tracker.file);
    if (existing >= 0) pending[existing] = tracker;
    else pending.push(tracker);
    this.pendingFixVerifications.set(folder.path, pending);
    panel.webview.postMessage({
      type: 'fixTrackingResult',
      index,
      tracked: true,
      text: `Tracked ${tracker.file}:${tracker.line} in extension memory for this VS Code session. The next replay will check whether the exact suggested replacement is present and compare the same labeled cases.`,
    });
  }

  _trackedFixStatus(workspacePath, tracker) {
    try {
      const workspaceRoot = fs.realpathSync(workspacePath);
      const target = path.resolve(workspaceRoot, tracker.file);
      const real = fs.realpathSync(target);
      const relative = path.relative(workspaceRoot, real);
      if (!relative || relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) return { status: 'unverifiable' };
      const stat = fs.statSync(real);
      if (!stat.isFile() || stat.size > 1_000_000) return { status: 'unverifiable' };
      const lines = fs.readFileSync(real, 'utf8').split(/\r\n|\n|\r/);
      const count = tracker.replacementLines;
      for (let start = 0; start + count <= lines.length; start += 1) {
        const block = lines.slice(start, start + count).join('\n').trim();
        const digest = crypto.createHash('sha256').update(block).digest('hex');
        if (digest === tracker.replacementHash) return { status: 'present', line: start + 1 };
      }
      return { status: 'not_found' };
    } catch {
      return { status: 'unverifiable' };
    }
  }

  async verifyCurrentSuite(panel, folder, results, baselineFingerprint) {
    const cases = (results || []).filter((item) => item && item.testCase && item.diagnosis && item.replay).slice(0, 100);
    if (!cases.length) {
      panel.webview.postMessage({ type: 'suiteVerificationResult', text: 'No captured, replayable labeled cases are available for a post-change comparison.' });
      return;
    }
    const routes = new Set(cases.map((item) => `${item.replay.port}:${item.replay.route}`));
    if (routes.size !== 1) {
      panel.webview.postMessage({ type: 'suiteVerificationResult', text: 'The captured cases use different app endpoints. Rerun the suite against one app endpoint before comparing a code change.' });
      return;
    }
    const confirmation = await vscode.window.showWarningMessage(
      `Confirm that the running app has reloaded your code change. RAG Doctor will replay ${cases.length} labeled case(s) with their original request settings. Captured before/after answers will be shown locally in this report. This may call the model provider, use API quota, and replace the latest local capture. It will not edit source.`,
      { modal: true }, 'Replay current app', 'Cancel',
    );
    if (confirmation !== 'Replay current app') {
      panel.webview.postMessage({ type: 'suiteVerificationResult', text: 'Post-change verification canceled; no app requests were sent.' });
      return;
    }
    const endpoint = cases[0].replay;
    const runId = crypto.randomBytes(8).toString('hex');
    const replayed = [];
    try {
      const backend = await this.ensureBackend(this.getWorkspaceFolders().map((item) => item.path));
      for (let index = 0; index < cases.length; index += 1) {
        const item = cases[index];
        panel.webview.postMessage({
          type: 'suiteVerificationProgress',
          text: `Post-change verification: replaying case ${index + 1}/${cases.length} (${item.testCase.case_id}).`,
        });
        const body = JSON.parse(JSON.stringify(item.replay.requestBody));
        if (Object.prototype.hasOwnProperty.call(body, 'session_id')) {
          const caseId = String(item.testCase.case_id || 'case').replace(/[^\w-]/g, '_').slice(0, 80);
          body.session_id = `rag-doctor-post-change-${runId}-${caseId}`;
        }
        try {
          const reset = await this.request(backend.port, 'POST', '/api/captured-run-reset', { workspace_path: folder.path });
          if (reset.status < 200 || reset.status >= 300) throw new Error(reset.data && reset.data.detail || 'Could not reset local run capture.');
          const response = await this.request(endpoint.port, 'POST', endpoint.route, body, 120000);
          if (response.status < 200 || response.status >= 300) throw new Error(`App returned HTTP ${response.status}.`);
          const evaluated = await this.request(backend.port, 'POST', '/api/evaluate-captured-run', {
            workspace_path: folder.path,
            expected_answer: item.testCase.expected_answer,
            case_id: item.testCase.case_id,
            gold_document_id: item.testCase.gold_document_id,
            question_type: item.testCase.question_type,
            top_k_final: Number.isInteger(body.top_k) ? body.top_k : null,
          });
          if (evaluated.status < 200 || evaluated.status >= 300) throw new Error(evaluated.data && evaluated.data.detail || `Analyzer returned HTTP ${evaluated.status}.`);
          replayed.push({ testCase: item.testCase, diagnosis: evaluated.data.diagnosis, appResponse: response.data, replay: item.replay });
        } catch (error) {
          replayed.push({ testCase: item.testCase, error: error.message || String(error), replay: item.replay });
        }
      }
      const sourceFingerprint = await fingerprintWorkspace(folder.path);
      const before = snapshotEvaluation(cases, baselineFingerprint);
      const after = snapshotEvaluation(replayed, sourceFingerprint);
      const metrics = compareSnapshots(before, after);
      if (after.cases.length) this.previousSuiteSnapshots.set(folder.path, after);
      const trackedFixes = this.pendingFixVerifications.get(folder.path) || [];
      const trackedLines = trackedFixes.length
        ? ['Tracked recommendation source checks:']
        : [];
      for (const fix of trackedFixes) {
        const status = this._trackedFixStatus(folder.path, fix);
        const state = status.status === 'present'
          ? `exact replacement found at ${fix.file}:${status.line}`
          : status.status === 'not_found'
            ? `exact replacement not found in ${fix.file}; inspect for an equivalent edit`
            : `could not recheck ${fix.file}; confirm the source manually`;
        trackedLines.push(`- ${fix.caseId || 'tracked fix'}: ${state}`);
      }
      const afterById = new Map(replayed.map((item) => [String(item.testCase && item.testCase.case_id || ''), item]));
      const answerComparisons = cases.map((item) => {
        const afterItem = afterById.get(String(item.testCase.case_id || ''));
        if (!afterItem || !afterItem.diagnosis) return null;
        const beforeAnswer = this._extractAppAnswer(item.appResponse) || this._extractAppAnswer(item.diagnosis.llm_answer);
        const afterAnswer = this._extractAppAnswer(afterItem.appResponse) || this._extractAppAnswer(afterItem.diagnosis.llm_answer);
        const retrievalBefore = this._retrievalMetadata(item.diagnosis);
        const retrievalAfter = this._retrievalMetadata(afterItem.diagnosis);
        if (!beforeAnswer && !afterAnswer && !retrievalBefore.length && !retrievalAfter.length) return null;
        return {
          case_id: String(item.testCase.case_id || ''),
          before: beforeAnswer ? beforeAnswer.text : 'Answer text not captured',
          after: afterAnswer ? afterAnswer.text : 'Answer text not captured',
          changed: beforeAnswer && afterAnswer ? beforeAnswer.text !== afterAnswer.text : null,
          truncated: Boolean((beforeAnswer && beforeAnswer.truncated) || (afterAnswer && afterAnswer.truncated)),
          retrieval_before: retrievalBefore,
          retrieval_after: retrievalAfter,
          retrieval_changed: JSON.stringify(retrievalBefore) !== JSON.stringify(retrievalAfter),
        };
      }).filter(Boolean);
      panel.webview.postMessage({
        type: 'suiteVerificationResult',
        text: [this.formatSuiteVerification(metrics, cases, replayed), ...trackedLines].join('\n'),
        answerComparisons,
      });
    } catch (error) {
      panel.webview.postMessage({
        type: 'suiteVerificationResult',
        text: `Post-change verification failed: ${error.message || String(error)}. The app source was not changed.`,
      });
    }
  }

  _extractAppAnswer(response) {
    if (typeof response === 'string') response = { answer: response };
    if (!response || typeof response !== 'object') return null;
    const candidates = [response.answer, response.response, response.output, response.result, response.llm_answer];
    let answer = candidates.find((value) => typeof value === 'string');
    if (!answer && response.message && typeof response.message.content === 'string') answer = response.message.content;
    if (!answer && Array.isArray(response.choices) && response.choices[0]) {
      const message = response.choices[0].message;
      if (message && typeof message.content === 'string') answer = message.content;
      else if (typeof response.choices[0].text === 'string') answer = response.choices[0].text;
    }
    if (!answer && Array.isArray(response.candidates) && response.candidates[0]) {
      const parts = response.candidates[0].content && response.candidates[0].content.parts;
      if (Array.isArray(parts)) answer = parts.map((part) => part && typeof part.text === 'string' ? part.text : '').filter(Boolean).join('\n');
    }
    if (typeof answer !== 'string' || !answer.trim()) return null;
    const normalized = answer.replace(/\r\n/g, '\n').trim();
    const maximum = 4000;
    return { text: normalized.slice(0, maximum), truncated: normalized.length > maximum };
  }

  _retrievalMetadata(diagnosis) {
    const chunks = diagnosis && Array.isArray(diagnosis.retrieved_chunks) ? diagnosis.retrieved_chunks : [];
    return chunks.slice(0, 10).map((chunk, index) => {
      if (!chunk || typeof chunk !== 'object') return null;
      const metadata = {
        rank: Number.isInteger(chunk.final_rank) ? chunk.final_rank : Number.isInteger(chunk.rank) ? chunk.rank : index + 1,
      };
      for (const [outputKey, aliases] of Object.entries({
        doc_id: ['doc_id', 'source', 'file'],
        chunk_id: ['chunk_id', 'id'],
      })) {
        const value = aliases.map((key) => chunk[key]).find((item) => typeof item === 'string' && item.trim());
        if (value) metadata[outputKey] = value.trim().slice(0, 200);
      }
      for (const key of ['dense_score', 'bm25_score', 'rerank_score']) {
        if (typeof chunk[key] === 'number' && Number.isFinite(chunk[key])) metadata[key] = chunk[key];
      }
      return metadata;
    }).filter(Boolean);
  }

  formatSuiteVerification(metrics, beforeCases, afterCases) {
    const paired = Number(metrics.pairedCount) || 0;
    const ratio = (before, after) => paired ? `${before}/${paired} → ${after}/${paired}` : 'unavailable; no paired outcomes';
    const latency = (value) => Number.isFinite(value) ? `${value.toFixed(1)} ms` : 'unavailable';
    const sourceChange = {
      changed: 'Tracked source/config files changed since the baseline run.',
      unchanged: 'No tracked source/config change was detected since the baseline run.',
      partial: 'Source comparison is partial; some files may be outside scan limits.',
      unavailable: 'Source change status is unavailable.',
    }[metrics.sourceChange] || 'Source change status is unavailable.';
    const lines = [
      `Post-change replay: ${paired}/${beforeCases.length} labeled case(s) paired.`,
      `Reference-supported outcomes: ${ratio(metrics.beforeSupported, metrics.afterSupported)}.`,
      `Gold source-document presence: ${ratio(metrics.beforeGold, metrics.afterGold)}.`,
      `Mean app-reported latency: ${latency(metrics.beforeLatencyMs)} → ${latency(metrics.afterLatencyMs)}.`,
      sourceChange,
      'This is a repeatable before/after check on the same case IDs and request settings. It does not prove the code change alone caused the result; provider/model variance may affect answers.',
    ];
    if (paired && metrics.afterSupported > metrics.beforeSupported) lines.push('More cases were reference-supported in this replay. Review per-case evidence before keeping the change.');
    else if (paired && metrics.afterSupported < metrics.beforeSupported) lines.push('Fewer cases were reference-supported in this replay; review or revert the change before proceeding.');
    else if (paired) lines.push('No reference-supported outcome increase was measured in this replay.');
    const afterById = new Map((afterCases || []).map((item) => [String(item.testCase && item.testCase.case_id || ''), item]));
    for (const item of beforeCases || []) {
      const caseId = String(item.testCase && item.testCase.case_id || 'unknown');
      const updated = afterById.get(caseId);
      const before = item.diagnosis && (item.diagnosis.root_cause || item.diagnosis.status) || 'unavailable';
      const after = updated && updated.diagnosis
        ? updated.diagnosis.root_cause || updated.diagnosis.status
        : updated && updated.error ? `replay failed (${updated.error})` : 'unavailable';
      lines.push(`- ${caseId}: ${before} → ${after}`);
    }
    return lines.join('\n');
  }

  async verifyEvaluationSuite(panel, folder, results, index) {
    const selected = results[Number(index)];
    const experiment = selected && selected.investigation && selected.investigation.experiment;
    if (!experiment || experiment.parameter !== 'top_k' || !selected.replay) {
      panel.webview.postMessage({ type: 'verificationResult', index, text: 'No supported request-time top_k experiment is available for this case.' });
      return;
    }
    const baselineTopK = selected.replay.requestBody.top_k;
    if (!Number.isInteger(baselineTopK) || baselineTopK !== experiment.baseline || !Number.isInteger(experiment.candidate) || experiment.candidate < 1 || experiment.candidate > 100 || experiment.candidate === baselineTopK) {
      panel.webview.postMessage({ type: 'verificationResult', index, text: 'The proposed experiment does not match the request parameter captured for this run. Rerun the evaluation before testing it.' });
      return;
    }
    const cases = results.filter((item) => item.testCase && item.diagnosis && item.replay && Number.isInteger(item.replay.requestBody.top_k) && item.replay.requestBody.top_k === baselineTopK);
    if (!cases.length) {
      panel.webview.postMessage({ type: 'verificationResult', index, text: 'No replayable labeled cases use the same baseline top_k.' });
      return;
    }
    const confirmation = await vscode.window.showWarningMessage(
      `RAG Doctor will replay each of ${cases.length} labeled case(s) twice: once with top_k=${baselineTopK} and once with top_k=${experiment.candidate}. This makes ${cases.length * 2} app calls, may consume API quota, and replaces the latest local capture. It will not edit app source.`,
      { modal: true }, 'Run verification', 'Cancel',
    );
    if (confirmation !== 'Run verification') {
      panel.webview.postMessage({ type: 'verificationResult', index, text: 'Verification canceled; no variant runs were sent.' });
      return;
    }

    const reportIndex = Number(index);
    try {
      const backend = await this.ensureBackend(this.getWorkspaceFolders().map((item) => item.path));
      const runId = crypto.randomBytes(8).toString('hex');
      const variants = [];
      const replayVariant = async (item, topK, variantName) => {
        const body = JSON.parse(JSON.stringify(item.replay.requestBody));
        body.top_k = topK;
        if (Object.prototype.hasOwnProperty.call(body, 'session_id')) {
          const caseId = String(item.testCase.case_id || 'case').replace(/[^\w-]/g, '_').slice(0, 80);
          body.session_id = `rag-doctor-verify-${runId}-${variantName}-${caseId}`;
        }
        const reset = await this.request(backend.port, 'POST', '/api/captured-run-reset', { workspace_path: folder.path });
        if (reset.status < 200 || reset.status >= 300) {
          throw new Error(reset.data && reset.data.detail ? reset.data.detail : 'Could not reset the previous local run capture.');
        }
        const response = await this.request(item.replay.port, 'POST', item.replay.route, body, 120000);
        if (response.status < 200 || response.status >= 300) {
          throw new Error(`App returned HTTP ${response.status}.`);
        }
        const evaluated = await this.request(backend.port, 'POST', '/api/evaluate-captured-run', {
          workspace_path: folder.path,
          expected_answer: item.testCase.expected_answer,
          case_id: item.testCase.case_id,
          gold_document_id: item.testCase.gold_document_id,
          question_type: item.testCase.question_type,
          top_k_final: topK,
        });
        if (evaluated.status < 200 || evaluated.status >= 300) {
          throw new Error(evaluated.data && evaluated.data.detail ? evaluated.data.detail : `Analyzer returned HTTP ${evaluated.status}.`);
        }
        return { body, appResponse: response.data, diagnosis: evaluated.data.diagnosis };
      };
      for (let index = 0; index < cases.length; index += 1) {
        const item = cases[index];
        const order = index % 2 === 0
          ? [['baseline', baselineTopK], ['candidate', experiment.candidate]]
          : [['candidate', experiment.candidate], ['baseline', baselineTopK]];
        const outcomes = {};
        for (const [variantName, topK] of order) {
          try {
            outcomes[variantName] = await replayVariant(item, topK, variantName);
          } catch (error) {
            outcomes[variantName] = { error: error.message || 'request failed.' };
          }
        }
        const baseline = outcomes.baseline;
        const candidate = outcomes.candidate;
        variants.push({
          testCase: item.testCase,
          baseline: baseline && baseline.diagnosis,
          baselineAppResponse: baseline && baseline.appResponse,
          diagnosis: candidate && candidate.diagnosis,
          appResponse: candidate && candidate.appResponse,
          error: [
            baseline && baseline.error ? `Baseline replay failed: ${baseline.error}` : '',
            candidate && candidate.error ? `Candidate replay failed: ${candidate.error}` : '',
          ].filter(Boolean).join(' '),
        });
        panel.webview.postMessage({ type: 'verificationProgress', index: reportIndex, text: `Verification progress: ${index + 1}/${cases.length} case(s) replayed with baseline and candidate settings.` });
      }
      panel.webview.postMessage({
        type: 'verificationResult',
        index,
        text: this.formatExperimentResults(baselineTopK, experiment.candidate, variants),
      });
    } catch (error) {
      panel.webview.postMessage({ type: 'verificationResult', index, text: `Experiment failed: ${error.message || String(error)}. The original source and request defaults were not changed.` });
    }
  }

  formatExperimentResults(baselineTopK, candidateTopK, variants) {
    const completed = variants.filter((item) => item.baseline && item.diagnosis);
    const supported = (diagnosis) => diagnosis && (
      diagnosis.root_cause === 'NO_FAILURE' || diagnosis.status === 'reference_supported'
    );
    const baselineSupported = completed.filter((item) => supported(item.baseline)).length;
    const candidateSupported = completed.filter((item) => supported(item.diagnosis)).length;
    const baselineGold = completed.filter((item) => item.baseline && item.baseline.gold_document_retrieved === true).length;
    const candidateGold = completed.filter((item) => item.diagnosis.gold_document_retrieved === true).length;
    const latency = (item) => {
      const value = item && item.appResponse && item.appResponse.latency_ms;
      if (value === null || value === undefined || value === '') return null;
      const parsed = Number(value);
      return Number.isFinite(parsed) ? parsed : null;
    };
    const meanLatency = (items) => {
      const values = items.map(latency).filter((value) => value !== null);
      return values.length ? `${(values.reduce((sum, value) => sum + value, 0) / values.length).toFixed(1)} ms` : 'unavailable';
    };
    const lines = [
      `Local A/B result: top_k ${baselineTopK} → ${candidateTopK}`,
      `${completed.length}/${variants.length} cases produced paired baseline and candidate diagnoses.`,
      `Reference-supported answers (strict current evaluator): ${completed.length ? `${baselineSupported}/${completed.length} → ${candidateSupported}/${completed.length}` : 'unavailable; no paired diagnosis completed'}.`,
      `Labeled source document present: ${completed.length ? `${baselineGold}/${completed.length} → ${candidateGold}/${completed.length}` : 'unavailable; no paired diagnosis completed'}. Document presence alone does not prove the relevant passage was retrieved.`,
      `Mean app-reported latency: ${meanLatency(completed.map((item) => ({ appResponse: item.baselineAppResponse })))} → ${meanLatency(completed)}.`,
      'Baseline-first and candidate-first order alternates across cases to reduce cache-warmup bias; provider/model variance can still affect results.',
    ];
    if (candidateSupported > baselineSupported) lines.push('The candidate improved strict reference-supported outcomes on this local suite.');
    else if (candidateSupported < baselineSupported) lines.push('The candidate reduced strict reference-supported outcomes on this local suite.');
    else lines.push('No strict reference-supported answer improvement was measured on this local suite.');
    lines.push('This tests only request-time top_k on these cases. It does not change the app default or prove improvement on other workloads.');
    for (const item of variants) {
      const before = item.baseline ? `${item.baseline.root_cause || item.baseline.status}` : (item.error && item.error.includes('Baseline replay failed:') ? 'baseline failed' : 'unavailable');
      const after = item.diagnosis ? `${item.diagnosis.root_cause || item.diagnosis.status}` : (item.error && item.error.includes('Candidate replay failed:') ? 'candidate failed' : 'unavailable');
      lines.push(`- ${item.testCase.case_id}: ${before} → ${after}`);
    }
    return lines.join('\n');
  }

  async connectApp() {
    if (!vscode.workspace.isTrusted) {
      vscode.window.showWarningMessage('Trust this workspace before connecting an app. RAG Doctor will add a connector file only after your confirmation.');
      return;
    }
    const folders = this.getWorkspaceFolders();
    if (!folders.length) {
      vscode.window.showWarningMessage('Open the local RAG application folder before connecting it.');
      return;
    }

    let folder = folders[0];
    if (folders.length > 1) {
      const selected = await vscode.window.showQuickPick(folders.map((item) => ({ label: item.name, description: item.path, item })), {
        placeHolder: 'Choose the RAG application workspace to connect',
      });
      if (!selected) return;
      folder = selected.item;
    }

    const manifestText = [];
    for (const filename of ['requirements.txt', 'pyproject.toml', 'Pipfile', 'package.json']) {
      const manifestPath = path.join(folder.path, filename);
      try {
        const stat = await fs.promises.stat(manifestPath);
        if (stat.isFile() && stat.size <= 256 * 1024) manifestText.push(await fs.promises.readFile(manifestPath, 'utf8'));
      } catch { /* Manifest is optional. */ }
    }
    const manifest = manifestText.join('\n').toLowerCase();
    const knownFrameworks = [
      ['LangChain / LangGraph', /langchain|langgraph/],
      ['LlamaIndex', /llama[-_]index|llamaindex/],
      ['Haystack', /haystack[-_]ai|from haystack|haystack/],
      ['Semantic Kernel', /semantic[-_]kernel|semantic_kernel/],
    ].filter(([, pattern]) => pattern.test(manifest)).map(([name]) => name);
    const hasPython = ['requirements.txt', 'pyproject.toml', 'Pipfile'].some((name) => fs.existsSync(path.join(folder.path, name))) ||
      fs.existsSync(path.join(folder.path, 'main.py'));
    const hasJavaScript = fs.existsSync(path.join(folder.path, 'package.json')) ||
      fs.existsSync(path.join(folder.path, 'index.js')) || fs.existsSync(path.join(folder.path, 'index.ts'));
    let language;
    if (hasPython && hasJavaScript) {
      language = await vscode.window.showQuickPick([
        { label: 'Python', description: 'Install the standard-library Python connector' },
        { label: 'JavaScript / TypeScript', description: 'Install the standard-library Node.js connector' },
      ], { placeHolder: 'Choose the language used by the RAG pipeline' }).then((choice) => choice && choice.label);
      if (!language) return;
    } else if (hasPython) language = 'Python';
    else if (hasJavaScript) language = 'JavaScript / TypeScript';
    else {
      language = await vscode.window.showQuickPick(['Python', 'JavaScript / TypeScript'], {
        placeHolder: 'Choose the language used by the RAG pipeline',
      });
      if (!language) return;
    }

    const isPython = language === 'Python';
    const filename = isPython ? 'rag_doctor_probe.py' : 'rag_doctor_probe.js';
    const source = path.join(this.context.extensionPath, 'connectors', filename);
    const destination = path.join(folder.path, filename);
    let connector;
    try {
      connector = await fs.promises.readFile(source);
    } catch {
      vscode.window.showErrorMessage(`The bundled ${language} connector is missing from this RAG Doctor installation.`);
      return;
    }

    let existing = false;
    try {
      const stat = await fs.promises.lstat(destination);
      if (stat.isSymbolicLink() || !stat.isFile()) {
        vscode.window.showWarningMessage(`RAG Doctor will not replace ${filename}: the existing path is not a regular file.`);
        return;
      }
      existing = true;
      const oldContents = await fs.promises.readFile(destination);
      if (oldContents.equals(connector)) {
        try {
          await this.ensureBackend(folders.map((item) => item.path));
        } catch (error) {
          vscode.window.showErrorMessage(error.message || 'Could not start the local RAG Doctor analyzer.');
          return;
        }
        this.showConnectionGuide(folder, language, knownFrameworks);
        return;
      }
    } catch (error) {
      if (error.code !== 'ENOENT') {
        vscode.window.showErrorMessage(`Could not inspect ${filename}: ${error.message}`);
        return;
      }
    }

    const action = existing ? 'Replace connector' : 'Install connector';
    const consent = await vscode.window.showInformationMessage(
      `${action} ${filename} in ${folder.name} and start RAG Doctor's local analyzer? This adds only the connector file; it does not edit or run your application code.`,
      action,
      'Cancel',
    );
    if (consent !== action) return;

    try {
      await this.ensureBackend(folders.map((item) => item.path));
    } catch (error) {
      vscode.window.showErrorMessage(error.message || 'Could not start the local RAG Doctor analyzer.');
      return;
    }

    try {
      await fs.promises.writeFile(destination, connector, { flag: existing ? 'w' : 'wx' });
    } catch (error) {
      vscode.window.showErrorMessage(`Could not install the connector: ${error.message}`);
      return;
    }
    this.showConnectionGuide(folder, language, knownFrameworks);
  }

  showConnectionGuide(folder, language, frameworks) {
    const isPython = language === 'Python';
    const snippet = isPython
      ? `from pathlib import Path\nfrom rag_doctor_probe import RAGDoctorProbe\n\n# Call after retrieval and generation, where these values are available.\nRAGDoctorProbe().capture(\n    question=user_question,\n    prompt=assembled_prompt,\n    retrieved_chunks=[\n        {"text": doc.page_content, "final_rank": rank}\n        for rank, doc in enumerate(retrieved_documents, start=1)\n    ],\n    llm_answer=str(model_response),\n    workspace_path=str(Path.cwd()),\n    # expected_answer=gold_answer,  # Supply a labeled reference for accuracy checks.\n    include_content=True,  # Explicit full-content mode for a local accuracy-debugging run.\n)`
      : `const path = require("path");\nconst { captureRun } = require(path.join(process.cwd(), "rag_doctor_probe.js"));\n\n// Call after retrieval and generation, where these values are available.\nawait captureRun({\n  question: userQuestion,\n  prompt: assembledPrompt, // Omitted from capture by default.\n  retrievedChunks: documents.map((doc, index) => ({\n    text: doc.pageContent ?? doc.text ?? String(doc),\n    final_rank: index + 1,\n    score: doc.score,\n  })),\n  answer: modelResponse,\n  workspacePath: process.cwd(),\n  // expectedAnswer: goldAnswer, // Supply a labeled reference for accuracy checks.\n  includeContent: true, // Explicit full-content mode for a local accuracy-debugging run.\n});`;
    const frameworkText = frameworks.length ? frameworks.join(', ') : 'No supported framework dependency was identified';
    const escapeHtml = (value) => String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    const panel = vscode.window.createWebviewPanel('ragDoctorConnect', 'Connect RAG App', vscode.ViewColumn.Beside, { enableScripts: true });
    const nonce = crypto.randomBytes(16).toString('base64');
    panel.webview.html = `<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${panel.webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';"><meta name="viewport" content="width=device-width,initial-scale=1"><style>body{font:14px var(--vscode-font-family);color:var(--vscode-foreground);background:var(--vscode-editor-background);padding:24px;max-width:850px;margin:auto}h1{font-size:22px}.muted{color:var(--vscode-descriptionForeground)}.card{padding:14px;border:1px solid var(--vscode-panel-border);border-radius:8px;margin:16px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:var(--vscode-textCodeBlock-background);padding:14px;border-radius:6px}.button{background:var(--vscode-button-background);color:var(--vscode-button-foreground);border:0;border-radius:4px;padding:8px 12px;cursor:pointer}</style></head><body><h1>Connect your RAG app</h1><p class="muted">Workspace: ${escapeHtml(folder.name)} · Detected: ${escapeHtml(frameworkText)}</p><div class="card"><b>Local connector ready · capture not active yet</b><p>The helper is installed, but RAG Doctor cannot see another process's in-memory prompt, retrieval results, or answer until the app calls it. The example sends the question, assembled prompt, retrieved passages, and answer only to 127.0.0.1. Full-content mode is for authorized local debugging; the latest capture stays in analyzer memory and is not written to a run log.</p></div><h2>Finish the connection</h2><ol><li>Open the file where your app calls retrieval and generates the answer.</li><li>Adapt the variable names in this example to your app.</li><li>Call it after both the retrieved documents and final answer are available.</li><li>Run your app, submit one prompt, then scan the workspace in RAG Doctor.</li></ol><p class="muted">RAG Doctor installed <code>${escapeHtml(isPython ? 'rag_doctor_probe.py' : 'rag_doctor_probe.js')}</code> at the workspace root. It did not edit application source. Full-content mode is enabled in this example to support accuracy debugging. Use only data you are authorized to inspect; content stays in the local analyzer memory.</p><pre id="snippet">${escapeHtml(snippet)}</pre><button class="button" id="copy">Copy integration example</button><p id="status" class="muted"></p><script nonce="${nonce}">const vscode=acquireVsCodeApi();document.getElementById('copy').addEventListener('click',()=>vscode.postMessage({type:'copySnippet'}));window.addEventListener('message',e=>{if(e.data.type==='copied')document.getElementById('status').textContent='Example copied to clipboard.';if(e.data.type==='copyError')document.getElementById('status').textContent=e.data.message;});</script></body></html>`;
    panel.webview.onDidReceiveMessage(async (message) => {
      if (!message || message.type !== 'copySnippet') return;
      try {
        await vscode.env.clipboard.writeText(snippet);
        panel.webview.postMessage({ type: 'copied' });
      } catch {
        panel.webview.postMessage({ type: 'copyError', message: 'Could not access the clipboard. Select and copy the example manually.' });
      }
    }, undefined, this.context.subscriptions);
    this.context.subscriptions.push(panel);
  }

  async openLocation(message) {
    const folders = this.getWorkspaceFolders();
    const folder = folders.find((item) => item.path === message.workspace);
    if (!folder || typeof message.file !== 'string') return;
    const target = vscode.Uri.joinPath(folder.uri, ...message.file.split(/[\\/]/));
    try {
      const resolved = path.resolve(target.fsPath);
      const relative = path.relative(folder.path, resolved);
      if (relative.startsWith('..') || path.isAbsolute(relative)) return;
      const document = await vscode.workspace.openTextDocument(target);
      const line = Math.max(1, Number(message.line) || 1) - 1;
      await vscode.window.showTextDocument(document, { preview: true, selection: new vscode.Range(line, 0, line, 0) });
    } catch (error) {
      vscode.window.showWarningMessage(`Could not open the reported location: ${error.message}`);
    }
  }

  getHtml(webview) {
    const nonce = crypto.randomBytes(16).toString('base64');
    return `<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'unsafe-inline'; script-src 'nonce-${nonce}';">
<title>RAG Doctor</title>
<style>
:root{color-scheme:dark;--panel:var(--vscode-editorWidget-background);--border:var(--vscode-panel-border);--muted:var(--vscode-descriptionForeground);--accent:var(--vscode-textLink-foreground);--danger:var(--vscode-errorForeground);--good:var(--vscode-testing-iconPassed)}
*{box-sizing:border-box}body{font:13px var(--vscode-font-family);color:var(--vscode-foreground);background:var(--vscode-sideBar-background);padding:16px;margin:0}.top{display:flex;justify-content:space-between;align-items:center;gap:10px}h1{font-size:16px;margin:0}.sub{color:var(--muted);font-size:11px;margin:5px 0 14px}.local{border:1px solid var(--border);border-radius:12px;padding:10px;margin:12px 0;color:var(--good);font-size:11px}.button{background:var(--vscode-button-background);color:var(--vscode-button-foreground);border:0;border-radius:3px;padding:7px 10px;cursor:pointer}.button:hover{background:var(--vscode-button-hoverBackground)}.section{margin-top:12px;border:1px solid var(--border);border-radius:6px;background:var(--panel)}.section>summary{padding:10px;font-size:12px;font-weight:600;color:var(--vscode-foreground);cursor:pointer;list-style-position:inside}.section[open]>summary{border-bottom:1px solid var(--border);color:var(--accent)}.section>.card,.section>.empty,.section>.meta{margin:10px}.section>.loc{margin:10px}.card{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:12px;margin:8px 0}.cause{font-weight:700;font-size:14px}.meta{color:var(--muted);font-size:11px}.text{white-space:pre-wrap;overflow-wrap:anywhere;margin:8px 0}.fix{border-left:3px solid var(--accent);padding:8px;margin-top:8px}.evidence{color:var(--muted);font-size:12px}.evidence li{margin:4px 0}.loc{background:none;border:0;color:var(--accent);padding:0;cursor:pointer;text-align:left}.tag{font-size:10px;border:1px solid var(--border);border-radius:9px;padding:2px 6px;color:var(--muted)}.empty,.error{padding:12px;border-radius:6px;background:var(--panel);color:var(--muted)}.error{color:var(--danger)}details{margin-top:7px}summary{cursor:pointer;color:var(--accent)}code{overflow-wrap:anywhere;color:var(--vscode-textPreformat-foreground)}
</style></head><body>
<div class="top"><h1>RAG Doctor</h1><div><button class="button" id="run">Run with Doctor</button> <button class="button" id="scan">Scan</button></div></div>
<div class="sub">Local workspace scan · opt-in runtime evaluation</div>
<div class="local">● Workspace discovery runs locally. Runtime prompt, response, and passage capture starts only after you choose Run with Doctor and approve the evaluation.</div>
<main id="content"><div class="empty">Workspace scan runs automatically when this view opens. Choose Run with Doctor to start the app and replay discovered evaluation cases.</div></main>
<script nonce="${nonce}">
const vscode=acquireVsCodeApi();const content=document.getElementById('content');
document.getElementById('scan').addEventListener('click',()=>vscode.postMessage({type:'scan'}));
document.getElementById('run').addEventListener('click',()=>vscode.postMessage({type:'runWithDoctor'}));
window.addEventListener('message',event=>{const m=event.data;if(m.type==='loading'){content.replaceChildren(el('div','empty','Scanning the current workspace locally…'));return}if(m.type==='error'){content.replaceChildren(el('div','error',m.message));return}if(m.type==='result'){render(m.results||[]);return}if(m.type==='latestInvestigationProgress'||m.type==='latestInvestigationResult'){const prefix=m.workspaceAudit?'workspace-ai-output-':'latest-ai-output-';const selector=m.workspaceAudit?'[data-workspace-ai="'+m.index+'"]':'[data-latest-ai="'+m.index+'"]';const output=document.getElementById(prefix+m.index);const button=document.querySelector(selector);if(output){output.hidden=false;output.textContent=m.text||'No investigation result.'}if(button&&m.type==='latestInvestigationResult')button.disabled=false}});
function el(tag,cls,text){const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n}
function locButton(result,file,line,label){const b=el('button','loc',label||(file+':'+line));b.addEventListener('click',()=>vscode.postMessage({type:'openLocation',workspace:result.path,file,line}));return b}
function card(title){const n=el('article','card');if(title)n.appendChild(el('div','cause',title));return n}
function section(title){const s=el('details','section');s.appendChild(el('summary','',title));return s}
function render(results){
  content.replaceChildren();
  if(!results.length){content.appendChild(el('div','empty','No workspace folders were returned.'));return}
  for(let resultIndex=0;resultIndex<results.length;resultIndex++){
    const result=results[resultIndex];
    const data=result.data||{};const cs=data.code_scan||{};const ls=data.log_scan||{};
    content.appendChild(el('div','meta','Workspace: '+result.name+' · memory only'));
    const workspaceAi=card('AI workspace audit');
    workspaceAi.appendChild(el('div','text','Ask the configured loopback model to trace bounded local code, configuration, logs, and the latest captured run if available. Captured prompt/response/passages are included only when present in that run. With no run, the audit is code-only and cannot determine answer quality.'));
    const workspaceAiButton=el('button','button','Review RAG code with local AI');workspaceAiButton.dataset.workspaceAi=String(resultIndex);workspaceAiButton.addEventListener('click',()=>{workspaceAiButton.disabled=true;const output=document.getElementById('workspace-ai-output-'+resultIndex);output.hidden=false;output.textContent='Reviewing workspace code with the configured local model…';vscode.postMessage({type:'investigateWorkspace',workspacePath:result.path,index:resultIndex})});workspaceAi.appendChild(workspaceAiButton);
    const workspaceAiOutput=el('pre','text');workspaceAiOutput.id='workspace-ai-output-'+resultIndex;workspaceAiOutput.hidden=true;workspaceAi.appendChild(workspaceAiOutput);content.appendChild(workspaceAi);
    const diagnosis=data.diagnosis;const dsec=section('Latest RAG run');
    if(!diagnosis){dsec.appendChild(el('div','empty',data.message||'No prompt/response run was found. Add a supported JSON/JSONL log entry or the probe hook.'))}
    else{
      const d=card(diagnosis.root_cause||'INSUFFICIENT_EVIDENCE');
      d.appendChild(el('div','tag',(diagnosis.status||'diagnosis')+' · '+(diagnosis.evidence_level||'limited')+' evidence · '+(diagnosis.source||'unknown source')));
      if(diagnosis.prompt_response_excluded)d.appendChild(el('div','meta','Privacy mode: prompt and response excluded.'));
      if(diagnosis.retrieved_text_included)d.appendChild(el('div','meta','Retrieved passage text was included in this local, memory-only diagnosis.'));
      else if(diagnosis.content_excluded)d.appendChild(el('div','meta','Retrieved passage text was excluded; only retrieval metadata is available.'));
      d.appendChild(el('p','text',diagnosis.explanation||''));
      if(diagnosis.question){d.appendChild(el('div','meta','Input question'));d.appendChild(el('div','text',diagnosis.question))}
      if(diagnosis.prompt){const det=el('details');det.appendChild(el('summary','','Captured prompt'));det.appendChild(el('pre','text',diagnosis.prompt));d.appendChild(det)}
      if(diagnosis.llm_answer){d.appendChild(el('div','meta','Model response'));d.appendChild(el('div','text',diagnosis.llm_answer))}
      if(diagnosis.recommended_fix){const fix=el('div','fix');fix.appendChild(el('b','','Suggested change'));fix.appendChild(el('div','text',diagnosis.recommended_fix));d.appendChild(fix)}
      if(diagnosis.source_location&&diagnosis.source_location.file)d.appendChild(locButton(result,diagnosis.source_location.file,diagnosis.source_location.line,'Open related code location'));
      if(diagnosis.source_location_note)d.appendChild(el('div','meta',diagnosis.source_location_note));
      if(diagnosis.log_location)d.appendChild(locButton(result,diagnosis.log_location.file,diagnosis.log_location.line,'Open source log event'));
      if(diagnosis.retrieval_observed!==null&&diagnosis.retrieval_observed!==undefined)d.appendChild(el('div','meta','Retrieval observed: '+(diagnosis.retrieval_observed?'yes':'no')+(diagnosis.retrieval_count!==undefined?' · candidates: '+diagnosis.retrieval_count:'')));
      if(diagnosis.metadata&&Object.keys(diagnosis.metadata).length){const det=el('details');det.appendChild(el('summary','','Runtime trace metadata'));const pre=el('pre','text',JSON.stringify(diagnosis.metadata,null,2));det.appendChild(pre);d.appendChild(det)}
      if(diagnosis.evidence&&diagnosis.evidence.length){const ul=el('ul','evidence');diagnosis.evidence.forEach(x=>ul.appendChild(el('li','',x)));d.appendChild(ul)}
      if(Array.isArray(diagnosis.retrieved_chunks)&&diagnosis.retrieved_chunks.length){const det=el('details');det.appendChild(el('summary','','Retrieved context'));diagnosis.retrieved_chunks.forEach((c,i)=>det.appendChild(el('pre','text','#'+(c.final_rank||i+1)+' '+(c.text||''))));d.appendChild(det)}
      if(diagnosis.data_transfer_events&&diagnosis.data_transfer_events.length){const det=el('details');det.appendChild(el('summary','','Captured local data-transfer events'));diagnosis.data_transfer_events.forEach(item=>det.appendChild(el('pre','text',JSON.stringify(item,null,2))));d.appendChild(det)}
      const aiButton=el('button','button','Investigate latest run with local AI');aiButton.dataset.latestAi=String(resultIndex);aiButton.addEventListener('click',()=>{aiButton.disabled=true;const output=document.getElementById('latest-ai-output-'+resultIndex);output.hidden=false;output.textContent='Investigating with the configured local model…';vscode.postMessage({type:'investigateLatestRun',workspacePath:result.path,index:resultIndex})});d.appendChild(aiButton);
      const aiOutput=el('pre','text');aiOutput.id='latest-ai-output-'+resultIndex;aiOutput.hidden=true;d.appendChild(aiOutput);
      dsec.appendChild(d)
    }
    content.appendChild(dsec);
    const warnings=cs.static_warnings||[];const wsec=section('Code findings ('+warnings.length+')');
    if(!warnings.length)wsec.appendChild(el('div','empty','No supported chunking or top-k issues found in scanned code.'));
    warnings.forEach(w=>{const c=card(w.root_cause||'Code finding');c.appendChild(el('div','text',w.problem||''));c.appendChild(locButton(result,w.file,w.line,w.file+':'+w.line));c.appendChild(el('div','fix',w.fix_suggestion||''));wsec.appendChild(c)});content.appendChild(wsec);
    const pipeline=cs.pipeline_locations||[];const psec=section('RAG code locations ('+pipeline.length+')');
    if(!pipeline.length)psec.appendChild(el('div','empty','No supported RAG pipeline patterns were found in Python/JavaScript/TypeScript files.'));
    pipeline.slice(0,80).forEach(item=>{const c=card(item.component);c.appendChild(el('pre','text',item.code_snippet||''));c.appendChild(locButton(result,item.file,item.line,item.file+':'+item.line));psec.appendChild(c)});content.appendChild(psec);
    const tests=cs.evaluation_cases||[];const testSec=section('Evaluation test cases ('+(cs.evaluation_case_count||tests.length)+')');
    if(!tests.length)testSec.appendChild(el('div','empty','No labeled cases found. Run with Doctor can still capture a manual app run; automated replay and answer scoring need JSON/JSONL records with question/query and gold_answer/expected_answer fields.'));
    tests.forEach(test=>{const c=card(test.case_id+' · '+(test.question_type||'test case'));c.appendChild(el('div','text',test.question));c.appendChild(el('div','meta','Expected: '+test.expected_answer));if(test.gold_document_id)c.appendChild(el('div','meta','Gold document: '+test.gold_document_id));c.appendChild(el('div','meta',test.file));testSec.appendChild(c)});content.appendChild(testSec);
    const datasets=cs.dataset_summary||[];const dataSec=section('Local dataset structure ('+datasets.length+' files)');
    if(!datasets.length)dataSec.appendChild(el('div','empty','No supported text, Markdown, CSV, or JSON dataset files found within scan limits.'));
    datasets.forEach(item=>{const c=card(item.format||'dataset');c.appendChild(el('div','text',item.file));c.appendChild(el('div','meta',item.num_paragraphs!==undefined?item.num_paragraphs+' paragraphs · average '+item.avg_paragraph_chars+' chars':item.num_rows_sampled!==undefined?item.num_rows_sampled+' sampled rows · '+item.column_count+' columns':(item.record_count||0)+' records · '+(item.top_level_keys||item.columns||[]).join(', ')));c.appendChild(locButton(result,item.file,1,'Open dataset'));dataSec.appendChild(c)});content.appendChild(dataSec);
    const xfers=cs.potential_data_transfer_locations||[];const xsec=section('Potential data-transfer code ('+xfers.length+')');
    xsec.appendChild(el('div','meta','Static matches only. These code paths may transfer data when executed; this is not proof a transfer occurred.'));
    if(!xfers.length)xsec.appendChild(el('div','empty','No supported network-call patterns found.'));
    xfers.slice(0,30).forEach(x=>{const c=card('Potential network call');c.appendChild(el('pre','text',x.code_snippet||''));c.appendChild(locButton(result,x.file,x.line,x.file+':'+x.line));xsec.appendChild(c)});content.appendChild(xsec);
    const lsTitle='Local logs ('+(ls.events_found||0)+' RAG-like events)';const logsec=section(lsTitle);
    logsec.appendChild(el('div','meta',(ls.files_scanned||[]).length+' bounded log files read · '+(ls.bytes_read||0)+' bytes · RAM only'));
    if(ls.latest_event)logsec.appendChild(locButton(result,ls.latest_event.file,ls.latest_event.line,'Latest event: '+ls.latest_event.file+':'+ls.latest_event.line));
    else logsec.appendChild(el('div','empty','No supported prompt/response record found in .log, .jsonl, or RAG/trace JSON files.'));
    content.appendChild(logsec)
  }
}
</script></body></html>`;
  }

  dispose() {
    if (this.ownsBackend && this.backendProcess && this.backendProcess.exitCode === null) {
      this.backendProcess.kill();
    }
    this.pendingFixVerifications.clear();
    this.previousSuiteSnapshots.clear();
    this.latestDiagnoses.clear();
    this.output.dispose();
  }
}

function activate(context) {
  const provider = new RagDoctorViewProvider(context);
  context.subscriptions.push(vscode.window.registerWebviewViewProvider('ragDoctor.dashboard', provider, { webviewOptions: { retainContextWhenHidden: true } }));
  context.subscriptions.push(vscode.commands.registerCommand('ragDoctor.openStudio', async () => {
    const panel = vscode.window.createWebviewPanel('ragDoctorStudio', 'RAG Doctor — Local Diagnosis', vscode.ViewColumn.Beside, { enableScripts: true, retainContextWhenHidden: true });
    provider.attach(panel.webview);
    await provider.scan(panel.webview);
  }));
  context.subscriptions.push(vscode.commands.registerCommand('ragDoctor.scanWorkspace', async () => {
    await vscode.commands.executeCommand('workbench.view.extension.ragDoctor');
    await vscode.commands.executeCommand('ragDoctor.dashboard.focus');
  }));
  context.subscriptions.push(vscode.commands.registerCommand('ragDoctor.connectApp', async () => {
    await provider.connectApp();
  }));
  context.subscriptions.push(vscode.commands.registerCommand('ragDoctor.runWithDoctor', async () => {
    await provider.runWithDoctor();
  }));
  context.subscriptions.push(vscode.commands.registerCommand('ragDoctor.showOutput', () => provider.output.show(true)));
  context.subscriptions.push({ dispose: () => provider.dispose() });
}

function deactivate() {}
module.exports = { activate, deactivate };
