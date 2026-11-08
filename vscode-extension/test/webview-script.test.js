const test = require('node:test');
const assert = require('node:assert/strict');
const Module = require('node:module');
const vm = require('node:vm');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

test('dashboard and extension wire static and latest-run AI investigations locally', async () => {
  let provider;
  const createdTerminals = [];
  const informationMessages = [];
  const executedCommands = [];
  const createdPanels = [];
  const vscodeStub = {
    ViewColumn: { Beside: 2 },
    window: {
      createOutputChannel: () => ({ appendLine() {}, dispose() {} }),
      createWebviewPanel: () => {
        const panel = {
          disposed: false,
          onDidDispose(handler) { this.disposeHandler = handler; },
          webview: {
            html: '',
            onDidReceiveMessage(handler) {
              this.messageHandler = handler;
              return { dispose: () => { panel.disposed = true; this.messageHandler = null; } };
            },
          },
        };
        createdPanels.push(panel);
        return panel;
      },
      createTerminal: (options) => {
        const terminal = { options, shown: false, command: null, show() { this.shown = true; }, sendText(command) { this.command = command; } };
        createdTerminals.push(terminal);
        return terminal;
      },
      showInputBox: async () => 'python app.py',
      showInformationMessage: async (...args) => { informationMessages.push(args); return 'Open RAG Doctor'; },
      registerWebviewViewProvider: (_id, value) => {
        provider = value;
        return { dispose() {} };
      },
      showWarningMessage: async () => 'Run verification',
    },
    commands: {
      registerCommand: () => ({ dispose() {} }),
      executeCommand: async (...args) => { executedCommands.push(args); },
    },
    workspace: { isTrusted: false, workspaceFolders: [], getConfiguration: () => ({ get: (_key, fallback) => fallback }) },
  };
  const originalLoad = Module._load;
  Module._load = function loadWithVscodeStub(request, parent, isMain) {
    if (request === 'vscode') return vscodeStub;
    return originalLoad.call(this, request, parent, isMain);
  };
  let html;
  const extensionSubscriptions = [];
  try {
    const extension = require('../extension');
    extension.activate({ extensionPath: __dirname, subscriptions: extensionSubscriptions });
    html = provider.getHtml({ cspSource: 'vscode-webview://test' });
  } finally {
    Module._load = originalLoad;
  }

  const match = html.match(/<script nonce="[^"]+">([\s\S]*?)<\/script>/);
  assert.ok(match, 'dashboard script should be present');
  new vm.Script(match[1]);
  assert.match(match[1], /investigateWorkspace/);
  assert.match(match[1], /investigateLatestRun/);
  assert.match(html, /Review RAG code with local AI/);
  assert.match(html, /Investigate latest run with local AI/);
  const fixText = provider.formatInvestigation({
    status: 'probable',
    root_cause: 'RETRIEVAL',
    summary: 'Candidate count is zero.',
    evidence: [{ verified: true, source: 'code:src/retriever.py:1', quote: 'top_k = 0' }],
    fix: {
      file: 'src/retriever.py',
      line: 1,
      end_line: 1,
      source_matches: true,
      syntax_status: 'valid',
      current_code: 'top_k = 0',
      replacement_code: 'top_k = 5',
      proposed_change: 'Use a positive candidate count.',
    },
    verification: 'Replay the same local cases.',
  });
  assert.match(fixText, /src\/retriever\.py:1/);
  assert.match(fixText, /Replace with:\ntop_k = 5/);
  assert.match(fixText, /Python syntax check passed/);
  assert.match(fixText, /Replay labeled cases against current app/);
  const runtimeFixText = provider.formatInvestigation({
    status: 'probable',
    root_cause: 'RANKING_K',
    summary: 'Captured scores run opposite to the source sort.',
    static_verifier: { checked: false, finding_detected: false, fallback_applied: false },
    runtime_verifier: {
      checked: true,
      finding_detected: true,
      finding_root_cause: 'RANKING_K',
      finding_file: 'src/retriever.py',
      fallback_applied: true,
    },
    model_confirmed_deterministic_check: true,
    model_assessment_before_verification: {
      root_cause: 'RANKING_K',
      status: 'probable',
      summary: 'The model identified the score-order mismatch.',
    },
    evidence: [],
    fix: null,
    verification: 'Replay the same query and suite.',
  });
  assert.match(runtimeFixText, /Hybrid investigation: PROBABLE · RANKING_K/);
  assert.match(runtimeFixText, /Runtime verifier: RANKING_K in src\/retriever\.py; the model confirmed the cause/);
  assert.match(runtimeFixText, /Model assessment before verification: RANKING_K · probable/);
  assert.match(runtimeFixText, /The model confirmed the cause, but its proposed edit did not match the source-verified patch/);
  const staticAuditTraceText = provider.formatInvestigation({
    status: 'insufficient_evidence',
    root_cause: 'UNKNOWN',
    summary: 'No concrete issue could be established.',
    static_verifier: {
      checked: true,
      finding_detected: false,
      trace: {
        python_files_scanned: 3,
        python_files_parsed: 2,
        chunk_step_candidate_count: 1,
        dropped_context_return_count: 0,
        inventory_error: null,
      },
    },
    evidence: [],
    fix: null,
    verification: 'Capture a run with retrieval and generation evidence.',
  });
  assert.match(staticAuditTraceText, /Static audit trace: Python files parsed 2\/3/);
  assert.match(staticAuditTraceText, /chunk-step candidates 1/);
  assert.match(staticAuditTraceText, /prompt builders dropping context 0/);
  const agentTraceText = provider.formatInvestigation({
    status: 'probable',
    root_cause: 'GENERATOR',
    summary: 'The prompt builder omits retrieved context.',
    agent_trace: {
      model_turns: 3,
      thinking_enabled: true,
      read_only_tool_calls: 2,
      verifier_review_retry_used: true,
      tools_used: ['list_files', 'read_file'],
      source_files_provided: ['src/prompts.ts'],
      source_files_opened: ['src/pipeline.ts'],
    },
    evidence: [],
    fix: null,
    verification: 'Assert that the generation prompt contains the retrieved context.',
  });
  assert.match(agentTraceText, /Agent inspection: 3 local model turn\(s\); 2 additional read-only tool call\(s\) \(list_files, read_file\); reasoning enabled; one focused review checked the verifier finding/);
  assert.match(agentTraceText, /Source excerpts supplied to the investigator: src\/prompts\.ts; additional files opened: src\/pipeline\.ts/);
  const thinkingFallbackText = provider.formatInvestigation({
    status: 'insufficient_evidence',
    root_cause: 'UNKNOWN',
    summary: 'No concrete issue was established.',
    agent_trace: { model_turns: 1, thinking_enabled: true, thinking_fallback_used: true, read_only_tool_calls: 0 },
    evidence: [], fix: null,
    verification: 'Capture a local run.',
  });
  assert.match(thinkingFallbackText, /retried without reasoning after the model returned no final answer/);

  vscodeStub.workspace.isTrusted = true;
  const folder = { name: 'rag-project', path: 'C:\\rag-project', uri: { fsPath: 'C:\\rag-project' } };
  provider.getWorkspaceFolders = () => [folder];
  provider.ensureBackend = async () => ({ port: 8765 });
  const posted = [];
  const requested = [];
  provider.request = async (_port, _method, route, payload) => {
    if (route.startsWith('/api/analyze-workspace')) {
      return { status: 200, data: { diagnosis: null, code_scan: {
        static_warnings: [{ file: 'src/retriever.py', line: 1, root_cause: 'RETRIEVAL' }],
        pipeline_locations: [{ file: 'src/generation.py', line: 20, component: 'generation' }],
      } } };
    }
    requested.push({ route, payload });
    return { status: 200, data: { investigation: {
      status: 'probable', root_cause: 'RETRIEVAL', summary: 'Candidate count needs review.',
      evidence: [], fix: null, verification: 'Replay the same local cases.',
    } } };
  };
  provider.getSettings = () => ({
    localModelName: 'test-local-model',
    localModelEndpoint: 'http://127.0.0.1:11434/api/chat',
    localModelThinking: true,
  });
  const webview = { postMessage: (message) => posted.push(message) };
  await provider.scan(webview);
  await provider.investigateLatestRun(webview, folder.path, 0, true);
  assert.equal(requested[0].route, '/api/agentic-investigate');
  assert.equal(requested[0].payload.local_model_thinking, true);
  assert.deepEqual(requested[0].payload.diagnosis.workspace_code_references.map((item) => item.file), [
    'src/retriever.py', 'src/generation.py',
  ]);
  assert.equal(posted.at(-1).type, 'latestInvestigationResult');
  assert.equal(posted.at(-1).workspaceAudit, true);

  const verificationRequests = [];
  provider.request = async (port, method, route, payload) => {
    verificationRequests.push({ port, method, route, payload });
    if (route === '/api/captured-run-reset') return { status: 200, data: { status: 'reset' } };
    if (route === '/api/query') return { status: 200, data: { answer: 'local answer', latency_ms: payload.top_k * 10 } };
    if (route === '/api/evaluate-captured-run') {
      return {
        status: 200,
        data: { diagnosis: payload.top_k_final === 10
          ? { root_cause: 'NO_FAILURE', status: 'reference_supported', gold_document_retrieved: true }
          : { root_cause: 'RETRIEVAL', status: 'diagnosed', gold_document_retrieved: false } },
      };
    }
    throw new Error(`Unexpected verification route: ${route}`);
  };
  const verificationCases = ['case-1', 'case-2'].map((caseId) => ({
    testCase: { case_id: caseId, question: `question ${caseId}`, expected_answer: 'answer', gold_document_id: 'doc-1' },
    diagnosis: { root_cause: 'RETRIEVAL' },
    replay: { port: 8000, route: '/api/query', requestBody: { query: `question ${caseId}`, session_id: `old-${caseId}`, top_k: 5 } },
    investigation: caseId === 'case-1' ? { experiment: { parameter: 'top_k', baseline: 5, candidate: 10 } } : undefined,
  }));
  await provider.verifyEvaluationSuite({ webview }, folder, verificationCases, 0);
  const appCalls = verificationRequests.filter((item) => item.route === '/api/query');
  assert.deepEqual(appCalls.map((item) => item.payload.top_k), [5, 10, 10, 5], 'variant order should alternate across cases');
  assert.equal(new Set(appCalls.map((item) => item.payload.session_id)).size, 4, 'every variant should use an isolated session');
  assert.ok(appCalls.every((item) => item.payload.session_id.startsWith('rag-doctor-verify-')));
  const evaluationCalls = verificationRequests.filter((item) => item.route === '/api/evaluate-captured-run');
  assert.deepEqual(evaluationCalls.map((item) => item.payload.top_k_final), [5, 10, 10, 5]);
  const verificationText = posted.at(-1).text;
  assert.match(verificationText, /2\/2 cases produced paired baseline and candidate diagnoses/);
  assert.match(verificationText, /Reference-supported answers.*0\/2 → 2\/2/);
  assert.match(verificationText, /order alternates across cases to reduce cache-warmup bias/);

  const syntaxWorkspace = fs.mkdtempSync(path.join(os.tmpdir(), 'rag-doctor-js-check-'));
  try {
    fs.writeFileSync(path.join(syntaxWorkspace, 'retriever.py'), 'top_k = 0\n');
    const postChangeFolder = { name: 'verification-workspace', path: syntaxWorkspace };
    const postChangeCases = [
      {
        testCase: { case_id: 'case-a', question: 'Question A', expected_answer: 'Answer A', gold_document_id: 'doc-a' },
        diagnosis: {
          root_cause: 'RETRIEVAL', status: 'needs_review', gold_document_retrieved: false, llm_answer: 'baseline answer A',
          provider_call: { status: 'failed', provider: 'google', model: 'gemini-2.5-flash', error: 'http_404' },
          retrieved_chunks: [{ doc_id: 'old-doc', chunk_id: 'old-chunk', final_rank: 3, dense_score: 0.4, text: 'not copied into comparison' }],
        },
        appResponse: { answer: 'baseline answer A', latency_ms: 20 },
        replay: { port: 8000, route: '/api/query', requestBody: { query: 'Question A', session_id: 'old-a', top_k: 5 } },
        investigation: { fix: {
          file: 'retriever.py', line: 1, end_line: 1, current_code: 'top_k = 0', replacement_code: 'top_k = 5',
          source_matches: true, proposed_change: 'Set a positive candidate limit.',
        } },
      },
      {
        testCase: { case_id: 'case-b', question: 'Question B', expected_answer: 'Answer B', gold_document_id: 'doc-b' },
        diagnosis: { root_cause: 'GENERATOR', status: 'needs_review', gold_document_retrieved: false },
        appResponse: { answer: 'baseline answer B', latency_ms: 30 },
        replay: { port: 8000, route: '/api/query', requestBody: { query: 'Question B', session_id: 'old-b', top_k: 5 } },
      },
      {
        testCase: { case_id: 'case-c', question: 'Question C', expected_answer: 'Answer C', gold_document_id: 'doc-c' },
        diagnosis: {
          root_cause: 'RETRIEVAL', status: 'needs_review', gold_document_retrieved: false,
          retrieved_chunks: [{ doc_id: 'metadata-doc-before', chunk_id: 'metadata-chunk-before', final_rank: 2 }],
        },
        appResponse: { latency_ms: 25 },
        replay: { port: 8000, route: '/api/query', requestBody: { query: 'Question C', session_id: 'old-c', top_k: 5 } },
      },
    ];
    const postChangeRequests = [];
    provider.getWorkspaceFolders = () => [postChangeFolder];
    provider.request = async (port, method, route, payload) => {
      postChangeRequests.push({ port, method, route, payload });
      if (route === '/api/captured-run-reset') return { status: 200, data: { status: 'reset' } };
      if (route === '/api/query') return payload.query === 'Question A' || payload.query === 'Question C'
        ? { status: 200, data: { latency_ms: 12 } }
        : { status: 200, data: { answer: `answer for ${payload.query}`, latency_ms: 12 } };
      if (route === '/api/evaluate-captured-run') return {
        status: 200,
        data: { diagnosis: {
          root_cause: 'NO_FAILURE', status: 'reference_supported', gold_document_retrieved: true,
          llm_answer: payload.case_id === 'case-c' ? undefined : `captured answer for ${payload.case_id}`,
          retrieved_chunks: payload.case_id === 'case-a'
            ? [{ doc_id: 'new-doc', chunk_id: 'new-chunk', final_rank: 1, dense_score: 0.95, text: 'must stay out of comparison' }]
            : payload.case_id === 'case-c'
              ? [{ doc_id: 'metadata-doc-after', chunk_id: 'metadata-chunk-after', final_rank: 1 }]
              : [],
        } },
      };
      throw new Error(`Unexpected post-change verification route: ${route}`);
    };
    vscodeStub.window.showWarningMessage = async () => 'Replay current app';
    const verificationMessages = [];
    const trackingMessages = [];
    provider.trackFixForVerification({ webview: { postMessage: (message) => trackingMessages.push(message) } }, postChangeFolder, postChangeCases, 0);
    assert.equal(trackingMessages.at(-1).tracked, true);
    fs.writeFileSync(path.join(syntaxWorkspace, 'retriever.py'), 'top_k = 5\n');
    fs.writeFileSync(path.join(syntaxWorkspace, 'reranker.py'), 'candidate_limit = 4\n');
    const alreadyAppliedCase = {
      testCase: { case_id: 'case-c' },
      investigation: { fix: {
        file: 'reranker.py', line: 1, end_line: 1, current_code: 'candidate_limit = 4', replacement_code: 'candidate_limit = 8',
        source_matches: true, proposed_change: 'Increase the candidate limit.',
      } },
    };
    fs.writeFileSync(path.join(syntaxWorkspace, 'reranker.py'), 'candidate_limit = 8\n');
    provider.trackFixForVerification({ webview: { postMessage: (message) => trackingMessages.push(message) } }, postChangeFolder, [alreadyAppliedCase], 0);
    assert.equal(trackingMessages.at(-1).tracked, true, 'tracking should accept the exact replacement when it was applied first');
    await provider.verifyCurrentSuite(
      { webview: { postMessage: (message) => verificationMessages.push(message) } },
      postChangeFolder,
      postChangeCases,
      { digest: 'baseline-digest', complete: true, files: 1, totalBytes: 9 },
    );
    const replayCalls = postChangeRequests.filter((item) => item.route === '/api/query');
    assert.equal(replayCalls.length, 3);
    assert.ok(replayCalls.every((item) => item.payload.session_id.startsWith('rag-doctor-post-change-')));
    assert.equal(new Set(replayCalls.map((item) => item.payload.session_id)).size, 3);
    assert.ok(postChangeRequests.filter((item) => item.route === '/api/evaluate-captured-run').every((item) => item.payload.expected_answer));
    const suiteVerification = verificationMessages.find((message) => message.type === 'suiteVerificationResult');
    assert.match(suiteVerification.text, /3\/3 labeled case\(s\) paired/);
    assert.match(suiteVerification.text, /Reference-supported outcomes \(requires answer and context evidence\): 0\/3 → 3\/3/);
    assert.match(suiteVerification.text, /Tracked source\/config files changed/);
    assert.match(suiteVerification.text, /exact replacement found at retriever\.py:1/);
    assert.match(suiteVerification.text, /exact replacement found at reranker\.py:1/);
    assert.equal(suiteVerification.answerComparisons.length, 3);
    assert.deepEqual(suiteVerification.answerComparisons.map((pair) => pair.before), ['baseline answer A', 'baseline answer B', 'Answer text not captured']);
    assert.deepEqual(suiteVerification.answerComparisons.map((pair) => pair.after), ['captured answer for case-a', 'answer for Question B', 'Answer text not captured']);
    assert.deepEqual(suiteVerification.answerComparisons.map((pair) => pair.changed), [true, true, null]);
    assert.equal(suiteVerification.answerComparisons[0].retrieval_changed, true);
    assert.deepEqual(suiteVerification.answerComparisons[0].retrieval_before, [{ rank: 3, doc_id: 'old-doc', chunk_id: 'old-chunk', dense_score: 0.4 }]);
    assert.deepEqual(suiteVerification.answerComparisons[0].retrieval_after, [{ rank: 1, doc_id: 'new-doc', chunk_id: 'new-chunk', dense_score: 0.95 }]);
    assert.equal(suiteVerification.answerComparisons[2].retrieval_changed, true);
    assert.deepEqual(suiteVerification.answerComparisons[2].retrieval_after, [{ rank: 1, doc_id: 'metadata-doc-after', chunk_id: 'metadata-chunk-after' }]);
    assert.equal(JSON.stringify(suiteVerification.answerComparisons).includes('not copied into comparison'), false);
    assert.equal(JSON.stringify(suiteVerification.answerComparisons).includes('must stay out of comparison'), false);
    assert.deepEqual(provider._extractAppAnswer({ answer: `${'x'.repeat(4001)}` }), { text: 'x'.repeat(4000), truncated: true });
    assert.deepEqual(provider._extractAppAnswer('captured provider output'), { text: 'captured provider output', truncated: false });
    assert.deepEqual(provider._extractAppAnswer({ choices: [{ message: { content: 'OpenAI-style output' } }] }), { text: 'OpenAI-style output', truncated: false });
    provider.showEvaluationReport(postChangeFolder, postChangeCases, {
      firstRun: true,
      baselineSaved: true,
      sourceFingerprint: { digest: 'baseline-digest', complete: true, files: 1, totalBytes: 9 },
    });
    const evaluationPanel = createdPanels.at(-1);
    const reportHtml = evaluationPanel.webview.html;
    assert.match(reportHtml, /Replay labeled cases against current app/);
    assert.match(reportHtml, /Track this exact fix/);
    assert.match(reportHtml, /Model provider call/);
    assert.match(reportHtml, /gemini-2\.5-flash/);
    assert.match(reportHtml, /HTTP 200 from the app does not prove/);
    assert.match(reportHtml, /verifyCurrentSuite/);
    const reportScript = reportHtml.match(/<script nonce="[^"]+">([\s\S]*?)<\/script>/);
    assert.ok(reportScript);
    new vm.Script(reportScript[1]);
    const answerContainer = {
      hidden: true,
      children: [],
      replaceChildren() { this.children = []; },
      appendChild(child) { this.children.push(child); },
      get childElementCount() { return this.children.length; },
    };
    const suiteOutput = { hidden: true, textContent: '' };
    const suiteButton = { disabled: true, textContent: '' };
    let webviewMessageHandler;
    const fakeDocument = {
      addEventListener() {},
      getElementById(id) {
        return id === 'suite-verification-output' ? suiteOutput
          : id === 'suite-verification-answers' ? answerContainer : null;
      },
      querySelector(selector) { return selector === '[data-verify-suite]' ? suiteButton : null; },
      createElement(tag) {
        return {
          tagName: tag,
          children: [],
          appendChild(child) { this.children.push(child); },
          textContent: '',
        };
      },
    };
    vm.runInNewContext(reportScript[1], {
      acquireVsCodeApi: () => ({ postMessage() {} }),
      document: fakeDocument,
      window: { addEventListener(_event, handler) { webviewMessageHandler = handler; } },
    });
    webviewMessageHandler({ data: {
      type: 'suiteVerificationResult',
      text: 'replay complete',
      answerComparisons: [{
        case_id: 'case-safe', before: '<img src=x onerror=alert(1)>', after: 'updated answer', changed: true, truncated: false,
        retrieval_changed: true,
        retrieval_before: [{ rank: 3, doc_id: 'before-doc' }],
        retrieval_after: [{ rank: 1, doc_id: 'after-doc' }],
      }],
    } });
    assert.equal(answerContainer.hidden, false);
    assert.equal(answerContainer.children.length, 1);
    assert.equal(answerContainer.children[0].children[2].textContent, '<img src=x onerror=alert(1)>', 'answer text must be rendered as text, not HTML');
    assert.equal(answerContainer.children[0].children[4].textContent, 'updated answer');
    assert.match(answerContainer.children[0].children[0].textContent, /retrieved set changed/);
    assert.equal(answerContainer.children[0].children[6].textContent, JSON.stringify([{ rank: 3, doc_id: 'before-doc' }], null, 2));
    const reportListener = evaluationPanel.webview.messageHandler;
    assert.equal(typeof reportListener, 'function');
    assert.equal(extensionSubscriptions.includes(reportListener), false, 'the result-bearing listener must not be registered for the extension lifetime');
    evaluationPanel.disposeHandler();
    assert.equal(evaluationPanel.disposed, true, 'closing the evaluation report should dispose its run-data listener');
    assert.equal(evaluationPanel.webview.messageHandler, null, 'closing the report should release the listener closure');

    postChangeRequests.length = 0;
    vscodeStub.window.showWarningMessage = async () => 'Cancel';
    const canceledMessages = [];
    await provider.verifyCurrentSuite(
      { webview: { postMessage: (message) => canceledMessages.push(message) } },
      postChangeFolder,
      postChangeCases,
      { digest: 'baseline-digest', complete: true, files: 1, totalBytes: 9 },
    );
    assert.equal(postChangeRequests.length, 0, 'canceling replay must not call the app or analyzer');
    assert.match(canceledMessages.at(-1).text, /canceled; no app requests were sent/);

    vscodeStub.window.showWarningMessage = async () => 'Replay current app';
    provider.request = async (_port, _method, route) => {
      postChangeRequests.push({ route });
      if (route === '/api/captured-run-reset') return { status: 200, data: {} };
      if (route === '/api/query') return { status: 200, data: { answer: 'uncaptured answer', latency_ms: 12 } };
      if (route === '/api/evaluate-captured-run') return { status: 404, data: { detail: 'No runtime capture was received for this workspace.' } };
      throw new Error(`Unexpected missing-capture route: ${route}`);
    };
    const missingCaptureMessages = [];
    await provider.verifyCurrentSuite(
      { webview: { postMessage: (message) => missingCaptureMessages.push(message) } },
      postChangeFolder,
      postChangeCases,
      { digest: 'baseline-digest', complete: true, files: 1, totalBytes: 9 },
    );
    const missingCaptureResult = missingCaptureMessages.find((message) => message.type === 'suiteVerificationResult');
    assert.match(missingCaptureResult.text, /0\/3 labeled case\(s\) paired/);
    assert.match(missingCaptureResult.text, /replay failed/);
    assert.match(missingCaptureResult.text, /no paired outcomes/);

    const commonJsSource = 'function buildPrompt(question, context) { return context; }';
    fs.writeFileSync(path.join(syntaxWorkspace, 'prompt.cjs'), `${commonJsSource}\n`);
    const validFix = { fix: {
      file: 'prompt.cjs', line: 1, end_line: 1,
      current_code: commonJsSource,
      replacement_code: 'function buildPrompt(question, context) { return `${context} ${question}`; }',
      source_matches: true,
    } };
    provider.validateProposedFix(syntaxWorkspace, validFix);
    assert.equal(validFix.fix.syntax_status, 'valid');
    assert.equal(validFix.fix.syntax_language, 'JavaScript');

    const invalidFix = { fix: {
      file: 'prompt.cjs', line: 1, end_line: 1,
      current_code: commonJsSource,
      replacement_code: 'function buildPrompt(question, context) { return ( }',
      source_matches: true,
    } };
    provider.validateProposedFix(syntaxWorkspace, invalidFix);
    assert.equal(invalidFix.fix.syntax_status, 'invalid');
    assert.equal(invalidFix.fix.replacement_code, '', 'invalid copy-ready source must be suppressed');

    fs.writeFileSync(path.join(syntaxWorkspace, 'prompt.cjs'), 'function buildPrompt(question, context) { return question; }\n');
    const staleFix = { fix: {
      file: 'prompt.cjs', line: 1, end_line: 1,
      current_code: commonJsSource,
      replacement_code: 'function buildPrompt(question, context) { return `${context} ${question}`; }',
      source_matches: true,
    } };
    provider.validateProposedFix(syntaxWorkspace, staleFix);
    assert.equal(staleFix.fix.syntax_status, 'source_changed');
    assert.equal(staleFix.fix.source_matches, false);
    assert.equal(staleFix.fix.replacement_code, '', 'stale source suggestions must be suppressed');
    assert.match(provider.formatInvestigation({
      status: 'probable', root_cause: 'GENERATOR', summary: 'Prompt field mismatch.',
      evidence: [], fix: staleFix.fix, verification: 'Investigate the updated source again.',
    }), /Workspace source changed since analysis/);

    fs.writeFileSync(path.join(syntaxWorkspace, 'redacted.py'), 'def build():\n    token = "local-secret"\n');
    const redactedFix = { fix: {
      file: 'redacted.py', line: 2, end_line: 2,
      current_code: '    token = "[REDACTED_SECRET]"',
      replacement_code: '    return "safe"',
      source_matches: true,
    } };
    provider.validateProposedFix(syntaxWorkspace, redactedFix);
    assert.equal(redactedFix.fix.syntax_status, 'source_unverifiable');
    assert.equal(redactedFix.fix.source_matches, false);
    assert.equal(redactedFix.fix.replacement_code, '', 'redacted source must not keep a copy-ready replacement');
    assert.match(provider.formatInvestigation({
      status: 'probable', root_cause: 'GENERATOR', summary: 'Prompt field mismatch.',
      evidence: [], fix: redactedFix.fix, verification: 'Inspect the current source again.',
    }), /Workspace source could not be rechecked/);

    fs.writeFileSync(path.join(syntaxWorkspace, 'large.py'), 'x'.repeat(1_000_001));
    const oversizedFix = { fix: {
      file: 'large.py', line: 1, end_line: 1,
      current_code: 'x', replacement_code: 'y', source_matches: true,
    } };
    provider.validateProposedFix(syntaxWorkspace, oversizedFix);
    assert.equal(oversizedFix.fix.syntax_status, 'source_unverifiable');
    assert.equal(oversizedFix.fix.source_matches, false);
    assert.equal(oversizedFix.fix.replacement_code, '', 'unreadable source must not keep a copy-ready replacement');

    const moduleWorkspace = path.join(syntaxWorkspace, 'esm');
    fs.mkdirSync(moduleWorkspace);
    fs.writeFileSync(path.join(moduleWorkspace, 'package.json'), JSON.stringify({ type: 'module' }));
    fs.writeFileSync(path.join(moduleWorkspace, 'prompt.js'), 'export const answer = 42;\n');
    const moduleFix = { fix: {
      file: 'prompt.js', line: 1, end_line: 1,
      current_code: 'export const answer = 42;',
      replacement_code: 'export const answer = ;',
      source_matches: true,
    } };
    provider.validateProposedFix(moduleWorkspace, moduleFix);
    assert.equal(moduleFix.fix.syntax_status, undefined, 'ES module syntax remains honestly unchecked');
    assert.equal(moduleFix.fix.replacement_code, 'export const answer = ;');

    const pythonSource = 'def answer(question):\n    return question\n';
    fs.writeFileSync(path.join(syntaxWorkspace, 'generator.py'), pythonSource);
    const stalePythonFix = { fix: {
      file: 'generator.py', line: 2, end_line: 2,
      current_code: '    return question',
      replacement_code: '    return "fixed"',
      source_matches: true,
      syntax_status: 'valid',
    } };
    fs.writeFileSync(path.join(syntaxWorkspace, 'generator.py'), 'def answer(question):\n    return "changed"\n');
    provider.validateProposedFix(syntaxWorkspace, stalePythonFix);
    assert.equal(stalePythonFix.fix.syntax_status, 'source_changed');
    assert.equal(stalePythonFix.fix.source_matches, false);
    assert.equal(stalePythonFix.fix.replacement_code, '');

    provider.context.extensionPath = path.resolve(__dirname, '..');
    provider.getWorkspaceFolders = () => [folder];
    provider.getSettings = () => ({ port: 8765 });
    provider.ensureBackend = async () => ({ port: 8765 });
    provider._candidatePorts = () => [8000];
    provider._findFastApi = async () => null;
    provider._waitForRuntimeAgent = async () => ({ status: 'ready' });
    provider._waitForFastApi = async () => { throw new Error('manual capture should not require FastAPI replay discovery'); };
    const manualRunRequests = [];
    provider.request = async (_port, _method, route, payload) => {
      manualRunRequests.push({ route, payload });
      if (route.startsWith('/api/analyze-workspace')) {
        return { status: 200, data: { code_scan: { evaluation_cases: [] } } };
      }
      return { status: 200, data: { status: 'reset' } };
    };
    vscodeStub.window.showWarningMessage = async () => 'Start app';
    await provider.runWithDoctor();
    assert.equal(createdTerminals.length, 1, 'an app without labeled test cases still launches');
    assert.equal(createdTerminals[0].command, 'python app.py');
    assert.equal(createdTerminals[0].options.env.RAG_DOCTOR_CAPTURE, '1');
    assert.equal(createdTerminals[0].options.env.RAG_DOCTOR_WORKSPACE, folder.path);
    assert.match(createdTerminals[0].options.env.NODE_OPTIONS, /--require=".*agent\/node\/rag-doctor-preload\.cjs"/);
    assert.match(informationMessages.at(-1)[0], /Python and Node\.js observation/);
    assert.ok(informationMessages.at(-1)[0].includes('Submit a prompt in the app'));
    assert.match(informationMessages.at(-1)[0], /click Scan workspace/i);
    assert.deepEqual(executedCommands.at(-1), ['ragDoctor.openStudio']);
    assert.ok(manualRunRequests.some((item) => item.route === '/api/runtime-agent-reset'));
    assert.ok(!manualRunRequests.some((item) => item.route === '/api/query'), 'manual mode does not invent or replay test questions');
  } finally {
    fs.rmSync(syntaxWorkspace, { recursive: true, force: true });
  }
});
