const test = require('node:test');
const assert = require('node:assert/strict');
const { EventEmitter } = require('node:events');
const fs = require('node:fs');
const path = require('node:path');
const { PassThrough } = require('node:stream');
const vm = require('node:vm');

const observerPath = path.resolve(__dirname, '..', 'agent', 'node', 'rag-doctor-preload.cjs');
const observer = require(observerPath);

test('Node observer recognizes supported provider endpoints and response formats', () => {
  assert.equal(observer.providerFor('https://generativelanguage.googleapis.com/v1/models/gemini:generateContent'), 'google');
  assert.equal(observer.providerFor('https://api.openai.com/v1/responses'), 'openai');
  assert.equal(observer.providerFor('https://api.anthropic.com/v1/messages'), 'anthropic');
  assert.equal(observer.providerFor('http://127.0.0.1:11434/api/chat'), 'ollama');
  assert.equal(observer.providerFor('https://example.com/api/chat'), null);
  assert.equal(observer.answerFromData({ output: [{ content: [{ type: 'output_text', text: 'Found it.' }] }] }, 'openai'), 'Found it.');
  assert.equal(observer.streamAnswer('event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"Found it."}\n\n', 'openai'), 'Found it.');
  assert.equal(observer.promptFromBody({ input: 'What is revenue?' }), 'USER: What is revenue?');
  const genericContext = observer.contextBlocks('Context: The filing reports revenue of 42 million for the year.\nQuestion: What was revenue?');
  assert.equal(genericContext.length, 1);
  assert.match(genericContext[0].text, /42 million/);
  assert.doesNotMatch(genericContext[0].text, /What was revenue/);
  assert.equal(genericContext[0].extraction_method, 'prompt_context_heading');
  assert.equal('rank' in genericContext[0], false, 'heuristic prompt parsing must not invent retrieval ranks');
});

test('Node observer does not patch transports unless Run with Doctor opted in', () => {
  const originalRequest = function request() {};
  const originalFetch = async function fetch() {};
  const fakeHttp = { request: originalRequest };
  const fakeRequire = (name) => {
    if (name === 'node:http') return fakeHttp;
    if (name === 'node:https') return { request: originalRequest };
    if (name === 'node:module') return { syncBuiltinESMExports() {} };
    throw new Error(`Unexpected require: ${name}`);
  };
  fakeRequire.main = {};
  const module = { exports: {} };
  const sandbox = {
    module, exports: module.exports, require: fakeRequire, fetch: originalFetch,
    URL, Buffer, Symbol, ArrayBuffer, AbortSignal, Response,
    process: { env: { RAG_DOCTOR_CAPTURE: '0' }, pid: 789, cwd: () => 'C:/workspace', hrtime: { bigint: () => process.hrtime.bigint() } },
  };
  vm.runInNewContext(fs.readFileSync(observerPath, 'utf8'), sandbox, { filename: observerPath });
  assert.equal(fakeHttp.request, originalRequest);
  assert.equal(sandbox.fetch, originalFetch);
});

test('Node preload captures a supported fetch without changing its response', async () => {
  const calls = [];
  const localPayloads = [];
  const modelPayload = { message: { content: 'Revenue was $42 million.' } };
  const nativeFetch = async (input, init = {}) => {
    const url = String(input);
    if (url.includes('/api/chat')) return new Response(JSON.stringify(modelPayload), { status: 200, headers: { 'content-type': 'application/json' } });
    const payload = JSON.parse(init.body || '{}');
    calls.push(url);
    localPayloads.push({ url, payload });
    return new Response('{"status":"ready"}', { status: 200, headers: { 'content-type': 'application/json' } });
  };
  const fakeRequire = () => ({});
  fakeRequire.main = {};
  const module = { exports: {} };
  const sandbox = {
    module,
    exports: module.exports,
    require: fakeRequire,
    fetch: nativeFetch,
    URL,
    Buffer,
    Symbol,
    ArrayBuffer,
    AbortSignal,
    Response,
    process: { env: {
      RAG_DOCTOR_CAPTURE: '1',
      RAG_DOCTOR_PORT: '8765',
      RAG_DOCTOR_WORKSPACE: 'C:/workspace',
    }, pid: 123, cwd: () => 'C:/workspace', hrtime: { bigint: () => process.hrtime.bigint() } },
  };
  vm.runInNewContext(fs.readFileSync(observerPath, 'utf8'), sandbox, { filename: observerPath });
  const prompt = [
    '[Source #1 | Chunk ID: c-1 | Filing: report-2025]',
    'Revenue was $42 million.',
    '--- USER QUESTION ---',
    'What was revenue?',
  ].join('\n');
  const response = await sandbox.fetch('http://127.0.0.1:11434/api/chat', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model: 'test-model', messages: [{ role: 'user', content: prompt }] }),
  });
  const responseText = await response.text();
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.deepEqual(JSON.parse(responseText), modelPayload, 'the application receives its normal response body');
  const ready = localPayloads.find((item) => item.url.endsWith('/api/runtime-agent-ready'))?.payload;
  const run = localPayloads.find((item) => item.url.endsWith('/api/capture-live-run'))?.payload;
  assert.equal(ready.observer_installed, true);
  assert.equal(ready.providers.includes('ollama'), true);
  assert.equal(run.question, 'What was revenue?');
  assert.equal(run.llm_answer, 'Revenue was $42 million.');
  assert.equal(run.retrieved_chunks.length, 1);
  assert.equal(run.retrieved_chunks[0].chunk_id, 'c-1');
  assert.equal(run.retrieved_chunks[0].final_rank, 1);
  assert.equal(run.metadata.provider, 'ollama');
  assert.equal(run.include_content, true);
  assert.deepEqual(calls, [
    'http://127.0.0.1:8765/api/runtime-agent-ready',
    'http://127.0.0.1:8765/api/capture-live-run',
  ]);
});

test('Node preload observes core HTTP SDK transport without consuming its response', async () => {
  const localPayloads = [];
  let capturedResolve;
  const captured = new Promise((resolve) => { capturedResolve = resolve; });
  const modelResponse = JSON.stringify({ choices: [{ message: { content: 'The answer is grounded.' } }] });
  const fakeHttp = {
    request(options) {
      const request = new EventEmitter();
      let requestBody = '';
      request.write = (chunk, encoding) => { requestBody += Buffer.isBuffer(chunk) ? chunk.toString() : String(chunk); return true; };
      request.end = (chunk) => {
        if (chunk !== undefined) requestBody += Buffer.isBuffer(chunk) ? chunk.toString() : String(chunk);
        const isLocal = options.hostname === '127.0.0.1';
        if (isLocal) {
          const localPayload = JSON.parse(requestBody || '{}');
          localPayloads.push({ url: `http://127.0.0.1:${options.port}${options.path}`, payload: localPayload });
          if (options.path === '/api/capture-live-run') capturedResolve(localPayload);
        }
        setImmediate(() => {
          const response = new PassThrough();
          response.statusCode = 200;
          response.headers = {};
          request.emit('response', response);
          response.end(isLocal ? '{"status":"ready"}' : modelResponse);
        });
        return request;
      };
      request.sentBody = () => requestBody;
      request.requestOptions = options;
      return request;
    },
  };
  const fakeHttps = { request: () => { throw new Error('not expected in this test'); } };
  const fakeRequire = (name) => {
    if (name === 'node:http') return fakeHttp;
    if (name === 'node:https') return fakeHttps;
    if (name === 'node:module') return { syncBuiltinESMExports() {} };
    if (name === 'node:zlib') return require('node:zlib');
    throw new Error(`Unexpected require: ${name}`);
  };
  fakeRequire.main = {};
  const module = { exports: {} };
  const sandbox = {
    module, exports: module.exports, require: fakeRequire,
    URL, Buffer, Symbol, ArrayBuffer, AbortSignal, Response,
    process: { env: { RAG_DOCTOR_CAPTURE: '1', RAG_DOCTOR_PORT: '8765', RAG_DOCTOR_WORKSPACE: 'C:/workspace' },
      pid: 456, cwd: () => 'C:/workspace', hrtime: { bigint: () => process.hrtime.bigint() } },
  };
  vm.runInNewContext(fs.readFileSync(observerPath, 'utf8'), sandbox, { filename: observerPath });

  const requestBody = JSON.stringify({ model: 'gpt-test', messages: [{ role: 'user', content: 'Context: The filing reports revenue was 42 million in the year.\nQuestion: How much?' }] });
  const request = fakeHttp.request({ hostname: 'api.openai.com', path: '/v1/chat/completions', method: 'POST' });
  const appResponse = new Promise((resolve) => {
    request.once('response', (response) => {
      let text = '';
      response.setEncoding('utf8');
      response.on('data', (part) => { text += part; });
      response.on('end', () => resolve(text));
    });
  });
  request.write(requestBody);
  request.end();
  assert.equal(await appResponse, modelResponse);
  const run = await Promise.race([captured, new Promise((_, reject) => setTimeout(() => reject(new Error('HTTP capture was not reported')), 1_000))]);
  assert.equal(run.question, 'Context: The filing reports revenue was 42 million in the year.\nQuestion: How much?');
  assert.equal(run.llm_answer, 'The answer is grounded.');
  assert.equal(run.prompt_context_blocks.length, 1);
  assert.match(run.prompt_context_blocks[0].text, /42 million/);
  assert.equal(run.retrieved_chunks.length, 0, 'prompt-derived context must not be mislabeled as retriever output');
  assert.equal(run.metadata.provider, 'openai');
  assert.deepEqual(localPayloads.map((item) => new URL(item.url).host), ['127.0.0.1:8765', '127.0.0.1:8765']);
});
