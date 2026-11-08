/* Opt-in, process-local observer loaded by Run with Doctor through NODE_OPTIONS. */
'use strict';

const MAX_CAPTURE_BYTES = 1_000_000;
const INSTALL_GUARD = Symbol.for('rag-doctor.fetch-observer-installed');
const nativeFetch = typeof globalThis.fetch === 'function' ? globalThis.fetch.bind(globalThis) : null;
let nativeLoopbackRequest = null;

function providerFor(url) {
  let parsed;
  try { parsed = new URL(String(url)); } catch { return null; }
  const host = parsed.hostname.toLowerCase();
  const path = parsed.pathname.toLowerCase();
  let provider;
  if (host === 'generativelanguage.googleapis.com' || (host === 'aiplatform.googleapis.com' && path.includes('/publishers/google/'))) provider = 'google';
  else if (host === 'api.openai.com' || host.endsWith('.openai.azure.com')) provider = 'openai';
  else if (host === 'api.anthropic.com') provider = 'anthropic';
  else if (['127.0.0.1', 'localhost', '::1', '[::1]'].includes(host) && (parsed.port === '11434' || path.includes('/api/generate') || path.includes('/api/chat'))) provider = 'ollama';
  if (!provider || !['generatecontent', 'chat/completions', '/responses', '/messages', '/api/generate', '/api/chat'].some((part) => path.includes(part))) return null;
  return provider;
}

function asText(value) {
  if (typeof value === 'string') return value;
  if (Array.isArray(value)) return value.map(asText).filter(Boolean).join('\n');
  if (value && typeof value === 'object') {
    if (typeof value.text === 'string') return value.text;
    if (typeof value.content === 'string' || Array.isArray(value.content)) return asText(value.content);
  }
  return '';
}

async function requestBody(input, init) {
  let raw = init && init.body;
  if (raw == null && input && typeof input.clone === 'function' && typeof input.url === 'string') {
    try { raw = await input.clone().text(); } catch { return {}; }
  }
  if (typeof raw === 'string') {
    try { const parsed = JSON.parse(raw); return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {}; } catch { return {}; }
  }
  if (Buffer.isBuffer(raw) || raw instanceof Uint8Array) {
    try { const parsed = JSON.parse(Buffer.from(raw).toString('utf8')); return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {}; } catch { return {}; }
  }
  return raw && typeof raw === 'object' && !ArrayBuffer.isView(raw) ? raw : {};
}

function promptFromBody(body) {
  if (typeof body.prompt === 'string') return body.prompt;
  const blocks = [];
  if (typeof body.input === 'string') blocks.push(`USER: ${body.input}`);
  else if (Array.isArray(body.input)) {
    for (const item of body.input) {
      if (typeof item === 'string') blocks.push(`USER: ${item}`);
      else if (item && typeof item === 'object') {
        const content = asText(item.content ?? item.text ?? item.input);
        if (content) blocks.push(`${String(item.role || item.type || 'message').toUpperCase()}: ${content}`);
      }
    }
  }
  for (const message of Array.isArray(body.messages) ? body.messages : []) {
    if (!message || typeof message !== 'object') continue;
    const content = asText(message.content ?? message.parts);
    if (content) blocks.push(`${String(message.role || 'message').toUpperCase()}: ${content}`);
  }
  for (const message of Array.isArray(body.contents) ? body.contents : []) {
    if (!message || typeof message !== 'object') continue;
    const parts = Array.isArray(message.parts) ? message.parts.map((part) => part && typeof part.text === 'string' ? part.text : '').filter(Boolean).join('\n') : asText(message.parts);
    if (parts) blocks.push(`${String(message.role || 'user').toUpperCase()}: ${parts}`);
  }
  const system = asText(body.system ?? body.systemInstruction);
  if (system) blocks.unshift(`SYSTEM: ${system}`);
  return blocks.join('\n\n');
}

function questionFromBody(body, prompt) {
  const labeled = prompt.match(/--- USER QUESTION ---\s*(.*?)\s*(?:--- ANALYST RESPONSE|$)/is);
  if (labeled) return labeled[1].trim().slice(0, 10_000);
  for (const key of ['question', 'query', 'input']) if (typeof body[key] === 'string') return body[key].slice(0, 10_000);
  for (const key of ['messages', 'contents']) {
    const entries = Array.isArray(body[key]) ? body[key] : [];
    for (const message of entries.slice().reverse()) {
      if (!message || typeof message !== 'object' || !['user', 'human'].includes(String(message.role || 'user').toLowerCase())) continue;
      const text = asText(message.content ?? message.parts);
      if (text) return text.slice(0, 10_000);
    }
  }
  return '';
}

function financeChunks(prompt) {
  const marker = /\[Source #(?<rank>\d+)\s*\|\s*Chunk ID:\s*(?<chunk>[^|\]]+)\|\s*Filing:\s*(?<doc>[^\]]+)\]\s*\n/ig;
  const matches = [...prompt.matchAll(marker)];
  return matches.map((match, index) => ({
    text: prompt.slice(match.index + match[0].length, index + 1 < matches.length ? matches[index + 1].index : prompt.length)
      .split(/\n\s*--- USER QUESTION ---/i, 1)[0].trim(),
    chunk_id: match.groups.chunk.trim(),
    doc_id: match.groups.doc.trim(),
    final_rank: Number(match.groups.rank),
  }));
}

function contextBlocks(prompt) {
  const sourceMarkers = [...prompt.matchAll(/^\s*(?:\[(?:source|document|passage)\s*#?\s*\d+[^\]]*\]|(?:source|document|passage)\s*#?\s*\d+\s*[:\-])[^\n]*$/gim)];
  if (sourceMarkers.length) {
    return sourceMarkers.slice(0, 10).map((match, index) => {
      const end = index + 1 < sourceMarkers.length ? sourceMarkers[index + 1].index : prompt.length;
      const text = prompt.slice(match.index + match[0].length, end).split(/^\s*(?:question|query|answer|response)\s*[:\-].*$/im, 1)[0].trim();
      return text.length >= 20 ? { text: text.slice(0, 10_000), label: match[0].trim().slice(0, 160), prompt_position: index + 1, extraction_method: 'prompt_source_marker', confidence: 'high' } : null;
    }).filter(Boolean);
  }
  const heading = /^\s*(?:#{1,6}[ \t]*)?(?:(?:system|developer|user|assistant|human)[ \t]*:[ \t]*)?(?<label>context(?: information| documents| passages| chunks)?|(?:relevant )?documents|(?:relevant )?passages|use the following (?:retrieved )?(?:context|documents|passages|chunks))\b[ \t]*:?[ \t]*(?<inline>[^\r\n]*)$/gim;
  const matches = [...prompt.matchAll(heading)];
  return matches.slice(0, 5).map((match, index) => {
    const end = index + 1 < matches.length ? matches[index + 1].index : prompt.length;
    const inline = String(match.groups && match.groups.inline || '').trim();
    const following = prompt.slice(match.index + match[0].length, end);
    const text = [inline, following].filter(Boolean).join('\n')
      .split(/^\s*(?:question|query|answer|response|user)\s*[:\-].*$/im, 1)[0]
      .replace(/^\s*[-_=]{3,}\s*$/gm, '').trim();
    return text.length >= 20 ? { text: text.slice(0, 10_000), label: String(match.groups && match.groups.label || 'Context section').trim(), prompt_position: index + 1, extraction_method: 'prompt_context_heading', confidence: 'moderate' } : null;
  }).filter(Boolean);
}

function answerFromData(data, provider) {
  try {
    if (provider === 'google') return (data.candidates[0].content.parts || []).map((part) => asText(part)).filter(Boolean).join('\n');
    if (provider === 'openai') {
      if (Array.isArray(data.choices)) return asText(data.choices[0].message?.content ?? data.choices[0].text ?? '');
      if (Array.isArray(data.output)) return data.output.flatMap((item) => item.content || []).map((item) => asText(item.text ?? item)).filter(Boolean).join('\n');
      return '';
    }
    if (provider === 'anthropic') return asText(data.content || []);
    if (provider === 'ollama') return asText(data.response ?? data.message?.content ?? '');
  } catch { return ''; }
  return '';
}

function streamAnswer(text, provider) {
  const fragments = [];
  const frames = provider === 'ollama' ? text.split(/\r?\n/) : text.split(/\r?\n\s*\r?\n/);
  for (const frame of frames) {
    const raw = provider === 'ollama' ? frame : frame.split(/\r?\n/).filter((line) => line.startsWith('data:')).map((line) => line.slice(5).trimStart()).join('\n');
    if (!raw || raw === '[DONE]') continue;
    let data;
    try { data = JSON.parse(raw); } catch { continue; }
    let fragment = '';
    try {
      if (provider === 'google') fragment = (data.candidates?.[0]?.content?.parts || []).map((part) => asText(part)).join('');
      else if (provider === 'openai') {
        if (Array.isArray(data.choices)) fragment = asText(data.choices[0]?.delta?.content ?? data.choices[0]?.message?.content ?? data.choices[0]?.text ?? '');
        else if (data.type === 'response.output_text.delta') fragment = asText(data.delta ?? '');
        else if (data.type === 'response.completed' && fragments.length === 0) fragment = answerFromData(data.response || {}, provider);
      }
      else if (provider === 'anthropic') fragment = data.type === 'content_block_delta' ? asText(data.delta?.text ?? '') : asText(data.completion ?? '');
      else if (provider === 'ollama') fragment = asText(data.response ?? data.message?.content ?? '');
    } catch { /* Ignore malformed stream frames while preserving the app response. */ }
    if (fragment) fragments.push(fragment);
  }
  return fragments.join('');
}

async function responseTextLimited(response) {
  if (!response.body || typeof response.body.getReader !== 'function') return { text: '', overflow: false };
  const reader = response.body.getReader();
  const chunks = [];
  let size = 0;
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      const chunk = Buffer.from(value);
      if (size + chunk.length > MAX_CAPTURE_BYTES) {
        await reader.cancel().catch(() => {});
        return { text: '', overflow: true };
      }
      chunks.push(chunk);
      size += chunk.length;
    }
  } finally {
    try { reader.releaseLock(); } catch { /* Reader may already be canceled. */ }
  }
  return { text: Buffer.concat(chunks, size).toString('utf8'), overflow: false };
}

function emitCapture(url, body, responseStatus, raw, elapsedMs, overflow) {
  const provider = providerFor(url);
  if (!provider) return null;
  let answer = '';
  let error = null;
  const streaming = body.stream === true;
  if (responseStatus < 200 || responseStatus >= 300) error = `http_${responseStatus}`;
  else if (overflow) error = 'stream_capture_limit_exceeded';
  else {
    try {
      if (streaming) answer = streamAnswer(raw, provider);
      else answer = answerFromData(JSON.parse(raw), provider);
      if (!answer) error = streaming ? 'stream_response_not_recognized' : 'response_shape_not_recognized';
    } catch { error = 'unparseable_response'; }
  }
  const prompt = promptFromBody(body);
  const chunks = financeChunks(prompt);
  const blocks = chunks.length ? [] : contextBlocks(prompt);
  const truncated = overflow || prompt.length > 30_000 || answer.length > 20_000;
  return {
    question: questionFromBody(body, prompt),
    prompt: prompt.slice(0, 30_000),
    retrieved_chunks: chunks.slice(0, 10),
    prompt_context_blocks: blocks,
    llm_answer: answer.slice(0, 20_000),
    workspace_path: process.env.RAG_DOCTOR_WORKSPACE || process.cwd(),
    retrieval_observed: chunks.length ? true : null,
    retrieval_count: chunks.length || null,
    metadata: {
      provider,
      model: typeof body.model === 'string' && body.model.trim() ? body.model.trim().slice(0, 120) : provider,
      operation: 'generate',
      error,
      streaming,
      capture_truncated: Boolean(truncated),
      context_extraction_method: chunks.length ? 'finance_source_labels' : blocks[0]?.extraction_method || 'none',
      prompt_context_block_count: blocks.length,
      latency_ms: Math.max(0, Number(elapsedMs.toFixed(3))),
    },
    include_content: true,
  };
}

function requestUrl(args, defaultProtocol) {
  const first = args[0];
  let parsed = null;
  let options = {};
  if (first instanceof URL || typeof first === 'string') {
    try { parsed = new URL(String(first)); } catch { return null; }
    if (args[1] && typeof args[1] === 'object' && !(args[1] instanceof URL)) options = args[1];
  } else if (first && typeof first === 'object') {
    options = first;
    if (args[1] && typeof args[1] === 'object' && !(args[1] instanceof URL)) options = { ...options, ...args[1] };
  } else return null;

  const protocol = options.protocol || (parsed && parsed.protocol) || defaultProtocol;
  const hostname = options.hostname || options.host || (parsed && parsed.hostname) || 'localhost';
  const port = options.port || (parsed && parsed.port);
  const requestPath = options.path || (parsed && `${parsed.pathname}${parsed.search}`) || '/';
  const hostWithPort = String(hostname).includes(':') && !String(hostname).startsWith('[')
    ? `[${hostname}]${port ? `:${port}` : ''}`
    : `${hostname}${port ? `:${port}` : ''}`;
  try { return new URL(`${protocol}//${hostWithPort}${requestPath.startsWith('/') ? requestPath : `/${requestPath}`}`).toString(); }
  catch { return null; }
}

function requestChunk(chunk, encoding) {
  try {
    if (Buffer.isBuffer(chunk)) return chunk;
    if (chunk instanceof Uint8Array) return Buffer.from(chunk);
    if (typeof chunk === 'string') return Buffer.from(chunk, typeof encoding === 'string' ? encoding : 'utf8');
  } catch { /* Unsupported chunk encodings are omitted from the capture. */ }
  return null;
}

function observeHttpModule(httpModule, defaultProtocol) {
  if (!httpModule || typeof httpModule.request !== 'function' || httpModule.request[INSTALL_GUARD]) return;
  const originalRequest = httpModule.request;
  function observedRequest(...args) {
    const url = requestUrl(args, defaultProtocol);
    const provider = url && providerFor(url);
    const request = originalRequest.apply(this, args);
    if (!provider || !request || typeof request.once !== 'function') return request;

    const started = process.hrtime.bigint();
    const requestChunks = [];
    let requestBytes = 0;
    let requestOverflow = false;
    const appendRequestChunk = (chunk, encoding) => {
      const buffer = requestChunk(chunk, encoding);
      if (!buffer || requestOverflow) return;
      if (requestBytes + buffer.length > MAX_CAPTURE_BYTES) {
        requestOverflow = true;
        requestChunks.length = 0;
        return;
      }
      requestChunks.push(buffer);
      requestBytes += buffer.length;
    };

    const originalWrite = request.write;
    if (typeof originalWrite === 'function') {
      request.write = function ragDoctorObservedWrite(chunk, encoding, callback) {
        appendRequestChunk(chunk, encoding);
        return originalWrite.apply(this, arguments);
      };
    }
    const originalEnd = request.end;
    if (typeof originalEnd === 'function') {
      request.end = function ragDoctorObservedEnd(chunk, encoding, callback) {
        if (chunk !== undefined && chunk !== null && (typeof chunk === 'string' || Buffer.isBuffer(chunk) || chunk instanceof Uint8Array)) {
          appendRequestChunk(chunk, encoding);
        }
        return originalEnd.apply(this, arguments);
      };
    }

    request.once('response', (response) => {
      if (!response || typeof response.push !== 'function') return;
      const responseChunks = [];
      let responseBytes = 0;
      let responseOverflow = false;
      let finalized = false;
      const originalPush = response.push;
      response.push = function ragDoctorObservedPush(chunk, encoding) {
        if (chunk !== null && chunk !== undefined && !responseOverflow) {
          const buffer = requestChunk(chunk, encoding);
          if (buffer) {
            if (responseBytes + buffer.length > MAX_CAPTURE_BYTES) {
              responseOverflow = true;
              responseChunks.length = 0;
            } else {
              responseChunks.push(buffer);
              responseBytes += buffer.length;
            }
          }
        }
        const result = originalPush.apply(this, arguments);
        if ((chunk === null || chunk === undefined) && !finalized) {
          finalized = true;
          void (async () => {
            let responseBody = responseOverflow ? Buffer.alloc(0) : Buffer.concat(responseChunks, responseBytes);
            const encoding = String(response.headers && response.headers['content-encoding'] || '').toLowerCase();
            if (responseBody.length && encoding) {
              try {
                const [nodeMajor, nodeMinor] = String(process.versions && process.versions.node || '0.0').split('.').map(Number);
                if (nodeMajor < 16 || (nodeMajor === 16 && nodeMinor < 17)) throw new Error('bounded decompression is unavailable');
                const zlib = require('node:zlib');
                responseBody = encoding === 'gzip' ? zlib.gunzipSync(responseBody, { maxOutputLength: MAX_CAPTURE_BYTES })
                  : encoding === 'deflate' ? zlib.inflateSync(responseBody, { maxOutputLength: MAX_CAPTURE_BYTES })
                    : encoding === 'br' ? zlib.brotliDecompressSync(responseBody, { maxOutputLength: MAX_CAPTURE_BYTES }) : responseBody;
              } catch { responseBody = Buffer.alloc(0); responseOverflow = true; }
            }
            const raw = responseBody.toString('utf8');
            let body = {};
            if (!requestOverflow) {
              try {
                const parsed = JSON.parse(Buffer.concat(requestChunks, requestBytes).toString('utf8'));
                if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) body = parsed;
              } catch { /* Non-JSON requests still produce a bounded metadata-only capture. */ }
            }
            const elapsedMs = Number(process.hrtime.bigint() - started) / 1_000_000;
            const payload = emitCapture(url, body, Number(response.statusCode || 0), raw, elapsedMs, requestOverflow || responseOverflow);
            if (payload) await postLocal('/api/capture-live-run', payload);
          })().catch(() => {});
        }
        return result;
      };
    });
    return request;
  }
  Object.defineProperty(observedRequest, INSTALL_GUARD, { value: true });
  httpModule.request = observedRequest;
}

function installHttpObservers() {
  try {
    const http = require('node:http');
    const https = require('node:https');
    nativeLoopbackRequest = http.request.bind(http);
    observeHttpModule(http, 'http:');
    observeHttpModule(https, 'https:');
    require('node:module').syncBuiltinESMExports();
  } catch { /* A restricted Node runtime may not expose the built-in transport modules. */ }
}

async function postLocal(route, payload) {
  const port = String(process.env.RAG_DOCTOR_PORT || '8765');
  const body = JSON.stringify(payload);
  try {
    if (nativeFetch) {
      await nativeFetch(`http://127.0.0.1:${port}${route}`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body,
        signal: typeof AbortSignal.timeout === 'function' ? AbortSignal.timeout(2_000) : undefined,
      });
      return;
    }
    if (!nativeLoopbackRequest) return;
    await new Promise((resolve) => {
      const request = nativeLoopbackRequest({
        hostname: '127.0.0.1', port, path: route, method: 'POST',
        headers: { 'content-type': 'application/json', 'content-length': Buffer.byteLength(body) },
      });
      request.on('response', (response) => {
        if (typeof response.resume === 'function') response.resume();
        response.on('end', resolve);
      });
      request.on('error', resolve);
      if (typeof request.setTimeout === 'function') request.setTimeout(2_000, () => request.destroy());
      request.end(body);
    });
  } catch { /* Local diagnostics must never fail the application request. */ }
}

function install() {
  if (process.env.RAG_DOCTOR_CAPTURE !== '1' || globalThis[INSTALL_GUARD]) return;
  globalThis[INSTALL_GUARD] = true;
  installHttpObservers();
  const workspace = process.env.RAG_DOCTOR_WORKSPACE || process.cwd();
  void postLocal('/api/runtime-agent-ready', {
    workspace_path: workspace,
    process_id: process.pid,
    providers: ['google', 'openai', 'anthropic', 'ollama'],
    observer_installed: true,
    detail: 'Observed supported Google, OpenAI, Anthropic, and Ollama calls through Node.js fetch/http/https transports.',
  });

  if (!nativeFetch) return;
  globalThis.fetch = async function ragDoctorObservedFetch(input, init) {
    const url = typeof input === 'string' || input instanceof URL ? String(input) : String(input && input.url || '');
    if (!providerFor(url)) return nativeFetch(input, init);
    const started = process.hrtime.bigint();
    const bodyPromise = requestBody(input, init);
    const response = await nativeFetch(input, init);
    let copy;
    try { copy = response.clone(); } catch { return response; }
    void (async () => {
      const [body, captured] = await Promise.all([bodyPromise, responseTextLimited(copy)]);
      const elapsedMs = Number(process.hrtime.bigint() - started) / 1_000_000;
      const payload = emitCapture(url, body, response.status, captured.text, elapsedMs, captured.overflow);
      if (payload) await postLocal('/api/capture-live-run', payload);
    })().catch(() => {});
    return response;
  };
}

if (require.main !== module) install();

module.exports = {
  providerFor,
  promptFromBody,
  questionFromBody,
  financeChunks,
  contextBlocks,
  answerFromData,
  streamAnswer,
  emitCapture,
  requestUrl,
};
