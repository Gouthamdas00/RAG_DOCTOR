"use strict";

// Stdlib-only helper for JavaScript/TypeScript RAG applications.
const http = require("http");

function captureRun(run, options = {}) {
  const endpoint = new URL(options.serverUrl || "http://127.0.0.1:8765");
  if (endpoint.protocol !== "http:" || !["127.0.0.1", "localhost", "::1"].includes(endpoint.hostname)) {
    return Promise.reject(new Error("RAG Doctor probe only supports an HTTP loopback endpoint."));
  }
  const includeContent = options.includeContent === true || run.includeContent === true || run.include_content === true;
  const chunks = run.retrievedChunks || run.retrieved_chunks || [];
  const metadataChunks = includeContent ? chunks : chunks.filter((item) => item && typeof item === "object").map((item) => {
    const safe = {};
    for (const key of ["chunk_id", "doc_id", "id", "source", "final_rank", "rank", "score", "dense_score", "bm25_score", "rerank_score", "pre_rerank_rank"]) {
      if (key in item) safe[key] = item[key];
    }
    return safe;
  });
  const payload = Buffer.from(JSON.stringify({
    question: includeContent ? run.question : "",
    prompt: includeContent ? (run.prompt || null) : null,
    retrieved_chunks: metadataChunks,
    llm_answer: includeContent ? (run.answer || run.llmAnswer || run.llm_answer || "") : "",
    workspace_path: run.workspacePath || run.workspace_path || process.cwd(),
    expected_answer: includeContent ? (run.expectedAnswer || run.expected_answer || null) : null,
    source_document_text: includeContent ? (run.sourceDocumentText || run.source_document_text || null) : null,
    oracle_passed: run.oraclePassed ?? run.oracle_passed ?? null,
    max_index_gold_coverage: run.maxIndexGoldCoverage ?? run.max_index_gold_coverage ?? null,
    top_k_final: run.topKFinal ?? run.top_k_final ?? null,
    data_transfer_events: run.dataTransferEvents || run.data_transfer_events || [],
    retrieval_observed: run.retrievalObserved ?? run.retrieval_observed ?? true,
    retrieval_count: run.retrievalCount ?? run.retrieval_count ?? (run.retrievedChunks || run.retrieved_chunks || []).length,
    metadata: run.metadata || {},
    include_content: includeContent,
  }));

  return new Promise((resolve) => {
    const req = http.request({
      hostname: endpoint.hostname,
      port: Number(endpoint.port || 80),
      path: "/api/capture-live-run",
      method: "POST",
      headers: { "content-type": "application/json", "content-length": payload.length },
      timeout: options.timeoutMs || 5000,
    }, (res) => {
      const chunks = [];
      res.on("data", (chunk) => chunks.push(chunk));
      res.on("end", () => {
        try { resolve(JSON.parse(Buffer.concat(chunks).toString("utf8"))); }
        catch { resolve({ status: "offline", detail: "Local RAG Doctor service is unavailable." }); }
      });
    });
    req.on("error", () => resolve({ status: "offline", detail: "Local RAG Doctor service is unavailable." }));
    req.on("timeout", () => req.destroy());
    req.end(payload);
  });
}

module.exports = { captureRun };
