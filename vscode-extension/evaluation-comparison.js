function finiteNumber(value) {
  if (value === null || value === undefined || value === '') return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function supported(diagnosis) {
  return Boolean(diagnosis && (
    diagnosis.root_cause === 'NO_FAILURE' || diagnosis.status === 'reference_supported'
  ));
}

function snapshotEvaluation(results, sourceFingerprint) {
  const cases = [];
  const items = results || [];
  for (const item of items) {
    if (!item || !item.testCase || !item.diagnosis) continue;
    cases.push({
      caseId: String(item.testCase.case_id || ''),
      rootCause: String(item.diagnosis.root_cause || ''),
      status: String(item.diagnosis.status || ''),
      supported: supported(item.diagnosis),
      goldDocumentRetrieved: typeof item.diagnosis.gold_document_retrieved === 'boolean'
        ? item.diagnosis.gold_document_retrieved
        : null,
      latencyMs: finiteNumber(item.appResponse && item.appResponse.latency_ms),
    });
  }
  return {
    sourceFingerprint: sourceFingerprint || null,
    totalCases: items.length,
    capturedCases: cases.length,
    cases,
  };
}

function meanLatency(cases, key) {
  const values = cases.map((item) => item[key]).filter((value) => value !== null);
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}

function compareSnapshots(previous, current) {
  const before = new Map((previous && previous.cases || []).map((item) => [item.caseId, item]));
  const after = new Map((current && current.cases || []).map((item) => [item.caseId, item]));
  const paired = [];
  for (const [caseId, beforeCase] of before) {
    const afterCase = after.get(caseId);
    if (afterCase) paired.push({ caseId, before: beforeCase, after: afterCase });
  }
  const count = (predicate) => paired.filter(predicate).length;
  const beforeFingerprint = previous && previous.sourceFingerprint;
  const afterFingerprint = current && current.sourceFingerprint;
  let sourceChange = 'unavailable';
  if (beforeFingerprint && afterFingerprint && beforeFingerprint.digest && afterFingerprint.digest) {
    sourceChange = beforeFingerprint.complete && afterFingerprint.complete
      ? beforeFingerprint.digest === afterFingerprint.digest ? 'unchanged' : 'changed'
      : 'partial';
  }
  return {
    pairedCount: paired.length,
    beforeTotalCases: Number(previous && previous.totalCases) || 0,
    afterTotalCases: Number(current && current.totalCases) || 0,
    beforeCapturedCases: Number(previous && previous.capturedCases) || 0,
    afterCapturedCases: Number(current && current.capturedCases) || 0,
    beforeSupported: count((item) => item.before.supported),
    afterSupported: count((item) => item.after.supported),
    beforeGold: count((item) => item.before.goldDocumentRetrieved === true),
    afterGold: count((item) => item.after.goldDocumentRetrieved === true),
    beforeLatencyMs: meanLatency(paired.map((item) => item.before), 'latencyMs'),
    afterLatencyMs: meanLatency(paired.map((item) => item.after), 'latencyMs'),
    sourceChange,
    cases: paired.map((item) => ({
      caseId: item.caseId,
      beforeCause: item.before.rootCause || item.before.status || 'unknown',
      afterCause: item.after.rootCause || item.after.status || 'unknown',
      supportChanged: item.before.supported !== item.after.supported,
      goldChanged: item.before.goldDocumentRetrieved !== item.after.goldDocumentRetrieved,
    })),
  };
}

module.exports = { compareSnapshots, snapshotEvaluation };
