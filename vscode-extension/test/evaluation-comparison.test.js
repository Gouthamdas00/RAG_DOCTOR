const test = require('node:test');
const assert = require('node:assert/strict');
const { compareSnapshots, snapshotEvaluation } = require('../evaluation-comparison');

test('snapshots retain evaluation metrics but never retain prompts or answers', () => {
  const snapshot = snapshotEvaluation([{
    testCase: { case_id: 'case-a', question: 'private question' },
    diagnosis: { root_cause: 'NO_FAILURE', status: 'reference_supported', gold_document_retrieved: true, llm_answer: 'private answer' },
    appResponse: { latency_ms: 12.5, answer: 'private answer' },
  }], { digest: 'abc', complete: true });

  assert.deepEqual(snapshot.cases, [{
    caseId: 'case-a', rootCause: 'NO_FAILURE', status: 'reference_supported', supported: true,
    answerReferenceHit: null, goldDocumentRetrieved: true, latencyMs: 12.5,
  }]);
  assert.equal(JSON.stringify(snapshot).includes('private'), false);
});

test('compares paired cases and reports source edits separately from outcome changes', () => {
  const before = snapshotEvaluation([
    { testCase: { case_id: 'a' }, diagnosis: { root_cause: 'RETRIEVAL', gold_document_retrieved: false, answer_reference_check: { status: 'numeric_value_not_found' } }, appResponse: { latency_ms: 20 } },
    { testCase: { case_id: 'b' }, diagnosis: { root_cause: 'NO_FAILURE', gold_document_retrieved: true, answer_reference_check: { status: 'numeric_value_present' } }, appResponse: { latency_ms: 30 } },
  ], { digest: 'old', complete: true });
  const after = snapshotEvaluation([
    { testCase: { case_id: 'a' }, diagnosis: { root_cause: 'NO_FAILURE', gold_document_retrieved: true, answer_reference_check: { status: 'numeric_value_present' } }, appResponse: { latency_ms: 10 } },
    { testCase: { case_id: 'b' }, diagnosis: { root_cause: 'NO_FAILURE', gold_document_retrieved: true }, appResponse: { latency_ms: null } },
    { testCase: { case_id: 'new' }, diagnosis: { root_cause: 'NO_FAILURE' }, appResponse: {} },
  ], { digest: 'new', complete: true });

  const result = compareSnapshots(before, after);
  assert.equal(result.pairedCount, 2);
  assert.equal(result.beforeSupported, 1);
  assert.equal(result.afterSupported, 2);
  assert.equal(result.beforeReferenceAssessed, 2);
  assert.equal(result.afterReferenceAssessed, 1);
  assert.equal(result.beforeReferenceObserved, 1);
  assert.equal(result.afterReferenceObserved, 1);
  assert.equal(result.beforeGold, 1);
  assert.equal(result.afterGold, 2);
  assert.equal(result.beforeLatencyMs, 25);
  assert.equal(result.afterLatencyMs, 10);
  assert.equal(result.sourceChange, 'changed');
  assert.equal(result.cases[0].supportChanged, true);
});

test('partial source fingerprints are never reported as a complete source comparison', () => {
  const comparison = compareSnapshots(
    { sourceFingerprint: { digest: 'old', complete: false }, cases: [] },
    { sourceFingerprint: { digest: 'new', complete: true }, cases: [] },
  );
  assert.equal(comparison.sourceChange, 'partial');
});
