const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { fingerprintWorkspace } = require('../workspace-fingerprint');

test('source edits change the digest while hidden and dependency files are excluded', async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'rag-doctor-fingerprint-'));
  try {
    await fs.mkdir(path.join(root, 'src'));
    await fs.mkdir(path.join(root, 'node_modules', 'lib'), { recursive: true });
    await fs.mkdir(path.join(root, '.secrets'));
    await fs.writeFile(path.join(root, 'src', 'pipeline.py'), 'top_k = 3\n');
    await fs.writeFile(path.join(root, 'node_modules', 'lib', 'index.js'), 'ignored = true;');
    await fs.writeFile(path.join(root, '.secrets', 'config.json'), '{"token":"private"}');
    const first = await fingerprintWorkspace(root);
    const repeated = await fingerprintWorkspace(root);
    assert.equal(first.digest, repeated.digest);
    assert.equal(first.files, 1);
    assert.equal(first.complete, true);

    await fs.writeFile(path.join(root, 'src', 'pipeline.py'), 'top_k = 8\n');
    const changed = await fingerprintWorkspace(root);
    assert.notEqual(first.digest, changed.digest);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});

test('bounded fingerprints are marked partial when a source file exceeds limits', async () => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'rag-doctor-fingerprint-'));
  try {
    await fs.writeFile(path.join(root, 'large.py'), 'x'.repeat(20));
    const result = await fingerprintWorkspace(root, { maxFileBytes: 10 });
    assert.equal(result.files, 0);
    assert.equal(result.complete, false);
  } finally {
    await fs.rm(root, { recursive: true, force: true });
  }
});
