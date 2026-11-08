const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

const SOURCE_SUFFIXES = new Set([
  '.py', '.js', '.cjs', '.mjs', '.ts', '.tsx', '.jsx', '.vue', '.java', '.kt',
  '.go', '.rs', '.cs', '.rb', '.php', '.scala', '.json', '.yaml', '.yml',
  '.toml', '.ini', '.cfg', '.properties', '.lock',
]);
const SKIP_DIRECTORIES = new Set([
  '.git', '.venv', 'venv', 'node_modules', '__pycache__', '.cache', 'dist', 'build',
]);

async function fingerprintWorkspace(workspacePath, limits = {}) {
  const root = path.resolve(workspacePath);
  const maxFiles = limits.maxFiles || 500;
  const maxFileBytes = limits.maxFileBytes || 1024 * 1024;
  const maxTotalBytes = limits.maxTotalBytes || 32 * 1024 * 1024;
  const hash = crypto.createHash('sha256');
  const pending = [root];
  let files = 0;
  let inspected = 0;
  let totalBytes = 0;
  let complete = true;

  while (pending.length && inspected < maxFiles && totalBytes < maxTotalBytes) {
    const directory = pending.pop();
    let entries;
    try {
      entries = await fs.promises.readdir(directory, { withFileTypes: true });
    } catch {
      complete = false;
      continue;
    }
    entries.sort((a, b) => a.name.localeCompare(b.name));
    for (const entry of entries) {
      if (entry.name.startsWith('.') || entry.isSymbolicLink()) continue;
      const fullPath = path.join(directory, entry.name);
      if (entry.isDirectory()) {
        if (!SKIP_DIRECTORIES.has(entry.name)) pending.push(fullPath);
        continue;
      }
      if (!entry.isFile()) continue;
      if (!SOURCE_SUFFIXES.has(path.extname(entry.name).toLowerCase()) && entry.name.toLowerCase() !== 'requirements.txt') continue;
      inspected += 1;
      if (inspected > maxFiles) {
        complete = false;
        break;
      }
      try {
        const stat = await fs.promises.stat(fullPath);
        if (stat.size > maxFileBytes || totalBytes + stat.size > maxTotalBytes) {
          complete = false;
          continue;
        }
        const content = await fs.promises.readFile(fullPath);
        const relativePath = path.relative(root, fullPath).replace(/\\/g, '/');
        hash.update(relativePath);
        hash.update('\0');
        hash.update(content);
        hash.update('\0');
        files += 1;
        totalBytes += content.length;
      } catch {
        complete = false;
      }
      if (totalBytes >= maxTotalBytes) {
        complete = false;
        break;
      }
    }
  }
  if (pending.length || inspected >= maxFiles) complete = false;
  return { digest: hash.digest('hex'), complete, files, totalBytes };
}

module.exports = { fingerprintWorkspace };
