const fs = require('fs');
const path = require('path');

const extensionRoot = path.resolve(__dirname, '..');
const projectRoot = path.resolve(extensionRoot, '..');
const source = path.join(projectRoot, 'backend');
const frontendSource = path.join(projectRoot, 'frontend');
const runtime = path.join(extensionRoot, 'runtime');
const target = path.join(runtime, 'backend');
const frontendTarget = path.join(runtime, 'frontend');
const requirements = path.join(projectRoot, 'requirements.txt');
const skippedDirectories = new Set(['.git', '.venv', 'venv', 'node_modules', '__pycache__', '.cache', 'tests', 'dist', 'build']);

if (!fs.existsSync(path.join(source, 'app', 'api', 'server.py'))) {
  throw new Error(`RAG Doctor backend source was not found at ${source}`);
}
if (!fs.existsSync(requirements)) {
  throw new Error(`Python dependency manifest was not found at ${requirements}`);
}
if (!fs.existsSync(path.join(frontendSource, 'index.html'))) {
  throw new Error(`RAG Doctor frontend source was not found at ${frontendSource}`);
}

fs.mkdirSync(runtime, { recursive: true });
// runtime/backend and runtime/frontend are generated package inputs. Remove
// them before staging so files deleted or newly excluded from the source tree
// cannot survive from an older package build.
for (const generatedTarget of [target, frontendTarget]) {
  const relative = path.relative(runtime, generatedTarget);
  if (!relative || relative.startsWith('..') || path.isAbsolute(relative)) {
    throw new Error(`Refusing to replace a generated path outside ${runtime}: ${generatedTarget}`);
  }
  fs.rmSync(generatedTarget, { recursive: true, force: true });
}
fs.cpSync(source, target, {
  recursive: true,
  force: true,
  filter: (entry) => {
    const relative = path.relative(source, entry);
    const segments = relative.split(path.sep);
    return !segments.some((segment) => skippedDirectories.has(segment)) && !entry.endsWith('.pyc');
  },
});
fs.cpSync(frontendSource, frontendTarget, { recursive: true, force: true });
fs.copyFileSync(requirements, path.join(runtime, 'requirements.txt'));
console.log(`Staged local analyzer, frontend, and requirements into ${runtime}`);
