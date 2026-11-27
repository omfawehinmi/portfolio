#!/usr/bin/env node
// Serve the Vite production build at /portfolio/ on port 3000.
// `serve@14.2.6 -s` lists the /portfolio/ directory instead of returning
// index.html, so this uses `vite preview`, which honors the Vite base path.
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const root = path.join(__dirname, '..');
const dist = path.join(root, 'dist');

if (!fs.existsSync(dist)) {
  throw new Error('No dist/ folder. Run npm run build in the portfolio root first.');
}

const child = spawn(
  'npx',
  ['vite', 'preview', '--port', '3000', '--host', '127.0.0.1', '--strictPort'],
  { stdio: 'inherit', cwd: root },
);

const stop = () => {
  if (!child.killed) child.kill('SIGTERM');
};
process.on('SIGTERM', stop);
process.on('SIGINT', stop);
child.on('close', (code) => {
  if (code) {
    throw new Error(`vite preview stopped with status ${code}`);
  }
  stop();
});
