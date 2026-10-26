// Strict content check for the public pipeline case studies.
// Fails on account IDs, hostnames, emails, internal links, secrets paths,
// private identifiers, clock times and cron strings.
import { readFileSync } from 'node:fs'

const files = [
  'src/data/pipelines.js',
  'src/components/Pipelines.jsx',
  'src/components/ui/ArchitectureDiagram.jsx',
]

const banned = [
  ['account locator', /\b[A-Z]{2}\d{5}\b/],
  ['12-digit account id', /\b\d{12}\b/],
  ['email address', /[\w.+-]+@[\w-]+\.[\w.]+/],
  ['warehouse hostname', /snowflakecomputing\.com/i],
  ['machine hostname', /DESKTOP-[A-Z0-9]+/i],
  ['secrets path', /\/run\/secrets|secretmanager|secrets\//i],
  ['tunnel host', /trycloudflare/i],
  ['private repo link', /myorglab-repos|github\.com\/myorglab|household-platform/i],
  ['blob link', /\/blob\//],
  ['internal sources footer', /internal sources/i],
  ['private identifier', /\b(R|SVC|MCP|CS|MONARCH)_[A-Z0-9_]+\b/i],
  ['clock time', /\b\d{1,2}(:\d{2})?\s?(AM|PM)\b/i],
  ['24h clock time', /\b([01]?\d|2[0-3]):[0-5]\d\b/],
  ['timezone', /\b(UTC|GMT|ET|EST|EDT)\b/],
  ['cron string', /(^|['"\s])[\d*\/,-]+\s+[\d*\/,-]+\s+[\d*\/,-]+\s+[\d*\/,-]+\s+[\d*\/,-]+(['"\s]|$)/],
  ['banned word', new RegExp('\\b' + ['ex', 'it'].join('') + '\\b', 'i')],
]

const problems = []
for (const f of files) {
  readFileSync(f, 'utf8').split('\n').forEach((line, i) => {
    for (const [name, re] of banned) {
      if (re.test(line)) problems.push(`${f}:${i + 1} ${name}: ${line.trim().slice(0, 100)}`)
    }
  })
}

// Every planned node must say so in its status; planned edges must say "planned".
const { pipelines } = await import('../src/data/pipelines.js')
for (const p of pipelines) {
  if (p.architecture.edges.length !== p.architecture.lanes.length - 1) problems.push(`${p.id}: edge count`)
  for (const e of p.architecture.edges) {
    if (e.planned && !/planned/i.test(e.label)) problems.push(`${p.id}: planned edge label "${e.label}"`)
  }
  for (const l of p.architecture.lanes) for (const n of l.nodes) {
    if (!['live', 'paused', 'planned'].includes(n.status)) problems.push(`${p.id}: bad status ${n.title}`)
  }
}

if (problems.length) {
  throw new Error(`check-pipelines found ${problems.length} problem(s):\n${problems.join('\n')}`)
}
console.log(`check-pipelines: ${files.length} files clean, ${pipelines.length} case studies valid`)
