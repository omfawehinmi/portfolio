import test from 'node:test'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), '..')
const read = (rel) => fs.readFileSync(path.join(root, rel), 'utf8')

test('the project popup uses a native dialog with a div fallback, Esc, and focus return', () => {
  const modal = read('src/components/ui/Modal.jsx')
  assert.match(modal, /showModal\(/)
  assert.match(modal, /<dialog/)
  assert.match(modal, /HTMLDialogElement/)
  assert.match(modal, /role="dialog"/)
  assert.match(modal, /event\.key === 'Escape'/)
  assert.match(modal, /trigger\.focus\(/)
  assert.match(modal, /event\.target === event\.currentTarget/)
  const css = read('src/styles.css')
  assert.match(css, /@starting-style/)
  assert.match(css, /dialog\.project-dialog/)
  assert.match(css, /margin:\s*auto/)
  assert.match(css, /prefers-reduced-motion: reduce[\s\S]*dialog\.project-dialog[\s\S]*transition:\s*none/)
})

test('skill panels and the phone More fold interpolate height, with an instant fallback', () => {
  const css = read('src/styles.css')
  assert.match(css, /interpolate-size:\s*allow-keywords/)
  assert.match(css, /\.skills__open \.skills__list[\s\S]*height:\s*auto/)
  assert.match(css, /prefers-reduced-motion: reduce[\s\S]*\.skills__list[\s\S]*transition:\s*none/)
  assert.match(css, /\.project-tile__outcome[\s\S]*height:\s*1lh/)
  assert.match(css, /\.project-tile__outcome\.is-open[\s\S]*height:\s*auto/)
  assert.match(css, /prefers-reduced-motion: reduce[\s\S]*\.project-tile__outcome[\s\S]*transition:\s*none/)
  const projects = read('src/components/Projects.jsx')
  assert.match(projects, /project-tile__more/)
  assert.match(projects, /\{moreOpen \? 'Less' : 'More'\}/)
  assert.match(projects, /project\.outcome/)
})

test('the document opts into OS text scale and does not fix the root in pixels', () => {
  const html = read('index.html')
  assert.match(html, /<meta name="text-scale" content="scale" \/>/)
  const css = read('src/styles.css')
  assert.doesNotMatch(css, /html\s*\{[^}]*font-size:\s*16px/)
})
