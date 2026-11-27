import { useLayoutEffect, useRef, useState } from 'react'
import { DIAG } from '../../data/architecture.js'
import { noteFor, tableNote } from '../../data/nodeNotes.js'
import {
  dAirflow,
  dTerraform,
  dDatabase,
  dDatabaseLanes,
  dDbt,
  dGrokbot,
  dInfra,
  dLead,
  dMcp,
  dPipes,
  dRearch,
  dSec,
  dWorkflow,
} from './architectureSvg.js'

const builders = {
  lead: (tall, inst) => dLead(DIAG, tall, inst),
  mcp: (_tall, inst) => dMcp(DIAG, inst),
  workflow: (_tall, inst, variant) => dWorkflow(DIAG, inst, variant),
  // Per-DB Liam ERD (Style A green). Tile defaults to ledger; modal tabs pass dbId.
  domains: (_tall, inst, dbId) =>
    String(inst).includes('tile') ? dDatabaseLanes(DIAG, dbId || 'ledger', inst) : dDatabase(DIAG, dbId || 'ledger', inst),
  airflow: (_tall, inst) => dAirflow(DIAG, inst),
  terraform: (_tall, inst) => dTerraform(DIAG, inst),
  dbt: (_tall, inst) => dDbt(DIAG, inst),
  pipes: (_tall, inst) => dPipes(DIAG, inst),
  security: (_tall, inst) => dSec(DIAG, inst),
  infra: (_tall, inst) => dInfra(DIAG, inst),
  grokbot: (_tall, inst) => dGrokbot(DIAG, inst),
  rearch: (_tall, inst) => dRearch(inst),
}

export default function ArchitectureDiagram({
  kind,
  tall = false,
  inst = 't',
  dbId = null,
  variant = null,
}) {
  const [markup, setMarkup] = useState('')
  const [pop, setPop] = useState(null)
  const hostRef = useRef(null)
  const popRef = useRef(null)

  useLayoutEffect(() => {
    const build = builders[kind]
    if (!build) {
      setMarkup('')
      return undefined
    }
    let cancelled = false
    const render = () => {
      if (!cancelled) setMarkup(build(tall, inst, variant ?? dbId))
    }
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(render)
    } else {
      render()
    }
    return () => {
      cancelled = true
    }
  }, [kind, tall, inst, dbId, variant])

  useLayoutEffect(() => {
    const host = hostRef.current
    if (!host || !markup) return undefined
    const svg = host.querySelector('svg.bx-arch')
    if (!svg) return undefined

    const labelOf = (node) =>
      Array.from(node.querySelectorAll('text'))
        .map((el) => el.textContent)
        .join(' ')

    const nodeNote = (node) =>
      kind === 'domains'
        ? tableNote(node.dataset.schema, node.dataset.node)
        : noteFor(labelOf(node), kind)

    svg.querySelectorAll('.nd').forEach((node) => {
      if (nodeNote(node)) node.classList.add('has-note')
    })

    const clear = () => {
      svg.querySelectorAll('.nd.is-selected').forEach((el) => el.classList.remove('is-selected'))
      setPop(null)
    }

    const select = (node) => {
      if (!node || !node.classList.contains('nd') || node.classList.contains('nd-tag')) return
      const already = node.classList.contains('is-selected')
      clear()
      if (already) return
      node.classList.add('is-selected')
      const note = nodeNote(node)
      if (!note) return
      const hostBox = host.getBoundingClientRect()
      const box = node.getBoundingClientRect()
      const below = box.bottom - hostBox.top + 8
      const flip = below + 120 > hostBox.height && box.top - hostBox.top > 130
      setPop({
        ...note,
        left: Math.min(Math.max(box.left - hostBox.left + box.width / 2, 130), hostBox.width - 130),
        top: flip ? box.top - hostBox.top - 8 : below,
        flip,
      })
    }

    const onOutside = (event) => {
      const node = event.target.closest?.('.nd')
      if (node && svg.contains(node) && !node.classList.contains('nd-tag')) return
      // The popover is click-through so it never blocks the node beneath it; a click on its
      // text still counts as inside the description.
      const r = popRef.current?.getBoundingClientRect()
      if (r && event.clientX >= r.left && event.clientX <= r.right && event.clientY >= r.top && event.clientY <= r.bottom) return
      clear()
    }

    const onEsc = (event) => {
      if (event.key === 'Escape') clear()
    }

    const onPointer = (event) => {
      const node = event.target.closest?.('.nd')
      if (!node || !svg.contains(node)) return
      if (node.classList.contains('nd-tag')) return
      event.preventDefault()
      event.stopPropagation()
      select(node)
    }

    const onKey = (event) => {
      if (event.key !== 'Enter' && event.key !== ' ') return
      const node = event.target.closest?.('.nd')
      if (!node || !svg.contains(node) || node.classList.contains('nd-tag')) return
      event.preventDefault()
      select(node)
    }

    svg.addEventListener('click', onPointer)
    svg.addEventListener('keydown', onKey)
    document.addEventListener('click', onOutside, true)
    document.addEventListener('keydown', onEsc)
    return () => {
      svg.removeEventListener('click', onPointer)
      svg.removeEventListener('keydown', onKey)
      document.removeEventListener('click', onOutside, true)
      document.removeEventListener('keydown', onEsc)
    }
  }, [markup, kind])

  if (!markup) return null
  return (
    <div className="bx-svg-wrap">
      <div className="bx-svg" ref={hostRef} dangerouslySetInnerHTML={{ __html: markup }} />
      {pop ? (
        <div
          ref={popRef}
          className={`bx-node-pop${pop.flip ? ' is-above' : ''}`}
          role="status"
          style={{ left: pop.left, top: pop.top }}
        >
          <strong>{pop.title}</strong>
          <span>{pop.text}</span>
        </div>
      ) : null}
    </div>
  )
}
