import { useLayoutEffect, useRef, useState } from 'react'
import { DIAG } from '../../data/architecture.js'
import {
  dAirflow,
  dDatabase,
  dDbt,
  dErd,
  dFeat,
  dGrokbot,
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
  workflow: (_tall, inst) => dWorkflow(DIAG, inst),
  domains: (_tall, inst, dbId) => (dbId ? dDatabase(DIAG, dbId, inst) : dErd(DIAG, inst)),
  airflow: (_tall, inst) => dAirflow(DIAG, inst),
  dbt: (_tall, inst) => dDbt(DIAG, inst),
  pipes: (_tall, inst) => dPipes(DIAG, inst),
  security: (_tall, inst) => dSec(DIAG, inst),
  features: (_tall, inst) => dFeat(DIAG, inst),
  grokbot: (_tall, inst) => dGrokbot(DIAG, inst),
  rearch: (_tall, inst) => dRearch(inst),
}

export default function ArchitectureDiagram({ kind, tall = false, inst = 't', dbId = null }) {
  const [markup, setMarkup] = useState('')
  const hostRef = useRef(null)

  useLayoutEffect(() => {
    const build = builders[kind]
    if (!build) {
      setMarkup('')
      return undefined
    }
    let cancelled = false
    const render = () => {
      if (!cancelled) setMarkup(build(tall, inst, dbId))
    }
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(render)
    } else {
      render()
    }
    return () => {
      cancelled = true
    }
  }, [kind, tall, inst, dbId])

  useLayoutEffect(() => {
    const host = hostRef.current
    if (!host || !markup) return undefined
    const svg = host.querySelector('svg.bx-arch')
    if (!svg) return undefined

    const clear = () => {
      svg.querySelectorAll('.nd.is-selected').forEach((el) => el.classList.remove('is-selected'))
    }

    const select = (node) => {
      if (!node || !node.classList.contains('nd') || node.classList.contains('nd-tag')) return
      const already = node.classList.contains('is-selected')
      clear()
      if (!already) node.classList.add('is-selected')
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
    return () => {
      svg.removeEventListener('click', onPointer)
      svg.removeEventListener('keydown', onKey)
    }
  }, [markup])

  if (!markup) return null
  return <div className="bx-svg" ref={hostRef} dangerouslySetInnerHTML={{ __html: markup }} />
}
