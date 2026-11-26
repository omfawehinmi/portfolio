import { useLayoutEffect, useState } from 'react'
import { DIAG } from '../../data/architecture.js'
import { dAirflow, dDbt, dErd, dFeat, dLead, dMcp, dMychart, dPipes, dRearch, dSec, dStack } from './architectureSvg.js'

const builders = {
  lead: (tall, inst) => dLead(DIAG, tall, inst),
  mcp: (_tall, inst) => dMcp(DIAG, inst),
  domains: (_tall, inst) => dErd(DIAG, inst),
  airflow: (_tall, inst) => dAirflow(DIAG, inst),
  dbt: (_tall, inst) => dDbt(DIAG, inst),
  pipes: (_tall, inst) => dPipes(DIAG, inst),
  security: (_tall, inst) => dSec(DIAG, inst),
  stack: (_tall, inst) => dStack(DIAG, inst),
  features: (_tall, inst) => dFeat(DIAG, inst),
  rearch: (_tall, inst) => dRearch(inst),
  mychart: (tall, inst) => dMychart(DIAG, tall, inst),
}

export default function ArchitectureDiagram({ kind, tall = false, inst = 't' }) {
  const [markup, setMarkup] = useState('')

  useLayoutEffect(() => {
    const build = builders[kind]
    if (!build) {
      setMarkup('')
      return undefined
    }
    let cancelled = false
    const render = () => {
      if (!cancelled) setMarkup(build(tall, inst))
    }
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(render)
    } else {
      render()
    }
    return () => {
      cancelled = true
    }
  }, [kind, tall, inst])

  if (!markup) return null
  return <div className="bx-svg" dangerouslySetInnerHTML={{ __html: markup }} />
}
