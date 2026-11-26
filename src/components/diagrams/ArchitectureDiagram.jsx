import { useLayoutEffect, useState } from 'react'
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

function themeDark() {
  return typeof document !== 'undefined' && document.body.classList.contains('dark-theme')
}

export default function ArchitectureDiagram({ kind, tall = false, inst = 't', dbId = null }) {
  const [markup, setMarkup] = useState('')
  const [dark, setDark] = useState(themeDark)

  useLayoutEffect(() => {
    const onTheme = () => setDark(themeDark())
    const observer = new MutationObserver(onTheme)
    observer.observe(document.body, { attributes: true, attributeFilter: ['class'] })
    return () => observer.disconnect()
  }, [])

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
  }, [kind, tall, inst, dbId, dark])

  if (!markup) return null
  return <div className="bx-svg" dangerouslySetInnerHTML={{ __html: markup }} />
}
