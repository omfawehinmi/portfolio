const LABEL = { running: 'Running', paused: 'Paused', planned: 'Planned' }

// Running nodes show only the accent bar; pass `always` to render the Running pill (legend only).
export default function StatePill({ state, always = false }) {
  if (state === 'running' && !always) return null
  return <span className={`pl-pill is-${state}`}>{LABEL[state]}</span>
}
