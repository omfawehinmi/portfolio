const LABEL = { running: 'Running', paused: 'Paused', planned: 'Planned' }

export default function StatePill({ state }) {
  return <span className={`pl-pill is-${state}`}>{LABEL[state]}</span>
}
