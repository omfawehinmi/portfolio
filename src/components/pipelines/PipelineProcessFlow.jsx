import StatePill from './StatePill.jsx'
import { Legend } from './PipelineArchitectureDiagram.jsx'

const TYPE = { trigger: 'Trigger', decision: 'Decision', action: 'Step', outcome: 'Outcome' }

function stepText(step) {
  let t = `Step ${step.n}, ${step.type}: ${step.label}`
  if (step.state !== 'running') t += ` (${step.state})`
  if (step.branch) {
    t += `; ${step.branch.label.toLowerCase().startsWith('no') ? `if ${step.branch.label.toLowerCase()}` : step.branch.label.toLowerCase()}: ${step.branch.text}`
    if (step.branch.loopsTo) t += `, back to step ${step.branch.loopsTo}`
  }
  return t
}

export default function PipelineProcessFlow({ id, title, status, steps, ariaLabel, longDescription }) {
  const titleId = `${id}-flow-title`
  const descId = `${id}-flow-desc`

  return (
    <figure className="pl-figure" aria-labelledby={titleId} aria-describedby={descId}>
      <figcaption className="pl-figure__title" id={titleId}>
        {title} <span className="visually-hidden">: {ariaLabel}</span>
      </figcaption>
      {status ? <p className="pl-figure__status">{status}</p> : null}
      <p className="visually-hidden" id={descId}>
        {longDescription}
      </p>
      <ol className="visually-hidden">
        {steps.map((s) => (
          <li key={s.n}>{stepText(s)}</li>
        ))}
      </ol>
      <div className="pl-flow" aria-hidden="true">
        {steps.map((s, i) => (
          <div key={s.n} className={`pl-step${i === steps.length - 1 ? ' is-last' : ''}`}>
            <div className={`pl-step__card pl-node is-${s.state} is-${s.type}`}>
              <span className="pl-step__kicker">
                {s.n} · {TYPE[s.type]}
              </span>
              <strong className="pl-node__title">{s.label}</strong>
              {s.state !== 'running' ? <StatePill state={s.state} /> : null}
            </div>
            {s.branch ? (
              <div className={`pl-branch is-${s.branch.state}`}>
                <span className="pl-branch__label">{s.branch.label}</span>
                <span className={`pl-node pl-branch__card is-${s.branch.state}`}>
                  <span className="pl-node__title">{s.branch.text}</span>
                  {s.branch.loopsTo ? <span className="pl-node__detail">↺ back to step {s.branch.loopsTo}</span> : null}
                  {s.branch.state === 'planned' ? <StatePill state="planned" /> : null}
                </span>
              </div>
            ) : null}
            {i < steps.length - 1 ? <span className={`pl-flow__arrow is-${steps[i + 1].state}`} /> : null}
          </div>
        ))}
      </div>
      <Legend />
    </figure>
  )
}
