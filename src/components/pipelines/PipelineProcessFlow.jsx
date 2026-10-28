import FigureHeader from './FigureHeader.jsx'
import StatePill from './StatePill.jsx'
import { Legend } from './PipelineArchitectureDiagram.jsx'

const TYPE = { trigger: 'Trigger', decision: 'Decision', action: 'Step', outcome: 'Outcome' }

export function stepText(step) {
  let t = `Step ${step.n}, ${step.type}: ${step.label}`
  if (step.state !== 'running') t += ` (${step.state})`
  if (step.branch) {
    const l = step.branch.label.toLowerCase()
    t += `; ${l.startsWith('no') ? `if ${l}` : l}: ${step.branch.text}`
    if (step.branch.loopsTo) t += `, back to step ${step.branch.loopsTo}`
  }
  return t
}

export default function PipelineProcessFlow({ id, eyebrow, title, status, steps, ariaLabel, longDescription }) {
  const titleId = `${id}-flow-title`
  const descId = `${id}-flow-desc`

  return (
    <figure className="pl-figure" aria-labelledby={titleId} aria-describedby={descId}>
      <FigureHeader eyebrow={eyebrow} title={title} status={status} titleId={titleId} ariaLabel={ariaLabel} />
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
              {s.type === 'decision' ? (
                <svg className="pl-diamond" viewBox="0 0 100 100" preserveAspectRatio="none" focusable="false">
                  <polygon points="50,1 99,50 50,99 1,50" vectorEffect="non-scaling-stroke" />
                </svg>
              ) : null}
              <span className="pl-step__kicker">
                {s.n} · {TYPE[s.type]}
              </span>
              <strong className="pl-node__title">{s.label}</strong>
              <StatePill state={s.state} />
            </div>
            {s.branch ? (
              <div className={`pl-branch is-${s.branch.state}`}>
                <span className="pl-side__arrow">{s.branch.label === 'No' ? 'No' : s.branch.label}</span>
                <span className={`pl-node pl-branch__card is-${s.branch.state}`}>
                  <span className="pl-side__tag">Side path</span>
                  <span className="pl-node__title">{s.branch.text}</span>
                  {s.branch.loopsTo ? <span className="pl-node__detail">↺ back to step {s.branch.loopsTo}</span> : null}
                  <StatePill state={s.branch.state} />
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
