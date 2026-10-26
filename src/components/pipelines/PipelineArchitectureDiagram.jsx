import FigureHeader from './FigureHeader.jsx'
import StatePill from './StatePill.jsx'

function SidePath({ label }) {
  return (
    <span className="pl-side">
      <span className="pl-side__arrow">No</span>
      <span className="pl-side__card">
        <span className="pl-side__tag">Side path</span>
        <span>{label}</span>
      </span>
    </span>
  )
}

export default function PipelineArchitectureDiagram({ id, eyebrow, title, status, lanes, edges, sidePaths = [], ariaLabel, longDescription }) {
  const titleId = `${id}-arch-title`
  const descId = `${id}-arch-desc`

  return (
    <figure className="pl-figure" aria-labelledby={titleId} aria-describedby={descId}>
      <FigureHeader eyebrow={eyebrow} title={title} status={status} titleId={titleId} ariaLabel={ariaLabel} />
      <p className="visually-hidden" id={descId}>
        {longDescription}
        {sidePaths.map((sp) => ` Side path from ${sp.from}: ${sp.label}.`).join('')}
      </p>
      <ol className="pl-lanes" aria-hidden="true">
        {lanes.map((lane, i) => {
          const edge = edges[i]
          return (
            <li key={lane.key} className={`pl-lane-wrap${edge ? '' : ' is-last'}`}>
              <div className="pl-lane">
                <p className="pl-lane__label">{lane.label}</p>
                <ul className="pl-nodes">
                  {lane.nodes.map((node) => (
                    <li key={node.title} className={`pl-node is-${node.state}`}>
                      <span className="pl-node__top">
                        <strong className="pl-node__title">{node.title}</strong>
                        <StatePill state={node.state} />
                      </span>
                      <span className="pl-node__detail">{node.detail}</span>
                      {sidePaths
                        .filter((sp) => sp.from === node.title)
                        .map((sp) => (
                          <SidePath key={sp.label} label={sp.label} />
                        ))}
                      {node.join ? <span className="pl-node__join">{node.join}</span> : null}
                    </li>
                  ))}
                </ul>
              </div>
              {edge ? (
                <span className={`pl-edge is-${edge.state || 'running'}`}>
                  <span className="pl-edge__num">{i + 1}</span>
                  <span className="pl-edge__label">{edge.label}</span>
                  <span className="pl-edge__line" />
                </span>
              ) : null}
            </li>
          )
        })}
      </ol>
      <Legend />
    </figure>
  )
}

export function Legend() {
  return (
    <div className="pl-legend" aria-hidden="true">
      <span className="pl-legend__item">
        <span className="pl-legend__bar is-running" /> Running
      </span>
      <span className="pl-legend__item">
        <span className="pl-legend__bar is-paused" /> Deployed or built, paused
      </span>
      <span className="pl-legend__item">
        <span className="pl-legend__bar is-planned" /> Planned
      </span>
      <span className="pl-legend__item">
        <span className="pl-legend__edge" /> Data flow
      </span>
      <span className="pl-legend__item">
        <span className="pl-legend__edge is-planned" /> Planned flow
      </span>
      <span className="pl-legend__item">
        <span className="pl-side__arrow">No</span> Side path
      </span>
    </div>
  )
}
