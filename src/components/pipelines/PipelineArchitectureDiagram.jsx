import StatePill from './StatePill.jsx'

export default function PipelineArchitectureDiagram({ id, eyebrow, title, status, lanes, edges, ariaLabel, longDescription }) {
  const titleId = `${id}-arch-title`
  const descId = `${id}-arch-desc`

  return (
    <figure className="pl-figure" aria-labelledby={titleId} aria-describedby={descId}>
      <p className="pl-eyebrow">{eyebrow}</p>
      <figcaption className="pl-figure__title" id={titleId}>
        {title} <span className="visually-hidden">: {ariaLabel}</span>
      </figcaption>
      {status ? <p className="pl-figure__status">{status}</p> : null}
      <p className="visually-hidden" id={descId}>
        {longDescription}
      </p>
      <ol className="pl-lanes">
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
                      {node.note ? <span className="pl-node__note">{node.note}</span> : null}
                      {node.join ? <span className="pl-node__join" aria-hidden="true">{node.join}</span> : null}
                    </li>
                  ))}
                </ul>
              </div>
              {edge ? (
                <span className={`pl-edge is-${edge.state || 'running'}`} aria-hidden="true">
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
    <p className="pl-legend">
      <StatePill state="running" />
      <StatePill state="paused" />
      <span className="pl-pill is-planned">Planned (dashed)</span>
    </p>
  )
}
