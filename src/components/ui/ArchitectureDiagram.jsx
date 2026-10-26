const STATUS_LABEL = { live: 'Running', paused: 'Paused', planned: 'Planned' }

export default function ArchitectureDiagram({ id, diagram }) {
  const titleId = `${id}-arch-title`
  const descId = `${id}-arch-desc`

  return (
    <figure className="arch" aria-labelledby={titleId} aria-describedby={descId}>
      <figcaption className="arch__caption" id={titleId}>
        Architecture
      </figcaption>
      <p className="visually-hidden" id={descId}>
        {diagram.longDescription}
      </p>
      <ol className="arch__lanes" aria-label={diagram.ariaLabel}>
        {diagram.lanes.map((lane, i) => {
          const edge = diagram.edges[i]
          return (
            <li key={lane.key} className={`arch__lane-wrap${edge ? '' : ' is-last'}`}>
              <div className="arch__lane">
                <p className="arch__lane-label">{lane.label}</p>
                <ul className="arch__nodes">
                  {lane.nodes.map((node) => (
                    <li key={node.title} className={`arch__node is-${node.status}`}>
                      <span className="arch__node-top">
                        <strong className="arch__node-title">{node.title}</strong>
                        <span className="arch__pill">{STATUS_LABEL[node.status]}</span>
                      </span>
                      <span className="arch__node-detail">{node.detail}</span>
                      {node.note ? <span className="arch__node-note">{node.note}</span> : null}
                      {node.join ? <span className="arch__join" aria-hidden="true">{node.join}</span> : null}
                    </li>
                  ))}
                </ul>
              </div>
              {edge ? (
                <span className={`arch__edge${edge.planned ? ' is-planned' : ''}`} aria-hidden="true">
                  <span className="arch__edge-label">{edge.label}</span>
                  <span className="arch__edge-line" />
                </span>
              ) : null}
            </li>
          )
        })}
      </ol>
      <p className="arch__legend">
        <span className="arch__key is-live">Running</span>
        <span className="arch__key is-paused">Deployed or built, paused</span>
        <span className="arch__key is-planned">Planned (dashed)</span>
      </p>
    </figure>
  )
}
