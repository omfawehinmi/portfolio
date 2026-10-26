import { pipelines } from '../data/pipelines.js'
import ArchitectureDiagram from './ui/ArchitectureDiagram.jsx'
import Fold from './ui/Fold.jsx'
import SectionHeader from './ui/SectionHeader.jsx'

function List({ items, ordered }) {
  const Tag = ordered ? 'ol' : 'ul'
  return (
    <Tag className="case-points">
      {items.map((item) => (
        <li key={Array.isArray(item) ? item[0] : item}>
          {Array.isArray(item) ? (
            <>
              <strong>{item[0]}</strong> {item[1]}
            </>
          ) : (
            item
          )}
        </li>
      ))}
    </Tag>
  )
}

// Slot for the numbered process flow, authored in design review.
function FlowSlot({ pipeline }) {
  return (
    <div className="pipeline__flow-slot" data-flow-slot={pipeline.id} role="note">
      <p className="pipeline__flow-label">Process flow</p>
      <p>Placeholder: the numbered step-by-step flow for this pipeline goes here from the design review.</p>
    </div>
  )
}

export default function Pipelines() {
  return (
    <section className="pipelines section" id="pipelines">
      <SectionHeader title="Pipelines" subtitle="Four household data pipelines: the problem, the architecture, and what is running versus planned." />

      <div className="pipelines__list container">
        {pipelines.map((p) => (
          <article className="pipeline" id={`pipeline-${p.id}`} key={p.id}>
            <p className="pipeline__eyebrow">Pipeline {p.number}</p>
            <h3 className="pipeline__title">{p.title}</h3>
            <p className="pipeline__status">
              <strong>Status:</strong> {p.status}. {p.statusDetail}
            </p>
            <p className="pipeline__stack">{p.stack.join(' · ')}</p>

            <h4 className="pipeline__heading">Problem</h4>
            {p.problem.map((t) => (
              <p key={t}>{t}</p>
            ))}
            {p.problemPoints ? <List items={p.problemPoints} /> : null}

            <ArchitectureDiagram id={p.id} diagram={p.architecture} />
            {p.architectureNotes ? <List items={p.architectureNotes} /> : null}

            <FlowSlot pipeline={p} />

            <Fold title="Method">
              <List items={p.method} />
            </Fold>
            <Fold title="Reliability">
              {p.reliabilityIntro ? <p>{p.reliabilityIntro}</p> : null}
              <List items={p.reliability} ordered={p.reliabilityOrdered} />
            </Fold>
            <Fold title="Cost discipline">
              <List items={p.cost} />
            </Fold>
            {p.security ? (
              <Fold title="Security">
                <List items={p.security} />
              </Fold>
            ) : null}
            <p className="pipeline__shows">
              <strong>What it shows:</strong> {p.shows}
            </p>
          </article>
        ))}
      </div>
    </section>
  )
}
