import { useState } from 'react'
import { pipelines, pipelinesSection } from '../../data/pipelines.js'
import CaseTabs from '../ui/CaseTabs.jsx'
import Modal from '../ui/Modal.jsx'
import SectionHeader from '../ui/SectionHeader.jsx'
import PipelineArchitectureDiagram from './PipelineArchitectureDiagram.jsx'
import PipelineProcessFlow from './PipelineProcessFlow.jsx'

function List({ items, ordered }) {
  const Tag = ordered ? 'ol' : 'ul'
  return (
    <Tag className={`case-points${ordered ? ' case-points--ordered' : ''}`}>
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

function PipelineBody({ p }) {
  return (
    <div className="pl-body">
      <p className="pipeline__status">
        <strong>Status:</strong> {p.status}. {p.statusDetail}
      </p>
      <CaseTabs
        tabs={[
          {
            id: 'architecture',
            label: 'Architecture',
            content: (
              <PipelineArchitectureDiagram
                id={p.id}
                eyebrow={`Pipeline ${p.number} · Architecture`}
                title={p.title}
                status={p.status}
                {...p.architecture}
              />
            ),
          },
          {
            id: 'flow',
            label: 'Process flow',
            content: <PipelineProcessFlow
                id={p.id}
                eyebrow={`Pipeline ${p.number} · Process flow`}
                title={p.title}
                status={p.status}
                {...p.flow}
              />,
          },
        ]}
      />
      {p.architectureNotes ? <List items={p.architectureNotes} /> : null}
      <h4 className="pipeline__heading">Problem</h4>
      {p.problem.map((t) => (
        <p key={t}>{t}</p>
      ))}
      {p.problemPoints ? <List items={p.problemPoints} /> : null}
      <h4 className="pipeline__heading">Method</h4>
      <List items={p.method} />
      <h4 className="pipeline__heading">Reliability</h4>
      {p.reliabilityIntro ? <p>{p.reliabilityIntro}</p> : null}
      <List items={p.reliability} ordered={p.reliabilityOrdered} />
      <h4 className="pipeline__heading">Cost discipline</h4>
      <List items={p.cost} />
      {p.security ? (
        <>
          <h4 className="pipeline__heading">Security</h4>
          <List items={p.security} />
        </>
      ) : null}
      <h4 className="pipeline__heading">What it shows</h4>
      <p>{p.shows}</p>
      <p className="pipeline__stack">{p.stack.join(' · ')}</p>
    </div>
  )
}

export default function Pipelines() {
  const [openId, setOpenId] = useState(null)
  const open = pipelines.find((p) => p.id === openId)

  return (
    <section className="pipelines section" id="pipelines">
      <SectionHeader title={pipelinesSection.title} subtitle={pipelinesSection.subtitle} />
      <div className="container">
        <ul className="project-grid pipelines__grid">
          {pipelines.map((p) => (
            <li key={p.id}>
              <button type="button" className="project-tile pipelines__tile" onClick={() => setOpenId(p.id)}>
                <span className="project-tile__copy">
                  <span className="project-tile__kicker">Pipeline {p.number}</span>
                  <span className="project-tile__title">{p.title}</span>
                  <span className="pipelines__tile-status">{p.status}</span>
                  <span className="project-tile__outcome">{p.outcome}</span>
                </span>
              </button>
            </li>
          ))}
        </ul>
      </div>
      <Modal open={Boolean(open)} wide title={open ? `Pipeline ${open.number}: ${open.title}` : ''} onClose={() => setOpenId(null)}>
        {open ? <PipelineBody key={open.id} p={open} /> : null}
      </Modal>
    </section>
  )
}
