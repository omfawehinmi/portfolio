import { projects } from '../data/projects.js'
import Icon from './ui/Icon.jsx'
import Modal from './ui/Modal.jsx'
import CompactList from './ui/CompactList.jsx'
import ProjectCard from './ProjectCard.jsx'
import { labIds, tier2, tier2Heading } from '../data/tier2.js'
import SectionHeader from './ui/SectionHeader.jsx'
import FeatureRow from './FeatureRow.jsx'
import { featureRows } from '../data/featureRows.js'
import McpSpotlight from './McpSpotlight.jsx'
import MigrationCaseStudy from './case/MigrationCaseStudy.jsx'

function ProjectBody({ project, onOpen }) {
  const parts = (project.parts || []).map((id) => projects.find((item) => item.id === id)).filter(Boolean)
  const parent = project.parentId ? projects.find((item) => item.id === project.parentId) : null

  return (
    <>
      {project.videoFile ? (
        <video className="case-video" src={project.videoFile} controls playsInline muted loop />
      ) : null}
      {project.video ? (
        <div className="media-well">
          <iframe
            src={project.video}
            title={project.title}
            allow="autoplay; fullscreen; picture-in-picture; clipboard-write"
          />
        </div>
      ) : null}
      {project.thumb && !project.videoFile && !project.video ? (
        <img
          src={project.thumb}
          alt=""
          className="case-still"
          style={{ objectPosition: project.thumbPos || 'center top' }}
        />
      ) : null}
      {project.caseStudy === 'migration' ? <MigrationCaseStudy /> : null}
      {project.summary?.map((paragraph) => (
        <p key={paragraph}>{paragraph}</p>
      ))}
      {project.points ? (
        <ul className="case-points">
          {project.points.map((point) => (
            <li key={point}>{point}</li>
          ))}
        </ul>
      ) : null}
      {parts.length ? (
        <div className="case-parts">
          <p className="case-parts__label">Highlighted components</p>
          {parts.map((part) => (
            <button type="button" className="case-part" key={part.id} onClick={() => onOpen(part.id)}>
              <Icon name={part.icon} />
              <span>
                <strong>{part.title}</strong>
                <em>{part.outcome}</em>
              </span>
            </button>
          ))}
        </div>
      ) : null}
      <div className="case-links">
        {parent ? (
          <button type="button" className="button button--flex button--ghost" onClick={() => onOpen(parent.id)}>
            Back to {parent.title}
            <Icon name="uil-arrow-left" className="button__icon" />
          </button>
        ) : null}
        {project.links?.map((link) => (
          <a key={link.href} href={link.href} className="button button--flex" target="_blank" rel="noreferrer">
            {link.label}
            <Icon name="uil-external-link-alt" className="button__icon" />
          </a>
        ))}
        {project.github ? (
          <a href={project.github} className="button button--flex">
            GitHub <Icon name="uil-github-alt" className="button__icon" />
          </a>
        ) : null}
      </div>
    </>
  )
}

function componentChips(project) {
  return (project.parts || [])
    .map((id) => projects.find((item) => item.id === id))
    .filter(Boolean)
    .map((part) => ({ label: part.title, opens: part.id }))
}

export default function Projects({ openId, onOpen }) {
  const openProject = projects.find((project) => project.id === openId)
  const byId = (id) => projects.find((project) => project.id === id)

  return (
    <section className="projects section" id="projects">
      <SectionHeader title="Projects" subtitle="Production systems first. Components open from the system they belong to." />

      <div className="rb-wrap rb-projects">
        <p className="rb-kicker rb-projects__kicker">Production</p>
        <McpSpotlight onOpen={onOpen} />
        {featureRows.map((row, index) => {
          const project = byId(row.projectId)
          return (
            <FeatureRow
              key={row.id}
              {...row}
              chips={row.chips || componentChips(project)}
              chipsLabel={row.chipsLabel || `${project.title} components`}
              title={project.title}
              outcome={project.outcome}
              cta={{ label: project.modalTitle, opens: project.id }}
              eager={index === 0}
              onOpen={onOpen}
            />
          )
        })}

        <h3 className="rb-subh" id="t2">{tier2Heading}</h3>
        <div className="rb-t2">
          {tier2.map((card) => {
            const project = byId(card.id)
            return (
              <ProjectCard
                key={card.id}
                id={card.id}
                kicker={card.kicker}
                title={project.title}
                outcome={project.outcome}
                thumb={card.thumb}
                chips={componentChips(project)}
                onOpen={onOpen}
              />
            )
          })}
        </div>

        <h3 className="rb-subh">Lab</h3>
        <CompactList
          label="Lab projects"
          items={labIds.map((id) => {
            const project = byId(id)
            return { key: id, icon: project.icon, title: project.title, line: project.outcome, onClick: () => onOpen(id) }
          })}
        />
      </div>

      <Modal open={Boolean(openProject)} title={openProject?.modalTitle} onClose={() => onOpen(null)}>
        {openProject ? <ProjectBody key={openProject.id} project={openProject} onOpen={onOpen} /> : null}
      </Modal>
    </section>
  )
}
