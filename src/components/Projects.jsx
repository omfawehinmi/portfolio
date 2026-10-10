import { projectGroups, projects } from '../data/projects.js'
import Icon from './ui/Icon.jsx'
import Modal from './ui/Modal.jsx'
import ProjectMedia from './ui/ProjectMedia.jsx'
import SectionHeader from './ui/SectionHeader.jsx'

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

export default function Projects({ openId, onOpen }) {
  const openProject = projects.find((project) => project.id === openId)

  return (
    <section className="projects section" id="projects">
      <SectionHeader title="Projects" subtitle="Open a tile. SOF and Household Ledger components sit with the system they belong to." />

      <div className="projects__stack container">
        {projectGroups.map((group) => (
          <div className="project-group" key={group}>
            <h3 className="project-group__label">{group}</h3>
            <ul className={`project-grid${projects.some((item) => item.group === group && item.featured) ? ' project-grid--ledger' : ''}`}>
              {projects
                .filter((project) => project.group === group)
                .map((project) => (
                  <li key={project.id} className={project.featured ? 'is-featured' : project.parentId ? 'is-part' : undefined}>
                    <button type="button" className={`project-tile${project.parentId ? ' project-tile--part' : ''}`} onClick={() => onOpen(project.id)}>
                      <span className="project-tile__media">
                        <ProjectMedia project={project} />
                      </span>
                      <span className="project-tile__copy">
                        {project.parentId ? <span className="project-tile__kicker">Component</span> : null}
                        <span className="project-tile__title">{project.title}</span>
                        <span className="project-tile__outcome">{project.outcome}</span>
                      </span>
                    </button>
                  </li>
                ))}
            </ul>
          </div>
        ))}
      </div>

      <Modal open={Boolean(openProject)} title={openProject?.modalTitle} onClose={() => onOpen(null)}>
        {openProject ? <ProjectBody key={openProject.id} project={openProject} onOpen={onOpen} /> : null}
      </Modal>
    </section>
  )
}
