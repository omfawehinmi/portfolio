import { useEffect, useState } from 'react'
import { projects } from '../data.jsx'

export default function Projects() {
  const [openIndex, setOpenIndex] = useState(null)

  useEffect(() => {
    document.body.style.overflow = openIndex === null ? 'auto' : 'hidden'

    const onKeyDown = (event) => {
      if (event.key === 'Escape') setOpenIndex(null)
    }

    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.body.style.overflow = 'auto'
      document.removeEventListener('keydown', onKeyDown)
    }
  }, [openIndex])

  return (
    <section className="services section" id="projects">
      <h2 className="section__title">Featured Projects</h2>
      <br />
      <div className="services__container container grid">
        {projects.map((project, index) => (
          <div className="services__content" key={project.icon}>
            <div>
              <i className={`uil ${project.icon} services__icon`}></i>
              <h3 className="services__title">{project.title}</h3>
            </div>

            <span
              className="button button--flex button--small button--link services__button"
              onClick={() => setOpenIndex(index)}
            >
              View More
              <i className="uil uil-arrow-right button__icon"></i>
            </span>

            <div
              className={`services__modal${openIndex === index ? ' active-modal' : ''}`}
              onClick={(event) => {
                if (event.target === event.currentTarget) setOpenIndex(null)
              }}
            >
              <div className="services__modal-content">
                <h4 className="services__modal-title">{project.modalTitle}</h4>
                <i className="uil uil-times services__modal-close" onClick={() => setOpenIndex(null)}></i>
                <div className="services__modal-services">{project.body}</div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}
