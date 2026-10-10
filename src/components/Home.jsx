import { profile } from '../data/profile.js'
import { heroMedia, highlights, projects } from '../data/projects.js'
import { publicUrl } from '../utils.js'
import Icon from './ui/Icon.jsx'
import ProjectMedia from './ui/ProjectMedia.jsx'

export default function Home({ onOpenProject }) {
  const reel = heroMedia.map((item) => ({
    ...item,
    project: projects.find((project) => project.id === item.id),
  }))

  return (
    <section className="home section" id="home">
      <div className="home__container container grid">
        <div className="home__content grid">
          <div className="home__social">
            <a href={profile.linkedin} target="_blank" rel="noreferrer" className="home__social-icon" aria-label="LinkedIn">
              <Icon name="uil-linkedin-alt" />
            </a>
            <a href={profile.github} target="_blank" rel="noreferrer" className="home__social-icon" aria-label="GitHub">
              <Icon name="uil-github-alt" />
            </a>
            <a href={profile.myorglab} target="_blank" rel="noreferrer" className="home__social-icon" aria-label="MyOrgLab">
              <Icon name="uil-snowflake" />
            </a>
          </div>

          <div className="home__img">
            <svg className="home__blob" viewBox="0 0 200 187">
              <mask id="mask0" maskType="alpha">
                <path d="M190.312 36.4879C206.582 62.1187 201.309 102.826 182.328 134.186C163.346 165.547 130.807 187.559 100.226 186.353C69.6454 185.297 41.0228 161.023 21.7403 129.362C2.45775 97.8511 -7.48481 59.1033 6.67581 34.5279C20.9871 10.1032 59.7028 -0.149132 97.9666 0.00163737C136.23 0.303176 174.193 10.857 190.312 36.4879Z" />
              </mask>
              <g mask="url(#mask0)">
                <path d="M190.312 36.4879C206.582 62.1187 201.309 102.826 182.328 134.186C163.346 165.547 130.807 187.559 100.226 186.353C69.6454 185.297 41.0228 161.023 21.7403 129.362C2.45775 97.8511 -7.48481 59.1033 6.67581 34.5279C20.9871 10.1032 59.7028 -0.149132 97.9666 0.00163737C136.23 0.303176 174.193 10.857 190.312 36.4879Z" />
                <image className="home__blob-img" x="13" y="18" href={publicUrl('assets/img/home.png')} />
              </g>
            </svg>

            <div className="home__reel" aria-label="Project media">
              {reel.map((item) =>
                item.project ? (
                  <button
                    type="button"
                    className="home__reel-tile"
                    key={item.id}
                    onClick={() => onOpenProject(item.id)}
                  >
                    <ProjectMedia project={item.project} />
                    <span>{item.label}</span>
                  </button>
                ) : null,
              )}
            </div>
          </div>

          <div className="home__data">
            <p className="home__eyebrow">{profile.eyebrow}</p>
            <h1 className="home__title">Hi, I&apos;m {profile.firstName}</h1>
            <h3 className="home__subtitle">
              {profile.title}
              <span className="home__subtitle-sep"> · </span>
              {profile.titleAlt}
            </h3>
            <p className="home__lede">{profile.headline}</p>
            <p className="home__description">{profile.summary}</p>
            <p className="home__about">{profile.about}</p>
            <p className="home__meta">
              {profile.years} years in data · {profile.location}
            </p>
            <div className="home__actions">
              <a download="" href={publicUrl('assets/pdf/Michael-Fawehinmi-Resume.pdf')} className="button button--flex">
                Resume <Icon name="uil-download-alt" className="button__icon" />
              </a>
              <a href="#projects" className="button button--flex button--ghost">
                Projects <Icon name="uil-arrow-right" className="button__icon" />
              </a>
            </div>
          </div>
        </div>

        <div className="impact home__impact" aria-label="Selected impact">
          <p className="impact__kicker">Selected impact</p>
          <div className="impact__grid">
            {highlights.map((item) => (
              <button type="button" className="impact__card" key={item.label} onClick={() => onOpenProject(item.projectId)}>
                <Icon name={item.icon} className="impact__icon" />
                <h3 className="impact__label">{item.label}</h3>
                <p className="impact__detail">{item.detail}</p>
              </button>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}
