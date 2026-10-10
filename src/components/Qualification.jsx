import { useState } from 'react'
import { education, work } from '../data/experience.js'
import Fold from './ui/Fold.jsx'
import Icon from './ui/Icon.jsx'
import SectionHeader from './ui/SectionHeader.jsx'

function Role({ item }) {
  return (
    <article className="role">
      <div className="role__head">
        <h3 className="role__title">{item.title}</h3>
        {item.years && <span className="role__years">{item.years}</span>}
      </div>
      <p className="role__org">{item.subtitle}</p>
      {item.bullets?.length ? (
        <ul className="role__bullets">
          {item.bullets.map((bullet) => (
            <li key={bullet}>{bullet}</li>
          ))}
        </ul>
      ) : null}
    </article>
  )
}

export default function Qualification() {
  const [tab, setTab] = useState('work')
  const recentWork = work.slice(0, 2)
  const earlierWork = work.slice(2)
  const items = tab === 'work' ? recentWork : education

  return (
    <section className="qualification section" id="experience">
      <SectionHeader title="Work" subtitle="NetApp and Novanta up front. Earlier roles fold away." />

      <div className="qualification__container container">
        <div className="qualification__tabs">
          <button
            type="button"
            className={`qualification__button button--flex${tab === 'work' ? ' qualification__active' : ''}`}
            onClick={() => setTab('work')}
          >
            <Icon name="uil-briefcase-alt" className="qualification__icon" />
            Work
          </button>
          <button
            type="button"
            className={`qualification__button button--flex${tab === 'education' ? ' qualification__active' : ''}`}
            onClick={() => setTab('education')}
          >
            <Icon name="uil-graduation-cap" className="qualification__icon" />
            Education
          </button>
        </div>

        <div className="role-list">
          {items.map((item) => (
            <Role item={item} key={`${item.title}-${item.subtitle}`} />
          ))}
          {tab === 'work' ? (
            <Fold title={`Earlier roles (${earlierWork.length})`}>
              {earlierWork.map((item) => (
                <Role item={item} key={`${item.title}-${item.subtitle}`} />
              ))}
            </Fold>
          ) : null}
        </div>
      </div>
    </section>
  )
}
