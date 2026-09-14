import { useState } from 'react'
import { education, work } from '../data.jsx'

function QualificationItem({ item }) {
  const details = (
    <div>
      <h3 className="qualification__title">{item.title}</h3>
      <span className="qualification__subtitle">{item.subtitle}</span>
      <div className="qualification__calendar">
        <i className="uil uil-calendar-alt"></i> {item.years}
      </div>
    </div>
  )

  const marker = (
    <div className={item.wrapTime ? 'qualification__time' : undefined}>
      <span className="qualification__rounder"></span>
      <span className="qualification__line"></span>
    </div>
  )

  if (item.side === 'right') {
    return (
      <div className="qualification__data">
        <div></div>
        {marker}
        {details}
      </div>
    )
  }

  return (
    <div className="qualification__data">
      {details}
      {marker}
    </div>
  )
}

export default function Qualification() {
  const [tab, setTab] = useState('education')

  return (
    <section className="qualification section" id="qualification">
      <h2 className="section__title">Qualification</h2>
      <span className="section__subtitle">My Personal Journey</span>

      <div className="qualification__container container">
        <div className="qualification__tabs">
          <div
            className={`qualification__button button--flex${tab === 'education' ? ' qualification__active' : ''}`}
            onClick={() => setTab('education')}
          >
            <i className="uil uil-graduation-cap qualification__icon"></i>
            Education
          </div>
          <div
            className={`qualification__button button--flex${tab === 'work' ? ' qualification__active' : ''}`}
            onClick={() => setTab('work')}
          >
            <i className="uil uil-briefcase-alt qualification__icon"></i>
            Work
          </div>
        </div>

        <div className="qualification__sections">
          <div
            className={`qualification__content${tab === 'education' ? ' qualification__active' : ''}`}
            data-content=""
            id="education"
          >
            {education.map((item) => (
              <QualificationItem item={item} key={`${item.title}-${item.years}`} />
            ))}
          </div>

          <div
            className={`qualification__content${tab === 'work' ? ' qualification__active' : ''}`}
            data-content=""
            id="work"
          >
            {work.map((item) => (
              <QualificationItem item={item} key={`${item.title}-${item.subtitle}`} />
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}
