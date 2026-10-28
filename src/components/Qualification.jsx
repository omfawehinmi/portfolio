import { education, work } from '../data/experience.js'
import CompactList from './ui/CompactList.jsx'
import RoleCard from './RoleCard.jsx'
import SectionHeader from './ui/SectionHeader.jsx'

function EarlierRole({ item }) {
  return (
    <details className="rb-card rb-details">
      <summary className="rb-details__summary">
        <i className="uil uil-briefcase-alt rb-row-c__icon" aria-hidden="true" />
        <span className="rb-row-c__text">
          <span className="rb-row-c__title">{item.title}</span>
          <span className="rb-row-c__line">{item.subtitle} · {item.years}</span>
        </span>
        <i className="uil uil-angle-down rb-details__chev" aria-hidden="true" />
      </summary>
      <ul className="rb-role__bullets rb-details__body">
        {item.bullets.map((bullet) => (
          <li key={bullet}>{bullet}</li>
        ))}
      </ul>
    </details>
  )
}

export default function Qualification() {
  // Q3: NetApp stays first as the current role.
  const recentWork = work.slice(0, 2)
  const earlierWork = work.slice(2)

  return (
    <section className="qualification section" id="experience">
      <SectionHeader title="Work" subtitle="NetApp and Novanta up front. Earlier roles fold away." />

      <div className="rb-wrap rb-work">
        <div className="rb-roles">
          {recentWork.map((item) => (
            <RoleCard key={item.subtitle} title={item.title} org={item.subtitle} years={item.years} bullets={item.bullets} />
          ))}
        </div>

        <h3 className="rb-subh">Earlier roles</h3>
        <ul className="rb-clist" aria-label="Earlier roles">
          {earlierWork.map((item) => (
            <li key={item.subtitle}>
              <EarlierRole item={item} />
            </li>
          ))}
        </ul>

        {/* Q7: Leadership folds into Work as People management. */}
        <h3 className="rb-subh">People management</h3>
        <CompactList
          label="People management"
          items={[
            { icon: 'uil-users-alt', title: 'Manages 3 direct reports', line: '5 years of people management' },
          ]}
        />

        <h3 className="rb-subh">Education</h3>
        <CompactList
          label="Education"
          items={education.map((item) => ({
            icon: 'uil-graduation-cap',
            title: item.title,
            line: `${item.subtitle} · ${item.bullets[0]}`,
          }))}
        />
      </div>
    </section>
  )
}
