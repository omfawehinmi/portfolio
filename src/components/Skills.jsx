import { skillGroups } from '../data/skills.js'
import SectionHeader from './ui/SectionHeader.jsx'

export default function Skills() {
  return (
    <section className="skills section" id="skills">
      <SectionHeader title="Skills" subtitle="Open a stack when you need it." />

      <div className="rb-wrap">
        <ul className="rb-clist" aria-label="Skill stacks">
          {skillGroups.map((group) => (
            <li key={group.title}>
              <details className="rb-card rb-details">
                <summary className="rb-details__summary">
                  <i className={`uil ${group.icon} rb-row-c__icon`} aria-hidden="true" />
                  <span className="rb-row-c__text">
                    <span className="rb-row-c__title">{group.title}</span>
                    <span className="rb-row-c__line">{group.subtitle}</span>
                  </span>
                  <i className="uil uil-angle-down rb-details__chev" aria-hidden="true" />
                </summary>
                <ul className="tag-list rb-details__tags">
                  {group.items.map((skill) => (
                    <li className={`tag${group.featured ? ' tag--snowflake' : ''}`} key={skill}>
                      {skill}
                    </li>
                  ))}
                </ul>
              </details>
            </li>
          ))}
        </ul>
      </div>
    </section>
  )
}
