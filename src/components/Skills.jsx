import { useState } from 'react'
import { skillGroups } from '../data/skills.js'
import Icon from './ui/Icon.jsx'
import SectionHeader from './ui/SectionHeader.jsx'

export default function Skills() {
  const [openIndex, setOpenIndex] = useState(null)

  return (
    <section className="skills section" id="skills">
      <SectionHeader title="Skills" subtitle="Open a stack when you need it." />

      <div className="skills__container container grid">
        {skillGroups.map((group, index) => {
          const isOpen = openIndex === index

          return (
            <div key={group.title} className={group.featured ? 'skills__featured' : undefined}>
              <div className={`skills__content ${isOpen ? 'skills__open' : 'skills__close'}`}>
                <button
                  type="button"
                  className="skills__header"
                  aria-expanded={isOpen}
                  onClick={() => setOpenIndex((current) => (current === index ? null : index))}
                >
                  <Icon name={group.icon} className="skills__icon" />
                  <div>
                    <h3 className="skills__title">{group.title}</h3>
                    <span className="skills__subtitle">{group.subtitle}</span>
                  </div>
                  <Icon name="uil-angle-down" className="skills__arrow" />
                </button>

                <div className="skills__list">
                  <ul className="tag-list">
                    {group.items.map((skill) => (
                      <li className={`tag${group.featured ? ' tag--snowflake' : ''}`} key={skill}>
                        {skill}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </section>
  )
}
