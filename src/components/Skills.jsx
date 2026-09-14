import { useState } from 'react'
import { skillGroups } from '../data.jsx'

export default function Skills() {
  const [openIndex, setOpenIndex] = useState(null)

  const toggleSkill = (index) => {
    setOpenIndex((current) => (current === index ? null : index))
  }

  return (
    <section className="skills section" id="skills">
      <h2 className="section__title">Skills</h2>
      <span className="section__subtitle">My Technical Level</span>

      <div className="skills__container container grid">
        {skillGroups.map((group, index) => (
          <div key={group.title}>
            <div className={`skills__content ${openIndex === index ? 'skills__open' : 'skills__close'}`}>
              <div className="skills__header" onClick={() => toggleSkill(index)}>
                <i className={`uil ${group.icon} skills__icon`}></i>
                <div>
                  <h1 className="skills__title">{group.title}</h1>
                  <span className="skills__subtitle">{group.subtitle}</span>
                </div>
                <i className="uil uil-angle-down skills__arrow"></i>
              </div>

              <div className="skills__list grid">
                {group.items.map((skill) => (
                  <div className="skills__data" key={skill.name}>
                    <div className="skills__titles" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                      <h3 className="skills__name">{skill.name}</h3>
                      <span className="skills__number">{skill.percent}</span>
                    </div>
                    <div className="skills__bar">
                      <span className={`skills__percentage ${skill.barClass}`}></span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}
