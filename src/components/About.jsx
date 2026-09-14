import { publicUrl } from '../utils.js'

export default function About() {
  return (
    <section className="about section" id="about">
      <h2 className="section__title">About Me</h2>
      <span className="section__subtitle">My Introduction</span>

      <div className="about__container container grid">
        <img src={publicUrl('assets/img/michael_fawehinmi.png')} alt="Michael Fawehinmi" className="about__img" />

        <div className="about__data">
          <p className="about__description">
            Senior Data Engineer and AI solutions architect with 10+ years in big data analytics, specializing in
            Snowflake Cortex LLM development, Power BI enterprise reporting, and end-to-end AI pipeline design.
            Accomplished in translating complex business requirements into scalable, production-grade data and AI
            systems for C-suite stakeholders across financial services, healthcare, and tech industries.
          </p>

          <div className="about__info">
            <div>
              <span className="about__info-title">10+</span>
              <span className="about__info-name">
                Years <br /> experience
              </span>
            </div>
            <div>
              <span className="about__info-title">100+</span>
              <span className="about__info-name">
                Completed <br /> projects
              </span>
            </div>
            <div>
              <span className="about__info-title">4+</span>
              <span className="about__info-name">
                Companies <br /> worked
              </span>
            </div>
          </div>

          <div className="about__buttons">
            <a download="" href={publicUrl('assets/pdf/Michael-Fawehinmi-Resume.pdf')} className="button button--flex">
              Download Resume <i className="uil uil-download-alt button__icon"></i>
            </a>
          </div>
        </div>
      </div>
    </section>
  )
}
