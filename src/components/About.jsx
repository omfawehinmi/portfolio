import { profile } from '../data/profile.js'
import { publicUrl } from '../utils.js'
import Icon from './ui/Icon.jsx'
import Fold from './ui/Fold.jsx'
import SectionHeader from './ui/SectionHeader.jsx'

export default function About() {
  return (
    <section className="about section" id="about">
      <SectionHeader title="About" subtitle="Snowflake-first, production-shaped" />

      <div className="about__container container grid">
        <img src={publicUrl('assets/img/michael_fawehinmi.png')} alt="Michael Fawehinmi" className="about__img" />

        <div className="about__data">
          <p className="about__description">
            Senior Data Engineer and AI Solution Architect who treats Snowflake as the system of record for both analytics and agents. At
            Novanta I was the sole engineer who stood up the platform, wired Workday into certified layers, then put
            Cortex LLMs on that grain until Tier 0/1 reporting queues went away. At NetApp I land SuccessFactors,
            Fieldglass, Eightfold, and Visier into Snowflake and One Model.
          </p>

          <div className="about__info">
            <div>
              <span className="about__info-title">{profile.years}</span>
              <span className="about__info-name">
                Years <br /> in data
              </span>
            </div>
            <div>
              <span className="about__info-title">1</span>
              <span className="about__info-name">
                Snowflake <br /> platform, sole-owned
              </span>
            </div>
            <div>
              <span className="about__info-title">0</span>
              <span className="about__info-name">
                Tier 0/1 <br /> queues left
              </span>
            </div>
          </div>

          <div className="about__buttons">
            <a download="" href={publicUrl('assets/pdf/Michael-Fawehinmi-Resume.pdf')} className="button button--flex">
              Download resume <Icon name="uil-download-alt" className="button__icon" />
            </a>
            <a href={profile.myorglab} target="_blank" rel="noreferrer" className="button button--flex button--ghost">
              MyOrgLab <Icon name="uil-external-link-alt" className="button__icon" />
            </a>
          </div>

          <Fold title="MyOrgLab, agents, and what else I run">
            <p className="about__description">
              In parallel I am building{' '}
              <a href={profile.myorglab} target="_blank" rel="noreferrer">
                MyOrgLab
              </a>
              : governed domain agents, semantic HR, and MCP tool bridges that run in the customer&apos;s Snowflake
              account. The projects below are the proof: live products, a household Cortex finance agent I operate,
              camera tracking, the PALMS punching bag, and forensic ML.
            </p>
          </Fold>
        </div>
      </div>
    </section>
  )
}
