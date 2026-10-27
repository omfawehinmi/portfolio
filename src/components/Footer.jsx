import { profile } from '../data/profile.js'
import Icon from './ui/Icon.jsx'

export default function Footer({ children }) {
  return (
    <footer className="footer">
      <div className="footer__bg">
        <div className="footer__container container grid">
          <div>
            <h2 className="footer__title">{profile.firstName}</h2>
            <span className="footer__subtitle">
              {profile.title} · {profile.titleAlt}
            </span>
          </div>

          <ul className="footer__links">
            <li>
              <a href="#experience" className="footer__link">
                Work
              </a>
            </li>
            <li>
              <a href="#projects" className="footer__link">
                Projects
              </a>
            </li>
            <li>
              <a href="#contact" className="footer__link">
                Contact
              </a>
            </li>
          </ul>

          <div className="footer__socials">
            <a href={profile.linkedin} target="_blank" rel="noreferrer" className="footer__social" aria-label="LinkedIn">
              <Icon name="uil-linkedin-alt" />
            </a>
            <a href={profile.github} target="_blank" rel="noreferrer" className="footer__social" aria-label="GitHub">
              <Icon name="uil-github-alt" />
            </a>
            <a href={profile.myorglab} target="_blank" rel="noreferrer" className="footer__social" aria-label="MyOrgLab">
              <Icon name="uil-snowflake" />
            </a>
          </div>
        </div>

        <p className="footer__copy">&copy; {new Date().getFullYear()} {profile.name}. All rights reserved.</p>
      </div>
      {children}
    </footer>
  )
}
