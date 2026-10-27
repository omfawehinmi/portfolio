import { profile } from '../data/profile.js'
import { publicUrl } from '../utils.js'
import Fold from './ui/Fold.jsx'
import Icon from './ui/Icon.jsx'
import Picture from './ui/Picture.jsx'

export const heroProps = {
  headline: profile.headline,
  greeting: `Hi, I'm ${profile.firstName}`,
  roles: [profile.title, profile.titleAlt],
  summary: profile.summary,
  photo: { src: publicUrl('assets/img/home.png'), alt: '' },
  ctas: [
    { label: 'Resume', href: publicUrl('assets/pdf/Michael-Fawehinmi-Resume.pdf'), icon: 'uil-download-alt', download: true, primary: true },
    { label: 'Projects', href: '#projects', icon: 'uil-arrow-right' },
  ],
}

export default function SiteHero({ headline, greeting, roles, summary, photo, ctas }) {
  return (
    <section className="rb-hero" id="home" aria-labelledby="rb-hero-h">
      <div className="rb-wrap rb-hero__grid">
        <div className="rb-hero__text">
          <p className="rb-kicker">{profile.eyebrow}</p>
          <h1 className="rb-hero__h1" id="rb-hero-h">{headline}</h1>
          <p className="rb-hero__who">
            {greeting} · {roles.join(' · ')}
          </p>
          <p className="rb-hero__summary">{summary}</p>
          <Fold title="About">
            <p className="rb-hero__about">{profile.about}</p>
          </Fold>
          <p className="rb-hero__meta">
            {profile.years} years in data · {profile.location}
          </p>
          <div className="rb-hero__ctas">
            {ctas.map((cta) => (
              <a
                key={cta.label}
                href={cta.href}
                download={cta.download ? '' : undefined}
                className={`button button--flex rb-btn${cta.primary ? '' : ' button--ghost'}`}
              >
                {cta.label} <Icon name={cta.icon} className="button__icon" />
              </a>
            ))}
          </div>
        </div>
        <Picture className="rb-hero__photo" src={photo.src} alt={photo.alt} width={374} height={500} eager />
      </div>
    </section>
  )
}
