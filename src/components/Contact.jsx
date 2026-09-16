import { useState } from 'react'
import { profile } from '../data/profile.js'
import Icon from './ui/Icon.jsx'
import SectionHeader from './ui/SectionHeader.jsx'

export default function Contact() {
  const [copied, setCopied] = useState(false)

  const copyEmail = async () => {
    try {
      await navigator.clipboard.writeText(profile.email)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1800)
    } catch {
      setCopied(false)
    }
  }

  const details = [
    { icon: 'uil-phone', title: 'Call', value: profile.phone, href: profile.phoneHref },
    { icon: 'uil-envelope', title: 'Email', value: profile.email, href: `mailto:${profile.email}`, copy: true },
    { icon: 'uil-map-marker', title: 'Location', value: profile.location },
  ]

  return (
    <section className="contact section" id="contact">
      <SectionHeader title="Contact" subtitle="Snowflake, agents, or a messy warehouse. Start here." />

      <div className="contact__container container">
        {details.map((item) => (
          <div className="contact__information" key={item.title}>
            <Icon name={item.icon} className="contact__icon" />
            <div>
              <h3 className="contact__title">{item.title}</h3>
              {item.href ? (
                <a className="contact__subtitle" href={item.href}>
                  {item.value}
                </a>
              ) : (
                <span className="contact__subtitle">{item.value}</span>
              )}
              {item.copy ? (
                <button type="button" className="copy-btn" onClick={copyEmail}>
                  {copied ? 'Copied' : 'Copy'}
                </button>
              ) : null}
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}
