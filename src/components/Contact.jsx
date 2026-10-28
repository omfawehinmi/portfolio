import { useState } from 'react'
import { profile } from '../data/profile.js'
import { CompactRow } from './ui/CompactList.jsx'
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

  return (
    <section className="contact section" id="contact">
      <SectionHeader title="Contact" subtitle="Snowflake, agents, or a messy warehouse. Start here." />

      <div className="rb-wrap">
        <ul className="rb-clist rb-contact" aria-label="Contact details">
          <li>
            <CompactRow icon="uil-phone" title="Call" line={profile.phone} href={profile.phoneHref} />
          </li>
          <li className="rb-contact__email">
            <CompactRow icon="uil-envelope" title="Email" line={profile.email} href={`mailto:${profile.email}`} />
            <button type="button" className="button rb-btn rb-contact__copy" onClick={copyEmail} aria-label={`Copy ${profile.email}`}>
              <span aria-live="polite">{copied ? 'Copied' : 'Copy'}</span>
            </button>
          </li>
          <li>
            <CompactRow icon="uil-map-marker" title="Location" line={profile.location} />
          </li>
        </ul>
      </div>
    </section>
  )
}
