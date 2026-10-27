import { useEffect, useState } from 'react'
import { navLinks, profile } from '../data/profile.js'
import Icon from './ui/Icon.jsx'

export default function Header({ dark, onToggleTheme, scrolled, activeSection }) {
  const [menuOpen, setMenuOpen] = useState(false)

  useEffect(() => {
    if (!menuOpen) return undefined
    const onKeyDown = (e) => {
      if (e.key === 'Escape') setMenuOpen(false)
    }
    document.addEventListener('keydown', onKeyDown)
    return () => document.removeEventListener('keydown', onKeyDown)
  }, [menuOpen])

  return (
    <header className={`header${scrolled ? ' scroll-header' : ''}`} id="header">
      <nav className="nav container">
        <a href="#home" className="nav__logo">
          {profile.name}
        </a>

        <div className={`nav__menu${menuOpen ? ' show-menu' : ''}`} id="nav-menu">
          <ul className="nav__list grid">
            {navLinks.map((link) => (
              <li className="nav__item" key={link.id}>
                <a
                  href={link.href}
                  className={`nav__link${activeSection === link.id ? ' active-link' : ''}`}
                  onClick={() => setMenuOpen(false)}
                >
                  <Icon name={link.icon} className="nav__icon" /> {link.label}
                </a>
              </li>
            ))}
          </ul>
          <button
            type="button"
            className="nav__close"
            id="nav-close"
            onClick={() => setMenuOpen(false)}
            aria-label="Close navigation menu"
          >
            <Icon name="uil-times" />
          </button>
        </div>

        <div className="nav__btns">
          <button
            type="button"
            className="change-theme"
            id="theme-button"
            onClick={onToggleTheme}
            aria-label="Toggle color theme"
            aria-pressed={dark}
          >
            <i className={`uil ${dark ? 'uil-sun' : 'uil-moon'}`} aria-hidden="true"></i>
          </button>
          <button
            type="button"
            className="nav__toggle"
            id="nav-toggle"
            onClick={() => setMenuOpen((open) => !open)}
            aria-label="Toggle navigation menu"
            aria-expanded={menuOpen}
            aria-controls="nav-menu"
          >
            <Icon name="uil-apps" />
          </button>
        </div>
      </nav>
    </header>
  )
}
