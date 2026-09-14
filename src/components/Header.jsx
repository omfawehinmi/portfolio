import { useState } from 'react'
import { navLinks } from '../data.jsx'

export default function Header({ dark, onToggleTheme, scrolled, activeSection }) {
  const [menuOpen, setMenuOpen] = useState(false)

  return (
    <header className={`header${scrolled ? ' scroll-header' : ''}`} id="header">
      <nav className="nav container">
        <a href="#home" className="nav__logo">
          Michael Fawehinmi
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
                  <i className={`uil ${link.icon} nav__icon`}></i> {link.label}
                </a>
              </li>
            ))}
          </ul>
          <i className="uil uil-times nav__close" id="nav-close" onClick={() => setMenuOpen(false)}></i>
        </div>

        <div className="nav__btns">
          <i
            className={`uil ${dark ? 'uil-sun' : 'uil-moon'} change-theme`}
            id="theme-button"
            onClick={onToggleTheme}
          ></i>
          <div className="nav__toggle" id="nav-toggle" onClick={() => setMenuOpen(true)}>
            <i className="uil uil-apps"></i>
          </div>
        </div>
      </nav>
    </header>
  )
}
