export default function Footer() {
  return (
    <footer className="footer">
      <div className="footer__bg">
        <div className="footer__container container grid">
          <div>
            <h1 className="footer__title">Michael</h1>
            <span className="footer__subtitle">Senior Data Engineer &amp; AI Solutions Architect</span>
          </div>

          <ul className="footer__links">
            <li>
              <a href="#qualification" className="footer__link">
                Education | Work
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
            <a
              href="https://www.linkedin.com/in/michael-fawehinmi"
              target="_blank"
              rel="noreferrer"
              className="footer__social"
            >
              <i className="uil uil-linkedin-alt"></i>
            </a>
            <a
              href="https://github.com/omfawehinmi/portfolio"
              target="_blank"
              rel="noreferrer"
              className="footer__social"
            >
              <i className="uil uil-github-alt"></i>
            </a>
          </div>
        </div>

        <p className="footer__copy">&copy; 2026 Michael Fawehinmi. All rights reserved.</p>
      </div>
    </footer>
  )
}
