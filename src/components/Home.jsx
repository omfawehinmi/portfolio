import { publicUrl } from '../utils.js'

export default function Home() {
  return (
    <section className="home section" id="home">
      <div className="home__container container grid">
        <div className="home__content grid">
          <div className="home__social">
            <a
              href="https://www.linkedin.com/in/michael-fawehinmi"
              target="_blank"
              rel="noreferrer"
              className="home__social-icon"
            >
              <i className="uil uil-linkedin-alt"></i>
            </a>
            <a
              href="https://github.com/omfawehinmi/portfolio"
              target="_blank"
              rel="noreferrer"
              className="home__social-icon"
            >
              <i className="uil uil-github-alt"></i>
            </a>
          </div>

          <div className="home__img">
            <svg className="home__blob" viewBox="0 0 200 187">
              <mask id="mask0" maskType="alpha">
                <path d="M190.312 36.4879C206.582 62.1187 201.309 102.826 182.328 134.186C163.346 165.547 130.807 187.559 100.226 186.353C69.6454 185.297 41.0228 161.023 21.7403 129.362C2.45775 97.8511 -7.48481 59.1033 6.67581 34.5279C20.9871 10.1032 59.7028 -0.149132 97.9666 0.00163737C136.23 0.303176 174.193 10.857 190.312 36.4879Z" />
              </mask>
              <g mask="url(#mask0)">
                <path d="M190.312 36.4879C206.582 62.1187 201.309 102.826 182.328 134.186C163.346 165.547 130.807 187.559 100.226 186.353C69.6454 185.297 41.0228 161.023 21.7403 129.362C2.45775 97.8511 -7.48481 59.1033 6.67581 34.5279C20.9871 10.1032 59.7028 -0.149132 97.9666 0.00163737C136.23 0.303176 174.193 10.857 190.312 36.4879Z" />
                <image className="home__blob-img" x="13" y="18" href={publicUrl('assets/img/home.png')} />
              </g>
            </svg>
          </div>

          <div className="home__data">
            <h1 className="home__title">Hi, I&apos;m Michael</h1>
            <h3 className="home__subtitle">Senior Data Engineer | AI Solutions Architect</h3>
            <p className="home__description">
              Senior Data Engineer and AI solutions architect driving impactful business decisions through advanced
              analytics, LLM-powered intelligence, and automated process improvements. Deep expertise in Snowflake data
              architecture, Power BI enterprise reporting, and end-to-end AI pipeline development.
            </p>
            <a href="#contact" className="button button--flex">
              Contactme <i className="uil uil-message button__icon"></i>
            </a>
          </div>
        </div>

        <div className="home__scroll">
          <a href="#about" className="home__scroll-button button--flex">
            <i className="uil uil-mouse-alt home__scroll-mouse"></i>
            <span className="home__scroll-name">Scroll down</span>
            <i className="uil uil-arrow-down home__scroll-arrow"></i>
          </a>
        </div>
      </div>
    </section>
  )
}
