import { highlights } from '../data/projects.js'
import Icon from './ui/Icon.jsx'
import SiteHero, { heroProps } from './SiteHero.jsx'
export default function Home({ onOpenProject }) {
  return (
    <>
      <SiteHero {...heroProps} />
      <div className="rb-wrap">
        <div className="impact home__impact" aria-label="Selected impact">
          <p className="impact__kicker">Selected impact</p>
          <div className="impact__grid">
            {highlights.map((item) => (
              <button type="button" className="impact__card" key={item.label} onClick={() => onOpenProject(item.projectId)}>
                <Icon name={item.icon} className="impact__icon" />
                <h2 className="impact__label">{item.label}</h2>
                <p className="impact__detail">{item.detail}</p>
              </button>
            ))}
          </div>
        </div>
      </div>
    </>
  )
}
