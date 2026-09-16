import { highlights } from '../data/profile.js'
import Fold from './ui/Fold.jsx'
import Icon from './ui/Icon.jsx'

export default function Impact() {
  return (
    <section className="impact" aria-label="Selected impact">
      <div className="container">
        <Fold title="Selected impact" defaultOpen>
          <div className="impact__grid">
            {highlights.map((item) => (
              <article className="impact__card" key={item.label}>
                <Icon name={item.icon} className="impact__icon" />
                <h3 className="impact__label">{item.label}</h3>
                <p className="impact__detail">{item.detail}</p>
              </article>
            ))}
          </div>
        </Fold>
      </div>
    </section>
  )
}
