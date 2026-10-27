import Icon from './ui/Icon.jsx'

export default function ImpactBand({ items, onOpen }) {
  return (
    <section className="rb-band" aria-labelledby="rb-band-h">
      <div className="rb-wrap">
        <h2 className="rb-band__h" id="rb-band-h">Selected impact</h2>
        <ul className="rb-band__grid">
          {items.map((item) => (
            <li key={item.title}>
              <a
                className="rb-card rb-lift rb-band__card"
                href={item.href}
                onClick={(event) => {
                  if (!onOpen || !item.projectId) return
                  event.preventDefault()
                  onOpen(item.projectId)
                }}
              >
                <Icon name={item.icon} className="rb-band__icon" />
                <span className="rb-band__title">{item.title}</span>
                <span className="rb-band__body">{item.body}</span>
              </a>
            </li>
          ))}
        </ul>
      </div>
    </section>
  )
}
