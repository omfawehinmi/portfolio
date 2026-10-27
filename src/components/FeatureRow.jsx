import Picture from './ui/Picture.jsx'

export default function FeatureRow({ id, kicker, title, outcome, chips = [], chipsLabel, cta, media, flip = false, eager = false, onOpen, children }) {
  const headingId = `${id}-h`
  return (
    <article className={`rb-card rb-lift rb-row${flip ? ' rb-row--flip' : ''}`} id={id} aria-labelledby={headingId}>
      <div className="rb-row__text">
        <p className="rb-kicker">{kicker}</p>
        <h3 className="rb-row__title" id={headingId}>{title}</h3>
        <p className="rb-row__outcome">{outcome}</p>
        {chips.length ? (
          <ul className="rb-chips" aria-label={chipsLabel}>
            {chips.map((chip) => (
              <li key={chip.label}>
                <button
                  type="button"
                  className="rb-chip"
                  aria-haspopup="dialog"
                  onClick={() => onOpen?.(chip.opens)}
                >
                  {chip.label}
                </button>
              </li>
            ))}
          </ul>
        ) : null}
        {children}
        {cta ? (
          <div>
            <button type="button" className="button button--flex rb-btn" aria-haspopup="dialog" onClick={() => onOpen?.(cta.opens)}>
              {cta.label}
            </button>
          </div>
        ) : null}
      </div>
      <div className="rb-row__media">
        {media.src ? (
          <Picture
            src={media.src}
            alt={media.alt}
            width={media.width}
            height={media.height}
            eager={eager}
            style={media.position ? { objectPosition: media.position } : undefined}
          />
        ) : (
          <div className="rb-row__diagram" aria-hidden="true">
            <i className={`uil ${media.icon}`} aria-hidden="true" />
          </div>
        )}
        {media.badge ? <span className="rb-badge">{media.badge}</span> : null}
      </div>
    </article>
  )
}
