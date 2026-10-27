import Picture from './ui/Picture.jsx'

export default function ProjectCard({ id, kicker, title, outcome, thumb, chips = [], onOpen }) {
  const headingId = `card-${id}-h`
  return (
    <article className="rb-card rb-lift rb-pcard" aria-labelledby={headingId}>
      <div className="rb-pcard__thumb">
        <Picture
          src={thumb.src}
          width={thumb.width}
          height={thumb.height}
          style={thumb.position ? { objectPosition: thumb.position } : undefined}
        />
      </div>
      <p className="rb-kicker">{kicker}</p>
      <h3 className="rb-pcard__title" id={headingId}>
        <button type="button" className="rb-pcard__open" aria-haspopup="dialog" onClick={() => onOpen?.(id)}>
          {title}
        </button>
      </h3>
      <p>{outcome}</p>
      {chips.length ? (
        <ul className="rb-chips" aria-label={`${title} components`}>
          {chips.map((chip) => (
            <li key={chip.opens}>
              <button type="button" className="rb-chip" aria-haspopup="dialog" onClick={() => onOpen?.(chip.opens)}>
                {chip.label}
              </button>
            </li>
          ))}
        </ul>
      ) : null}
    </article>
  )
}
