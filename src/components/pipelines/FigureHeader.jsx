export default function FigureHeader({ eyebrow, title, status, titleId, ariaLabel }) {
  return (
    <header className="pl-figure__header">
      <p className="pl-eyebrow">{eyebrow}</p>
      <figcaption className="pl-figure__title" id={titleId}>
        {title} <span className="visually-hidden">: {ariaLabel}</span>
      </figcaption>
      {status ? <p className="pl-figure__status">Status: {status}</p> : null}
    </header>
  )
}
