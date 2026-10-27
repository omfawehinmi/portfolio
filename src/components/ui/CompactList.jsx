export function CompactRow({ icon, title, line, href, onClick, external }) {
  const inner = (
    <>
      <i className={`uil ${icon} rb-row-c__icon`} aria-hidden="true" />
      <span className="rb-row-c__text">
        <span className="rb-row-c__title">{title}</span>
        {line ? <span className="rb-row-c__line">{line}</span> : null}
      </span>
    </>
  )
  if (href) {
    return (
      <a className="rb-card rb-lift rb-row-c" href={href} {...(external ? { target: '_blank', rel: 'noreferrer' } : {})}>
        {inner}
      </a>
    )
  }
  if (onClick) {
    return (
      <button type="button" className="rb-card rb-lift rb-row-c" aria-haspopup="dialog" onClick={onClick}>
        {inner}
      </button>
    )
  }
  return <div className="rb-card rb-row-c">{inner}</div>
}

export default function CompactList({ items, label, className = '' }) {
  return (
    <ul className={`rb-clist ${className}`} aria-label={label}>
      {items.map((item) => (
        <li key={item.key || item.title}>
          <CompactRow {...item} />
        </li>
      ))}
    </ul>
  )
}
