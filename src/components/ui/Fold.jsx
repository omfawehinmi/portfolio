import { useState } from 'react'
import Icon from './Icon.jsx'

export default function Fold({ title, children, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen)

  return (
    <div className={`fold${open ? ' is-open' : ''}`}>
      <button type="button" className="fold__toggle" aria-expanded={open} onClick={() => setOpen((value) => !value)}>
        <span>{title}</span>
        <Icon name="uil-angle-down" className="fold__icon" />
      </button>
      {open ? <div className="fold__body">{children}</div> : null}
    </div>
  )
}
