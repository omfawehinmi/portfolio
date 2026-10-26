import { useEffect, useRef } from 'react'
import Icon from './Icon.jsx'

const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'

export function trapTab(event, container) {
  if (event.key !== 'Tab' || !container) return
  const items = [...container.querySelectorAll(FOCUSABLE)].filter((el) => !el.closest('[hidden]') && el.getAttribute('tabindex') !== '-1')
  if (!items.length) {
    event.preventDefault()
    return
  }
  const first = items[0]
  const last = items[items.length - 1]
  const active = container.ownerDocument.activeElement
  if (event.shiftKey && (active === first || !container.contains(active))) {
    event.preventDefault()
    last.focus()
  } else if (!event.shiftKey && (active === last || !container.contains(active))) {
    event.preventDefault()
    first.focus()
  }
}

export default function Modal({ open, title, onClose, children, wide = false }) {
  const dialogRef = useRef(null)
  const closeRef = useRef(null)
  const onCloseRef = useRef(onClose)
  onCloseRef.current = onClose

  useEffect(() => {
    if (!open) return undefined
    const opener = document.activeElement

    const onKeyDown = (event) => {
      if (event.key === 'Escape') onCloseRef.current()
      else trapTab(event, dialogRef.current)
    }

    document.body.style.overflow = 'hidden'
    document.addEventListener('keydown', onKeyDown)
    closeRef.current?.focus()
    return () => {
      document.body.style.overflow = 'auto'
      document.removeEventListener('keydown', onKeyDown)
      if (opener && typeof opener.focus === 'function' && opener.isConnected) opener.focus()
    }
  }, [open])

  return (
    <div
      className={`services__modal${open ? ' active-modal' : ''}`}
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose()
      }}
    >
      <div
        ref={dialogRef}
        className={`services__modal-content case-modal${wide ? ' case-modal--wide' : ''}`}
        role="dialog"
        aria-modal="true"
        aria-label={typeof title === 'string' ? title : 'Project details'}
      >
        <h4 className="services__modal-title">{title}</h4>
        <button ref={closeRef} type="button" className="services__modal-close" onClick={onClose} aria-label="Close">
          <Icon name="uil-times" />
        </button>
        <div className="services__modal-services">{children}</div>
      </div>
    </div>
  )
}
