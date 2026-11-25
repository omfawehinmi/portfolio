import { useEffect, useRef } from 'react'
import Icon from './Icon.jsx'

export function canUseNativeDialog() {
  return typeof HTMLDialogElement !== 'undefined'
    && typeof HTMLDialogElement.prototype.showModal === 'function'
}

function ModalBody({ title, kicker, onClose, children }) {
  return (
    <>
      <button type="button" className="services__modal-close" onClick={onClose} aria-label="Close">
        <Icon name="uil-times" />
      </button>
      {kicker ? <p className="bx-kicker">{kicker}</p> : null}
      {title ? <h4 className="services__modal-title">{title}</h4> : null}
      <div className="services__modal-services">{children}</div>
    </>
  )
}

export default function Modal({ open, title, kicker, onClose, children, className }) {
  const dialogRef = useRef(null)
  const triggerRef = useRef(null)
  const native = canUseNativeDialog()

  useEffect(() => {
    if (!native) {
      if (!open) return undefined
      const onKeyDown = (event) => {
        if (event.key === 'Escape') onClose()
      }
      document.body.style.overflow = 'hidden'
      document.addEventListener('keydown', onKeyDown)
      return () => {
        document.body.style.overflow = 'auto'
        document.removeEventListener('keydown', onKeyDown)
      }
    }

    const el = dialogRef.current
    if (!el) return undefined

    if (open) {
      if (!el.open) {
        triggerRef.current = document.activeElement
        el.showModal()
      }
      const closeBtn = el.querySelector('.services__modal-close')
      if (closeBtn) closeBtn.focus()
      return undefined
    }

    if (el.open) el.close()
    return undefined
  }, [open, native, onClose])

  const returnFocus = () => {
    const trigger = triggerRef.current
    if (trigger && typeof trigger.focus === 'function') {
      trigger.focus()
    }
  }

  const handleNativeClose = () => {
    if (open) onClose()
    returnFocus()
  }

  if (native) {
    return (
      <dialog
        ref={dialogRef}
        className={`services__modal-content case-modal project-dialog${className ? ` ${className}` : ''}`}
        aria-label={typeof title === 'string' ? title : 'Project details'}
        onClose={handleNativeClose}
        onClick={(event) => {
          if (event.target === event.currentTarget) event.currentTarget.close()
        }}
      >
        <ModalBody title={title} kicker={kicker} onClose={() => dialogRef.current && dialogRef.current.close()}>
          {children}
        </ModalBody>
      </dialog>
    )
  }

  return (
    <div
      className={`services__modal${open ? ' active-modal' : ''}`}
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose()
      }}
    >
      <div className={`services__modal-content case-modal${className ? ` ${className}` : ''}`} role="dialog" aria-modal="true" aria-label={typeof title === 'string' ? title : 'Project details'}>
        <ModalBody title={title} kicker={kicker} onClose={onClose}>{children}</ModalBody>
      </div>
    </div>
  )
}
