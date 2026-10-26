import { useEffect } from 'react'
import Icon from './Icon.jsx'

export default function Modal({ open, title, onClose, children, wide = false }) {
  useEffect(() => {
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
  }, [open, onClose])

  return (
    <div
      className={`services__modal${open ? ' active-modal' : ''}`}
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose()
      }}
    >
      <div className={`services__modal-content case-modal${wide ? ' case-modal--wide' : ''}`} role="dialog" aria-modal="true" aria-label={typeof title === 'string' ? title : 'Project details'}>
        <h4 className="services__modal-title">{title}</h4>
        <button type="button" className="services__modal-close" onClick={onClose} aria-label="Close">
          <Icon name="uil-times" />
        </button>
        <div className="services__modal-services">{children}</div>
      </div>
    </div>
  )
}
