import { useId, useRef, useState } from 'react'

export default function CaseTabs({ tabs }) {
  const [active, setActive] = useState(tabs[0]?.id)
  const base = useId().replace(/[^a-zA-Z0-9]/g, '')
  const refs = useRef({})

  const current = tabs.find((tab) => tab.id === active) ?? tabs[0]
  const tabId = (id) => `${base}-tab-${id}`
  const panelId = (id) => `${base}-panel-${id}`

  const onKeyDown = (event) => {
    const index = tabs.findIndex((tab) => tab.id === current.id)
    let next = null
    if (event.key === 'ArrowRight' || event.key === 'ArrowDown') next = (index + 1) % tabs.length
    if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') next = (index - 1 + tabs.length) % tabs.length
    if (event.key === 'Home') next = 0
    if (event.key === 'End') next = tabs.length - 1
    if (next === null) return
    event.preventDefault()
    setActive(tabs[next].id)
    refs.current[tabs[next].id]?.focus()
  }

  return (
    <div className="case-tabs">
      <div className="case-tabs__list" role="tablist" onKeyDown={onKeyDown}>
        {tabs.map((tab) => (
          <button
            key={tab.id}
            ref={(el) => {
              refs.current[tab.id] = el
            }}
            id={tabId(tab.id)}
            type="button"
            role="tab"
            aria-selected={tab.id === current.id}
            aria-controls={panelId(tab.id)}
            tabIndex={tab.id === current.id ? 0 : -1}
            className={`case-tabs__tab${tab.id === current.id ? ' is-active' : ''}`}
            onClick={() => setActive(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div className="case-tabs__panel" role="tabpanel" id={panelId(current.id)} aria-labelledby={tabId(current.id)} tabIndex={0}>
        {current.content}
      </div>
    </div>
  )
}
