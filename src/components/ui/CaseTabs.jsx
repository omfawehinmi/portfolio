import { useId, useRef, useState } from 'react'

export default function CaseTabs({ tabs }) {
  const [active, setActive] = useState(tabs[0]?.id)
  const refs = useRef({})
  const uid = useId()

  const current = tabs.find((tab) => tab.id === active) ?? tabs[0]

  const onKeyDown = (event) => {
    const i = tabs.findIndex((tab) => tab.id === current.id)
    let next = null
    if (event.key === 'ArrowRight') next = (i + 1) % tabs.length
    else if (event.key === 'ArrowLeft') next = (i - 1 + tabs.length) % tabs.length
    else if (event.key === 'Home') next = 0
    else if (event.key === 'End') next = tabs.length - 1
    if (next === null) return
    event.preventDefault()
    setActive(tabs[next].id)
    refs.current[tabs[next].id]?.focus()
  }

  return (
    <div className="case-tabs">
      <div className="case-tabs__list" role="tablist" onKeyDown={onKeyDown}>
        {tabs.map((tab) => {
          const selected = tab.id === current.id
          return (
            <button
              key={tab.id}
              ref={(el) => {
                refs.current[tab.id] = el
              }}
              id={`${uid}-tab-${tab.id}`}
              type="button"
              role="tab"
              aria-selected={selected}
              aria-controls={`${uid}-panel`}
              tabIndex={selected ? 0 : -1}
              className={`case-tabs__tab${selected ? ' is-active' : ''}`}
              onClick={() => setActive(tab.id)}
            >
              {tab.label}
            </button>
          )
        })}
      </div>
      <div className="case-tabs__panel" role="tabpanel" id={`${uid}-panel`} aria-labelledby={`${uid}-tab-${current.id}`}>
        {current.content}
      </div>
    </div>
  )
}
