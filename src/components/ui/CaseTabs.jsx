import { useState } from 'react'

export default function CaseTabs({ tabs }) {
  const [active, setActive] = useState(tabs[0]?.id)

  const current = tabs.find((tab) => tab.id === active) ?? tabs[0]

  return (
    <div className="case-tabs">
      <div className="case-tabs__list" role="tablist">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            type="button"
            role="tab"
            aria-selected={tab.id === current.id}
            className={`case-tabs__tab${tab.id === current.id ? ' is-active' : ''}`}
            onClick={() => setActive(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div className="case-tabs__panel" role="tabpanel">
        {current.content}
      </div>
    </div>
  )
}
