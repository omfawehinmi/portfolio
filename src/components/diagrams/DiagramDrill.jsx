import { useState } from 'react'
import ArchitectureDiagram from './ArchitectureDiagram.jsx'
import { DRILL_AREAS, DRILL_ITEMS, drillArea, drillItem } from '../../data/architectureDrill.js'

/**
 * Layered navigate-in: high-level area map → pick an area → pick a feed/DB/WH/tool/workflow
 * → full diagram + description + metadata. Selection actually swaps the SVG.
 */
export default function DiagramDrill({
  tall = false,
  inst = 'open',
  startArea = null,
  compact = false,
  onInteract,
}) {
  const [areaId, setAreaId] = useState(startArea)
  const [itemId, setItemId] = useState(() => {
    if (!startArea) return null
    return drillArea(startArea).defaultItem
  })

  const area = areaId ? drillArea(areaId) : null
  const items = areaId ? DRILL_ITEMS[areaId] || [] : []
  const item = areaId && itemId ? drillItem(areaId, itemId) : null

  const stop = (event) => {
    event.preventDefault()
    event.stopPropagation()
    onInteract?.(event)
  }

  const openArea = (event, id) => {
    stop(event)
    setAreaId(id)
    setItemId(drillArea(id).defaultItem)
  }

  const pickItem = (event, id) => {
    stop(event)
    setItemId(id)
  }

  const backToMap = (event) => {
    stop(event)
    setAreaId(null)
    setItemId(null)
  }

  // Layer 0 — clickable high-level labels
  if (!areaId) {
    return (
      <div className={`bx-drill${compact ? ' bx-drill--compact' : ''}`}>
        <figure className="bx-fig bx-fig-drill">
          <ArchitectureDiagram kind="drill-map" tall={tall} inst={inst} />
        </figure>
        <p className="bx-drill__hint">Click an area to open its diagrams, description, and detail.</p>
        <ul className="bx-drill__map" role="list">
          {DRILL_AREAS.map((entry) => (
            <li key={entry.id}>
              <button type="button" className="bx-drill__entry" onClick={(event) => openArea(event, entry.id)}>
                <strong>{entry.label}</strong>
                <span>{entry.mapHint}</span>
              </button>
            </li>
          ))}
        </ul>
      </div>
    )
  }

  // Layer 1 — area picker + selected item diagram/detail
  return (
    <div className={`bx-drill bx-drill--open${compact ? ' bx-drill--compact' : ''}`}>
      <div className="bx-drill__nav">
        <button type="button" className="bx-drill__back" onClick={backToMap}>
          ← Platform map
        </button>
        <p className="bx-drill__area">{area.label}</p>
      </div>

      <div className="bx-dbtabs bx-drill__tabs" role="tablist" aria-label={area.label}>
        {items.map((entry) => (
          <button
            key={entry.id}
            type="button"
            role="tab"
            aria-selected={item?.id === entry.id}
            className={`bx-dbtab${item?.id === entry.id ? ' is-active' : ''}`}
            onClick={(event) => pickItem(event, entry.id)}
          >
            <strong>{entry.label}</strong>
            <span>{entry.blurb}</span>
          </button>
        ))}
      </div>

      {item ? (
        <>
          <figure className="bx-fig bx-fig-drill">
            <ArchitectureDiagram
              key={`${areaId}-${item.id}-${inst}`}
              kind="drill-item"
              tall={tall}
              inst={`${inst}-${item.id}`}
              drillItem={item}
            />
            <figcaption>{item.blurb}</figcaption>
          </figure>
          <p className="bx-lede">{item.detail}</p>
          {item.meta?.length ? (
            <dl className="bx-drill__meta">
              {item.meta.map((row) => (
                <div key={row.k}>
                  <dt>{row.k}</dt>
                  <dd>{row.v}</dd>
                </div>
              ))}
            </dl>
          ) : null}
        </>
      ) : null}
    </div>
  )
}

/** Compact tile strip: defaults to one area’s items with live swap (no map layer). */
export function DiagramDrillStrip({ areaId, tall = false, inst = 'tile' }) {
  const area = drillArea(areaId)
  const items = DRILL_ITEMS[areaId] || []
  const [itemId, setItemId] = useState(area.defaultItem)
  const item = drillItem(areaId, itemId)

  return (
    <div className="bx-drill bx-drill--strip" onClick={(event) => event.stopPropagation()}>
      <div className="bx-dbtabs bx-drill__tabs" role="tablist" aria-label={area.label}>
        {items.map((entry) => (
          <button
            key={entry.id}
            type="button"
            role="tab"
            aria-selected={item?.id === entry.id}
            className={`bx-dbtab${item?.id === entry.id ? ' is-active' : ''}`}
            onClick={(event) => {
              event.preventDefault()
              event.stopPropagation()
              setItemId(entry.id)
            }}
          >
            <strong>{entry.label}</strong>
          </button>
        ))}
      </div>
      <figure className="bx-fig">
        <ArchitectureDiagram
          key={`${areaId}-${item?.id}-${inst}`}
          kind="drill-item"
          tall={tall}
          inst={`${inst}-${item?.id}`}
          drillItem={item}
        />
      </figure>
    </div>
  )
}
