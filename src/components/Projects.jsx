import { useEffect, useRef, useState } from 'react'
import { COPY, DIAG } from '../data/architecture.js'
import { intrinsicImageSize, publicUrl } from '../utils.js'
import ArchitectureDiagram from './diagrams/ArchitectureDiagram.jsx'
import McpMatrix from './McpMatrix.jsx'
import { PLATFORM } from './diagrams/architectureSvg.js'
import Icon from './ui/Icon.jsx'
import SectionHeader from './ui/SectionHeader.jsx'

/** Left jump rail: labeled anchors — names mirror what each block captures. */
const PROJECT_RAIL = [
  { id: 'lead', label: 'Household Snowflake + MCP' },
  { id: 'workflow', label: 'CI / CD ship path' },
  { id: 'pipes', label: 'Feeds → schemas' },
  { id: 'airflow', label: 'Airflow DAGs' },
  { id: 'dbt', label: 'dbt layers' },
  { id: 'terraform', label: 'Terraform · infra' },
  { id: 'domains', label: 'DBs · tables · models' },
  { id: 'mcp', label: 'MCP · agents · tools' },
  { id: 'security', label: 'Roles · WH · cost' },
  { id: 'features', label: 'Semantic views · Cortex Search' },
  { id: 'grokbot', label: 'Grokbot · per MCP' },
  { id: 'more', label: 'More work' },
]

const KICKERS = {
  mcp: 'Servers · agents · tools',
  workflow: 'CI / CD ship path',
  domains: 'Columns · tables · modeling',
  airflow: 'Scheduler inventory',
  dbt: 'Transform layers · triggers',
  terraform: 'Infrastructure · apply · recovery',
  pipes: 'Source feeds → schemas',
  security: 'Roles · warehouses · cost',
  features: 'Semantic views · Cortex Search',
  grokbot: 'Agents per MCP server',
}

const MORE_PREVIEW = {
  'hr-intelligence': { src: publicUrl('assets/img/bento/hr-intelligence.jpg'), sample: false },
  myorglab: { src: publicUrl('assets/img/bento/myorglab.jpg'), sample: false },
  'sof-portal': { src: publicUrl('assets/img/bento/sof-portal.jpg'), sample: true },
  /* Full assembly still + Blender punch clip (repo assets — not the cropped bento crop alone). */
  palms: {
    src: publicUrl('assets/img/palms-punch.png'),
    sample: false,
    fit: true,
    video: publicUrl('assets/video/hybrid-punches.mp4'),
  },
  camera: { src: publicUrl('assets/img/bento/camera.jpg'), sample: false },
  inpainting: { src: publicUrl('assets/img/bento/inpainting.jpg'), sample: false },
}

/** Live MyOrgLab marketing + app shots (from Project media / Cloudflare preview). */
const MYORGLAB_INDUSTRIES = [
  { id: 'healthcare', label: 'Healthcare', src: publicUrl('assets/img/myorglab/industry-healthcare.png') },
  { id: 'logistics', label: 'Logistics', src: publicUrl('assets/img/myorglab/industry-logistics.png') },
  { id: 'legal', label: 'Legal', src: publicUrl('assets/img/myorglab/industry-legal.png') },
  { id: 'finance', label: 'Finance', src: publicUrl('assets/img/myorglab/industry-finance.png') },
  { id: 'public-sector', label: 'Public Sector', src: publicUrl('assets/img/myorglab/industry-public-sector.png') },
]

const MYORGLAB_APPS = [
  { id: 'healthcare', label: 'Healthcare app', src: publicUrl('assets/img/myorglab/app-healthcare.png') },
  { id: 'logistics', label: 'Logistics app', src: publicUrl('assets/img/myorglab/app-logistics.png') },
  { id: 'legal', label: 'Legal app', src: publicUrl('assets/img/myorglab/app-legal.png') },
  { id: 'finance', label: 'Finance app', src: publicUrl('assets/img/myorglab/app-finance.png') },
  { id: 'public-sector', label: 'Public Sector app', src: publicUrl('assets/img/myorglab/app-public-sector.png') },
]

const OPEN_ALIASES = {
  lead: 'lead',
  mcp: 'mcp',
  workflow: 'workflow',
  domains: 'domains',
  airflow: 'airflow',
  dbt: 'dbt',
  terraform: 'terraform',
  pipes: 'pipes',
  security: 'security',
  features: 'features',
  grokbot: 'grokbot',
  stack: 'mcp',
  mychart: 'domains',
  myorglab: 'more-myorglab',
  'semantic-hr': 'more-hr-intelligence',
  'cortex-self-serve': 'more-hr-intelligence',
  sof: 'more-sof-portal',
  'camera-tracking': 'more-camera',
  'palms-bag': 'more-palms',
  inpainting: 'more-inpainting',
}

const moreItems = COPY.more.map((item) => ({
  ...item,
  id: `more-${item.prev}`,
  outcome: item.line,
}))

function resolveOpen(openId) {
  if (!openId) return { kind: null }
  const mapped = OPEN_ALIASES[openId] || openId
  if (mapped === 'lead') return { kind: null }
  const tile = COPY.tiles.find((item) => item.id === mapped)
  if (tile) return { kind: 'tile', tile }
  const more = moreItems.find((item) => item.id === mapped)
  if (more) return { kind: 'more', more }
  return { kind: null }
}

function useNarrow() {
  const [narrow, setNarrow] = useState(() =>
    typeof window !== 'undefined' ? window.matchMedia('(max-width: 767px)').matches : false,
  )

  useEffect(() => {
    const media = window.matchMedia('(max-width: 767px)')
    const onChange = () => setNarrow(media.matches)
    onChange()
    media.addEventListener('change', onChange)
    return () => media.removeEventListener('change', onChange)
  }, [])

  return narrow
}

/** Jump rail only when a reserved left column fits without colliding with tiles. */
function useRailVisible() {
  const [visible, setVisible] = useState(() =>
    typeof window !== 'undefined' ? window.matchMedia('(min-width: 1280px)').matches : false,
  )

  useEffect(() => {
    const media = window.matchMedia('(min-width: 1280px)')
    const onChange = () => setVisible(media.matches)
    onChange()
    media.addEventListener('change', onChange)
    return () => media.removeEventListener('change', onChange)
  }, [])

  return visible
}

function Chips({ tags }) {
  if (!tags?.length) return null
  return (
    <ul className="bx-chips">
      {tags.map((tag) => (
        <li key={tag}>{tag}</li>
      ))}
    </ul>
  )
}

const COLLAPSE_CHARS = 160

function FoldBody({ project, className = '', lines = 3, expandedExtra = null }) {
  const text = project?.outcome || ''
  const canCollapse = text.length > COLLAPSE_CHARS || Boolean(expandedExtra)
  const [open, setOpen] = useState(false)

  return (
    <div className={`project-tile__fold${canCollapse && !open ? ' is-collapsed' : ''}`}>
      <p
        className={`project-tile__outcome bx-body${className ? ` ${className}` : ''}${
          canCollapse && !open ? ' is-clamped' : ''
        }`}
        style={canCollapse && !open ? { WebkitLineClamp: lines } : undefined}
      >
        {text}
      </p>
      {canCollapse && open && expandedExtra ? (
        <div className="bx-fold-extra">{expandedExtra}</div>
      ) : null}
      {canCollapse ? (
        <button
          type="button"
          className="bx-moretoggle"
          aria-expanded={open}
          onClick={(event) => {
            event.preventDefault()
            event.stopPropagation()
            setOpen((value) => !value)
          }}
        >
          {open ? 'Show less' : 'Show more'}
        </button>
      ) : null}
    </div>
  )
}

function ExpandToggle({ expanded, onToggle, label, text = null }) {
  return (
    <button
      type="button"
      className={`bx-open${expanded ? ' is-expanded' : ''}${text ? ' bx-open--text' : ''}`}
      aria-expanded={expanded}
      aria-label={`${expanded ? 'Collapse' : 'Expand'} ${label}`}
      onClick={(event) => {
        event.stopPropagation()
        onToggle()
      }}
    >
      <Icon name={expanded ? 'uil-compress-arrows' : 'uil-expand-arrows-alt'} />
      {text ? <span>{expanded ? 'Collapse' : text}</span> : null}
    </button>
  )
}

function BulbIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M9 18h6" />
      <path d="M10 22h4" />
      <path d="M12 2a7 7 0 0 0-4 12.7c.6.5 1 1.3 1 2.1V17h6v-.2c0-.8.4-1.6 1-2.1A7 7 0 0 0 12 2z" />
    </svg>
  )
}

function DesignChoices({ title, choices }) {
  const [open, setOpen] = useState(false)
  const rootRef = useRef(null)

  useEffect(() => {
    if (!open) return undefined
    const onDown = (event) => {
      if (!rootRef.current?.contains(event.target)) setOpen(false)
    }
    const onKey = (event) => {
      if (event.key === 'Escape') setOpen(false)
    }
    document.addEventListener('click', onDown, true)
    document.addEventListener('keydown', onKey, true)
    return () => {
      document.removeEventListener('click', onDown, true)
      document.removeEventListener('keydown', onKey, true)
    }
  }, [open])

  if (!choices?.length) return null
  return (
    <div
      className="bx-choices"
      ref={rootRef}
      onClick={(event) => event.stopPropagation()}
      onKeyDown={(event) => event.stopPropagation()}
    >
      {open ? (
        <div className="bx-choices__card" role="region" aria-label={`Design choices: ${title}`}>
          <p className="bx-choices__title">Design choices · {title}</p>
          <ul>
            {choices.map((choice) => (
              <li key={choice.t}>
                <strong>{choice.t}</strong>
                <span>{choice.d}</span>
              </li>
            ))}
          </ul>
          <p className="bx-choices__src">From household-data-architecture design decisions</p>
        </div>
      ) : null}
      <button
        type="button"
        className={`bx-choices__btn${open ? ' is-open' : ''}`}
        aria-expanded={open}
        aria-label={`Design choices for ${title}`}
        onClick={() => setOpen((value) => !value)}
      >
        <BulbIcon />
        <span>Design choices</span>
      </button>
    </div>
  )
}

function PreviewMedia({ item, big = false }) {
  const preview = MORE_PREVIEW[item.prev]
  if (!preview) return null
  const size = intrinsicImageSize(preview.src)
  const alt = `${item.title} preview${preview.sample ? ', sample data' : ''}`
  const fitClass = preview.fit || preview.video ? ' bx-media--fit' : ''
  return (
    <div className={`bx-media${big ? ' bx-big' : ''}${fitClass}`}>
      {preview.video ? (
        <video
          className="bx-media__video"
          src={preview.video}
          poster={preview.src}
          controls={big}
          autoPlay={!big}
          muted
          loop
          playsInline
          preload="metadata"
          aria-label={alt}
          width={size?.width}
          height={size?.height}
        />
      ) : (
        <img alt={alt} src={preview.src} width={size?.width} height={size?.height} />
      )}
    </div>
  )
}

function tileHandlers(expanded, id, onOpen) {
  if (expanded) return {}
  return {
    role: 'button',
    tabIndex: 0,
    'aria-expanded': false,
    onClick: () => onOpen(id),
    onKeyDown: (event) => {
      if (event.target !== event.currentTarget) return
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault()
        onOpen(id)
      }
    },
  }
}

function useProjectRailActive(ids) {
  const [activeId, setActiveId] = useState(ids[0] || null)

  useEffect(() => {
    const nodes = ids
      .map((id) => document.getElementById(`project-${id}`))
      .filter(Boolean)
    if (!nodes.length) return undefined

    const visible = new Map()
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          const key = entry.target.id.replace(/^project-/, '')
          if (entry.isIntersecting) visible.set(key, entry.intersectionRatio)
          else visible.delete(key)
        })
        let best = null
        let bestRatio = -1
        visible.forEach((ratio, key) => {
          if (ratio > bestRatio) {
            best = key
            bestRatio = ratio
          }
        })
        if (best) setActiveId(best)
      },
      { rootMargin: '-35% 0px -45% 0px', threshold: [0, 0.15, 0.35, 0.55, 0.75, 1] },
    )
    nodes.forEach((node) => observer.observe(node))
    return () => observer.disconnect()
  }, [ids])

  return activeId
}

function ProjectsJumpRail({ items, activeId }) {
  return (
    <nav className="bx-rail" aria-label="Projects sections">
      <ol className="bx-rail__list">
        {items.map((item) => (
          <li key={item.id}>
            <a
              href={`#project-${item.id}`}
              className={`bx-rail__link${activeId === item.id ? ' is-active' : ''}`}
              aria-current={activeId === item.id ? 'true' : undefined}
            >
              <span className="bx-rail__dot" aria-hidden="true" />
              <span className="bx-rail__label">{item.label}</span>
            </a>
          </li>
        ))}
      </ol>
    </nav>
  )
}

function LeadTile({ tall }) {
  const L = COPY.lead
  return (
    <article id="project-lead" className="bx-tile bx-lead bx-tile--hero" data-testid="project-tile">
      <div className="bx-leadtop">
        <div>
          <p className="bx-kicker">Lead project</p>
          <h3 className="bx-h1">{L.title}</h3>
          {L.subhead ? <p className="bx-subhead">{L.subhead}</p> : null}
        </div>
      </div>
      <div className="bx-leadbody">
        <div className="bx-leadtext">
          <StatStrip stats={L.stats} />
          <FoldBody project={{ outcome: L.intro }} lines={3} />
        </div>
        <figure className="bx-fig bx-leadfig">
          <ArchitectureDiagram kind="lead" tall={tall} inst="tile" />
        </figure>
      </div>
    </article>
  )
}

function ShipPathChips({ value, onChange }) {
  return (
    <div className="bx-shipchips" role="tablist" aria-label="Release path by platform">
      {DIAG.workflow.platforms.map((p) => (
        <button
          key={p.id}
          type="button"
          role="tab"
          aria-selected={value === p.id}
          className={`bx-shipchip${value === p.id ? ' is-active' : ''}`}
          onClick={(event) => {
            event.stopPropagation()
            onChange(p.id)
          }}
        >
          {PLATFORM[p.logo] ? <img className="bx-shipchip-logo" src={PLATFORM[p.logo].src} alt="" /> : null}
          {p.label}
        </button>
      ))}
    </div>
  )
}

function TileFigure({ tile, tall, inst, ship, onShip, dbId, big = false, caption = null }) {
  const variant = tile.id === 'workflow' ? ship : tile.id === 'domains' ? dbId : null
  return (
    <figure className={`bx-fig${big ? ' bx-fig-big' : ''}${tile.id === 'domains' ? ' bx-fig--lanes' : ''}`}>
      {tile.id === 'workflow' ? <ShipPathChips value={ship} onChange={onShip} /> : null}
      {tile.id === 'mcp' ? <McpMatrix /> : <ArchitectureDiagram kind={tile.id} tall={tall} inst={inst} variant={variant} />}
      {caption ? <figcaption>{caption}</figcaption> : null}
    </figure>
  )
}

function ArchitectureTile({ tile, onOpen, tall, expanded }) {
  const [ship, setShip] = useState(DIAG.workflow.platforms[0].id)
  const [dbId, setDbId] = useState(tile.databases?.[0]?.id || null)
  const showRowsOnTile = tile.id === 'grokbot' || tile.id === 'security'
  const isHero = tile.id === 'mcp' || tile.id === 'domains' || tile.id === 'grokbot'
  return (
    <article
      id={`project-${tile.id}`}
      className={`bx-tile bx-t-${tile.id}${isHero ? ' bx-tile--hero' : ''}${expanded ? ' is-expanded' : ''}`}
      data-testid="project-tile"
      {...tileHandlers(expanded, tile.id, onOpen)}
    >
      <div className="bx-head">
        <div>
          <p className="bx-kicker">{KICKERS[tile.id]}</p>
          <h4 className="bx-h2">{tile.title}</h4>
        </div>
        <ExpandToggle
          expanded={expanded}
          label={tile.title}
          text={tile.id === 'domains' ? 'Expand for ERD' : null}
          onToggle={() => onOpen(expanded ? null : tile.id)}
        />
      </div>
      {expanded ? (
        <TilePanel tile={tile} tall={tall} ship={ship} onShip={setShip} dbId={dbId} onDb={setDbId} />
      ) : (
        <>
          <FoldBody project={{ outcome: tile.body }} lines={isHero ? 4 : 3} />
          {showRowsOnTile ? <RowTable rows={tile.rows} collapsed /> : null}
          {tile.id === 'domains' && tile.databases?.length ? (
            <div className="bx-dbtabs bx-dbtabs--tile" role="tablist" aria-label="Snowflake databases">
              {tile.databases.map((db) => (
                <button
                  key={db.id}
                  type="button"
                  role="tab"
                  aria-selected={dbId === db.id}
                  className={`bx-dbtab${dbId === db.id ? ' is-active' : ''}`}
                  onClick={(event) => {
                    event.stopPropagation()
                    setDbId(db.id)
                  }}
                >
                  <strong>{db.name}</strong>
                  <span>{db.status}</span>
                </button>
              ))}
            </div>
          ) : null}
          <TileFigure tile={tile} tall={tall} inst="tile" ship={ship} onShip={setShip} dbId={dbId} />
        </>
      )}
      <DesignChoices title={tile.title} choices={tile.choices} />
    </article>
  )
}

function MoreCard({ item, featured, onOpen, expanded }) {
  const isMyOrgLab = item.prev === 'myorglab'
  const industries = isMyOrgLab ? COPY.industries : null
  const industryExtra = industries ? (
    <div className="bx-fold-industries">
      <p className="bx-industries">
        {industries.line}{' '}
        <a
          href={industries.href}
          onClick={(event) => event.stopPropagation()}
          target="_blank"
          rel="noreferrer"
        >
          {industries.link}
        </a>
      </p>
      <ShotStrip title="Industry pages" shots={MYORGLAB_INDUSTRIES} />
    </div>
  ) : null
  // Option 3: full-width featured strip + uniform media-top cards in an even grid.
  return (
    <article
      id={`project-${item.id}`}
      className={`bx-tile bx-card ${featured ? 'bx-more-hero' : 'bx-more-card'}${isMyOrgLab ? ' bx-card--myorglab' : ''}${
        item.prev === 'palms' ? ' bx-card--palms' : ''
      }${expanded ? ' is-expanded' : ''}`}
      data-testid="project-tile"
      {...tileHandlers(expanded, item.id, onOpen)}
    >
      {expanded ? (
        <div className="bx-cardtext bx-cardtext--open">
          <h4 className="bx-h3">{item.title}</h4>
          <MorePanel item={item} />
        </div>
      ) : (
        <>
          <PreviewMedia item={item} big={featured} />
          <div className="bx-cardtext">
            <h4 className="bx-h3">{item.title}</h4>
            <FoldBody project={item} expandedExtra={industryExtra} lines={isMyOrgLab ? 3 : 3} />
          </div>
        </>
      )}
      <ExpandToggle expanded={expanded} label={item.title} onToggle={() => onOpen(expanded ? null : item.id)} />
    </article>
  )
}

function StatStrip({ stats }) {
  if (!stats?.length) return null
  return (
    <ul className="bx-stats">
      {stats.map((stat) => (
        <li key={stat.label}>
          <strong>{stat.n}</strong>
          <span>{stat.label}</span>
        </li>
      ))}
    </ul>
  )
}

function RowTable({ rows, collapsed = false, previewCount = 2 }) {
  const [open, setOpen] = useState(false)
  if (!rows?.length) return null
  const canCollapse = collapsed && rows.length > previewCount
  const visible = canCollapse && !open ? rows.slice(0, previewCount) : rows
  return (
    <div className="bx-rows">
      {visible.map((row) => (
        <div className="bx-rowline" key={row.name}>
          <code>{row.name}</code>
          <p>{row.role}</p>
          {row.schedule ? <span className="bx-rowmeta">{row.schedule}</span> : null}
          {row.status ? <span className="bx-rowstatus">{row.status}</span> : null}
        </div>
      ))}
      {canCollapse ? (
        <button
          type="button"
          className="bx-moretoggle"
          aria-expanded={open}
          onClick={(event) => {
            event.preventDefault()
            event.stopPropagation()
            setOpen((value) => !value)
          }}
        >
          {open ? 'Show less' : `Show ${rows.length - previewCount} more`}
        </button>
      ) : null}
    </div>
  )
}

function DomainsPanel({ tile, tall, dbId, setDbId }) {
  const databases = tile.databases || []
  const active = databases.find((db) => db.id === dbId) || databases[0]
  return (
    <div className="bx-panel-body bx-panel-body--erd">
      <div className="bx-dbtabs" role="tablist" aria-label="Snowflake databases">
        {databases.map((db) => (
          <button
            key={db.id}
            type="button"
            role="tab"
            aria-selected={active?.id === db.id}
            className={`bx-dbtab${active?.id === db.id ? ' is-active' : ''}`}
            onClick={() => setDbId(db.id)}
          >
            <strong>{db.name}</strong>
            <span>{db.status}</span>
          </button>
        ))}
      </div>
      <figure className="bx-fig bx-fig-big bx-fig--erd">
        <ArchitectureDiagram kind="domains" tall={tall} inst="open" dbId={active?.id} />
        <figcaption>{active ? `${active.name}: ${active.oneLine}` : tile.caption}</figcaption>
      </figure>
      {/* ERD + short oneLine caption only — no long detail/body dump */}
    </div>
  )
}

function TilePanel({ tile, tall, ship, onShip, dbId, onDb }) {
  if (tile.id === 'domains') return <DomainsPanel tile={tile} tall={tall} dbId={dbId} setDbId={onDb} />
  if (tile.id === 'workflow')
    return (
      <div className="bx-panel-body">
        <p>{tile.body}</p>
        <TileFigure tile={tile} tall={tall} inst="open" ship={ship} onShip={onShip} big caption={tile.caption} />
        <RowTable rows={tile.rows} />
      </div>
    )
  return (
    <div className="bx-panel-body">
      <TileFigure tile={tile} tall={tall} inst="open" ship={ship} onShip={onShip} big caption={tile.caption} />
      <p>{tile.body}</p>
      <RowTable rows={tile.rows} />
    </div>
  )
}

function ShotStrip({ title, shots }) {
  if (!shots?.length) return null
  return (
    <div className="bx-shots">
      <p className="bx-shots__label">{title}</p>
      <ul className="bx-shots__list">
        {shots.map((shot) => {
          const size = intrinsicImageSize(shot.src)
          return (
            <li key={shot.id}>
              <img
                src={shot.src}
                alt={shot.label}
                width={size?.width}
                height={size?.height}
                loading="lazy"
              />
              <span>{shot.label}</span>
            </li>
          )
        })}
      </ul>
    </div>
  )
}

function MorePanel({ item }) {
  const showMyOrgLabShots = item.prev === 'myorglab'
  return (
    <div className="bx-panel-body">
      <PreviewMedia item={item} big />
      <p>{item.line}</p>
      <Chips tags={item.tags} />
      {showMyOrgLabShots ? (
        <>
          <ShotStrip title="Industry pages" shots={MYORGLAB_INDUSTRIES} />
          <ShotStrip title="App surfaces" shots={MYORGLAB_APPS} />
        </>
      ) : null}
    </div>
  )
}

export default function Projects({ openId, onOpen }) {
  const tall = useNarrow()
  const railVisible = useRailVisible()
  const open = resolveOpen(openId)
  const railIds = PROJECT_RAIL.map((item) => item.id)
  const activeRail = useProjectRailActive(railVisible ? railIds : [])
  const expandedId = open.kind === 'tile' ? open.tile.id : open.kind === 'more' ? open.more.id : null

  useEffect(() => {
    if (!expandedId) return
    const el = document.getElementById(`project-${expandedId}`)
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }, [expandedId])

  return (
    <section className="projects section" id="projects">
      <SectionHeader title={COPY.section.title} subtitle={COPY.section.kicker} />

      {/* Rail padding applies only to this shell — title stays flush above content. */}
      <div className="bx-with-rail">
        {railVisible ? <ProjectsJumpRail items={PROJECT_RAIL} activeId={activeRail} /> : null}

        <div className="container bx">
          <p className="bx-intro">{COPY.section.intro}</p>
          <div className="bx-grid">
            <LeadTile tall={tall} />
            {COPY.tiles.map((tile) => (
              <ArchitectureTile key={tile.id} tile={tile} onOpen={onOpen} tall={tall} expanded={expandedId === tile.id} />
            ))}
          </div>
          <h3 id="project-more" className="bx-morehead">
            More shipped work
          </h3>
          <div className="bx-grid bx-more">
            {moreItems.map((item, index) => (
              <MoreCard key={item.id} item={item} featured={index === 0} onOpen={onOpen} expanded={expandedId === item.id} />
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}
