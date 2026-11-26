import { useEffect, useState } from 'react'
import { COPY } from '../data/architecture.js'
import { projects } from '../data/projects.js'
import { intrinsicImageSize, publicUrl } from '../utils.js'
import ArchitectureDiagram from './diagrams/ArchitectureDiagram.jsx'
import Icon from './ui/Icon.jsx'
import Modal from './ui/Modal.jsx'
import SectionHeader from './ui/SectionHeader.jsx'

/** Left jump rail: labeled anchors — names mirror what each block captures. */
const PROJECT_RAIL = [
  { id: 'lead', label: 'Lead platform' },
  { id: 'workflow', label: 'CI / CD ship path' },
  { id: 'pipes', label: 'Feeds → schemas' },
  { id: 'airflow', label: 'Airflow DAGs' },
  { id: 'dbt', label: 'dbt transforms' },
  { id: 'domains', label: 'DBs · tables · models' },
  { id: 'mcp', label: 'MCP · agents · tools' },
  { id: 'security', label: 'Roles · WH · cost' },
  { id: 'features', label: 'SV_* · CS_* catalogs' },
  { id: 'grokbot', label: 'Grokbot · per MCP' },
  { id: 'more', label: 'More work' },
]

const KICKERS = {
  mcp: 'Servers · agents · tools',
  workflow: 'CI / CD ship path',
  domains: 'Columns · tables · modeling',
  airflow: 'Scheduler inventory',
  dbt: 'Transform layers',
  pipes: 'Source feeds → schemas',
  security: 'Roles · warehouses · cost',
  features: 'SV_* · CS_* catalogs',
  grokbot: 'Agents per MCP server',
}

const LEAD_CAPTION =
  'As-built: MCP_HOUSEHOLD_LEDGER_GROK (read) and MCP_HOUSEHOLD_LEDGER (write) in front of SV_* / CS_* on HOUSEHOLD_LEDGER.'

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
  if (mapped === 'lead') return { kind: 'lead' }
  const tile = COPY.tiles.find((item) => item.id === mapped)
  if (tile) return { kind: 'tile', tile }
  const more = moreItems.find((item) => item.id === mapped)
  if (more) return { kind: 'more', more }
  const legacy = projects.find((item) => item.id === openId)
  if (legacy) return { kind: 'legacy', project: legacy }
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

function OpenIcon() {
  return (
    <span className="bx-open" aria-hidden="true">
      <Icon name="uil-expand-arrows-alt" />
    </span>
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

function openTile(event, id, onOpen) {
  onOpen(id)
}

function onTileKey(event, id, onOpen) {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault()
    onOpen(id)
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

function LeadTile({ onOpen, tall }) {
  const L = COPY.lead
  return (
    <article
      id="project-lead"
      className="bx-tile bx-lead bx-tile--hero"
      data-testid="project-tile"
      tabIndex={0}
      role="button"
      aria-haspopup="dialog"
      onClick={(event) => openTile(event, 'lead', onOpen)}
      onKeyDown={(event) => onTileKey(event, 'lead', onOpen)}
    >
      <div className="bx-leadtop">
        <div>
          <p className="bx-kicker">Lead project</p>
          <h3 className="bx-h1">{L.title}</h3>
          <p className="bx-subhead">{L.subhead}</p>
        </div>
        <OpenIcon />
      </div>
      <div className="bx-leadbody">
        <div className="bx-leadtext">
          <p className="bx-lede">{L.oneLine}</p>
          <FoldBody project={{ outcome: L.intro }} lines={3} />
          <StatStrip stats={L.stats} />
          <Chips tags={L.tags} />
        </div>
        <figure className="bx-fig bx-leadfig">
          <ArchitectureDiagram kind="lead" tall={tall} inst="tile" />
        </figure>
      </div>
    </article>
  )
}

function ArchitectureTile({ tile, onOpen, tall }) {
  const showRowsOnTile = tile.id === 'mcp' || tile.id === 'grokbot' || tile.id === 'security'
  const isHero = tile.id === 'mcp' || tile.id === 'domains' || tile.id === 'grokbot'
  return (
    <article
      id={`project-${tile.id}`}
      className={`bx-tile bx-t-${tile.id}${isHero ? ' bx-tile--hero' : ''}`}
      data-testid="project-tile"
      tabIndex={0}
      role="button"
      aria-haspopup="dialog"
      onClick={(event) => openTile(event, tile.id, onOpen)}
      onKeyDown={(event) => onTileKey(event, tile.id, onOpen)}
    >
      <div className="bx-head">
        <div>
          <p className="bx-kicker">{KICKERS[tile.id]}</p>
          <h4 className="bx-h2">{tile.title}</h4>
        </div>
        <OpenIcon />
      </div>
      <FoldBody project={{ outcome: tile.body }} lines={isHero ? 4 : 3} />
      {showRowsOnTile ? <RowTable rows={tile.rows} collapsed /> : null}
      {tile.id === 'domains' && tile.databases?.length ? (
        <ul className="bx-dbtabs bx-dbtabs--tile" aria-hidden="true">
          {tile.databases.map((db, i) => (
            <li key={db.id} className={`bx-dbtab${i === 0 ? ' is-active' : ''}`}>
              <strong>{db.name}</strong>
              <span>{db.status}</span>
            </li>
          ))}
        </ul>
      ) : null}
      <figure className="bx-fig">
        <ArchitectureDiagram kind={tile.id} tall={tall} inst="tile" />
      </figure>
    </article>
  )
}

function MoreCard({ item, featured, onOpen }) {
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
      className={`bx-tile bx-card ${featured ? 'bx-more-hero' : 'bx-more-card'}${isMyOrgLab ? ' bx-card--myorglab' : ''}${
        item.prev === 'palms' ? ' bx-card--palms' : ''
      }`}
      data-testid="project-tile"
      tabIndex={0}
      role="button"
      aria-haspopup="dialog"
      onClick={(event) => openTile(event, item.id, onOpen)}
      onKeyDown={(event) => onTileKey(event, item.id, onOpen)}
    >
      <PreviewMedia item={item} big={featured} />
      <div className="bx-cardtext">
        <h4 className="bx-h3">{item.title}</h4>
        <FoldBody project={item} expandedExtra={industryExtra} lines={isMyOrgLab ? 3 : 3} />
      </div>
      <OpenIcon />
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

function LeadPanel({ tall }) {
  const L = COPY.lead
  return (
    <div className="bx-panel-body">
      <p className="bx-subhead">{L.subhead}</p>
      <StatStrip stats={L.stats} />
      <figure className="bx-fig bx-leadfig bx-fig-big">
        <ArchitectureDiagram kind="lead" tall={tall} inst="open" />
        <figcaption>{LEAD_CAPTION}</figcaption>
      </figure>
      <p className="bx-lede">{L.oneLine}</p>
      <p>{L.intro}</p>
      <Chips tags={L.tags} />
    </div>
  )
}

function DomainsPanel({ tile, tall }) {
  const databases = tile.databases || []
  const [dbId, setDbId] = useState(databases[0]?.id || null)
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

function TilePanel({ tile, tall }) {
  if (tile.id === 'domains') return <DomainsPanel tile={tile} tall={tall} />
  return (
    <div className="bx-panel-body">
      <figure className="bx-fig bx-fig-big">
        <ArchitectureDiagram kind={tile.id} tall={tall} inst="open" />
        <figcaption>{tile.caption}</figcaption>
      </figure>
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

function LegacyBody({ project, onOpen }) {
  const parts = (project.parts || []).map((id) => projects.find((item) => item.id === id)).filter(Boolean)
  const parentItem = project.parentId ? projects.find((item) => item.id === project.parentId) : null

  return (
    <>
      {project.videoFile ? (
        <video className="case-video" src={project.videoFile} controls playsInline muted loop />
      ) : null}
      {project.video ? (
        <div className="media-well">
          <iframe
            src={project.video}
            title={project.title}
            allow="autoplay; fullscreen; picture-in-picture; clipboard-write"
          />
        </div>
      ) : null}
      {project.thumb && !project.videoFile && !project.video ? (
        <img
          src={project.thumb}
          alt=""
          className="case-still"
          width={intrinsicImageSize(project.thumb)?.width}
          height={intrinsicImageSize(project.thumb)?.height}
          style={{ objectPosition: project.thumbPos || 'center top' }}
        />
      ) : null}
      {project.outcome ? <p className="case-outcome">{project.outcome}</p> : null}
      {project.summary?.map((paragraph) => (
        <p key={paragraph}>{paragraph}</p>
      ))}
      {project.points ? (
        <ul className="case-points">
          {project.points.map((point) => (
            <li key={point}>{point}</li>
          ))}
        </ul>
      ) : null}
      {parts.length ? (
        <div className="case-parts">
          <p className="case-parts__label">Highlighted components</p>
          {parts.map((part) => (
            <button type="button" className="case-part" key={part.id} onClick={() => onOpen(part.id)}>
              <Icon name={part.icon} />
              <span>
                <strong>{part.title}</strong>
                <em>{part.outcome}</em>
              </span>
            </button>
          ))}
        </div>
      ) : null}
      <div className="case-links">
        {parentItem ? (
          <button type="button" className="button button--flex button--ghost" onClick={() => onOpen(parentItem.id)}>
            Back to {parentItem.title}
            <Icon name="uil-arrow-left" className="button__icon" />
          </button>
        ) : null}
        {project.links?.map((link) => (
          <a key={link.href} href={link.href} className="button button--flex" target="_blank" rel="noreferrer">
            {link.label}
            <Icon name="uil-external-link-alt" className="button__icon" />
          </a>
        ))}
        {project.github ? (
          <a href={project.github} className="button button--flex">
            GitHub <Icon name="uil-github-alt" className="button__icon" />
          </a>
        ) : null}
      </div>
    </>
  )
}

export default function Projects({ openId, onOpen }) {
  const tall = useNarrow()
  const railVisible = useRailVisible()
  const open = resolveOpen(openId)
  const railIds = PROJECT_RAIL.map((item) => item.id)
  const activeRail = useProjectRailActive(railVisible ? railIds : [])
  const modalTitle =
    open.kind === 'lead'
      ? COPY.lead.title
      : open.kind === 'tile'
        ? open.tile.title
        : open.kind === 'more'
          ? open.more.title
          : open.kind === 'legacy'
            ? open.project.modalTitle
            : undefined
  const modalKicker =
    open.kind === 'lead'
      ? 'Lead project'
      : open.kind === 'tile'
        ? KICKERS[open.tile.id]
        : open.kind === 'more'
          ? 'More shipped work'
          : undefined

  return (
    <section className="projects section" id="projects">
      <SectionHeader title={COPY.section.title} subtitle={COPY.section.kicker} />

      {/* Rail padding applies only to this shell — title stays flush above content. */}
      <div className="bx-with-rail">
        {railVisible ? <ProjectsJumpRail items={PROJECT_RAIL} activeId={activeRail} /> : null}

        <div className="container bx">
          <p className="bx-intro">{COPY.section.intro}</p>
          <div className="bx-grid">
            <LeadTile onOpen={onOpen} tall={tall} />
            {COPY.tiles.map((tile) => (
              <ArchitectureTile key={tile.id} tile={tile} onOpen={onOpen} tall={tall} />
            ))}
          </div>
          <h3 id="project-more" className="bx-morehead">
            More shipped work
          </h3>
          <div className="bx-grid bx-more">
            {moreItems.map((item, index) => (
              <MoreCard key={item.id} item={item} featured={index === 0} onOpen={onOpen} />
            ))}
          </div>
        </div>
      </div>

      <Modal
        open={Boolean(open.kind)}
        title={modalTitle}
        kicker={modalKicker}
        onClose={() => onOpen(null)}
        className={
          open.kind && open.kind !== 'legacy'
            ? `bx-dialog${open.kind === 'tile' && open.tile?.id === 'domains' ? ' bx-dialog--erd-full' : ''}`
            : undefined
        }
      >
        {open.kind === 'lead' ? <LeadPanel tall={tall} /> : null}
        {open.kind === 'tile' ? <TilePanel tile={open.tile} tall={tall} /> : null}
        {open.kind === 'more' ? <MorePanel item={open.more} /> : null}
        {open.kind === 'legacy' ? <LegacyBody key={open.project.id} project={open.project} onOpen={onOpen} /> : null}
      </Modal>
    </section>
  )
}
