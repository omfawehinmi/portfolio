import { Fragment, useEffect, useState } from 'react'
import { COPY } from '../data/architecture.js'
import { projects } from '../data/projects.js'
import { intrinsicImageSize, publicUrl } from '../utils.js'
import ArchitectureDiagram from './diagrams/ArchitectureDiagram.jsx'
import Icon from './ui/Icon.jsx'
import Modal from './ui/Modal.jsx'
import SectionHeader from './ui/SectionHeader.jsx'
const KICKERS = {
  mcp: 'Request path',
  domains: "ERD, crow\u2019s foot",
  airflow: 'Scheduler inventory',
  dbt: 'Transform and deploy',
  pipes: 'Source feeds',
  security: 'Access zones',
  stack: 'Layer stack',
  features: 'Capability matrix',
  mychart: 'Healthcare data ingestion',
}

const LEAD_CAPTION =
  'As-built: MCP_HOUSEHOLD_LEDGER_GROK (read) and MCP_HOUSEHOLD_LEDGER (write) in front of SV_* / CS_* on HOUSEHOLD_LEDGER.'

const MORE_PREVIEW = {
  'hr-intelligence': { src: publicUrl('assets/img/bento/hr-intelligence.jpg'), sample: true },
  myorglab: { src: publicUrl('assets/img/bento/myorglab.jpg'), sample: true },
  'sof-portal': { src: publicUrl('assets/img/bento/sof-portal.jpg'), sample: true },
  palms: { src: publicUrl('assets/img/bento/palms.jpg'), sample: false },
  camera: { src: publicUrl('assets/img/bento/camera.jpg'), sample: false },
  inpainting: { src: publicUrl('assets/img/bento/inpainting.jpg'), sample: false },
}

const OPEN_ALIASES = {
  lead: 'lead',
  mcp: 'mcp',
  domains: 'domains',
  airflow: 'airflow',
  dbt: 'dbt',
  pipes: 'pipes',
  security: 'security',
  stack: 'stack',
  features: 'features',
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
  if (mapped === 'mychart') return { kind: 'mychart' }
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

function FoldBody({ project, className = '' }) {
  return (
    <div className="project-tile__fold">
      <p className={`project-tile__outcome bx-body${className ? ` ${className}` : ''}`}>
        {project.outcome}
      </p>
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
  return (
    <div className={`bx-media${big ? ' bx-big' : ''}`}>
      <img
        alt={alt}
        src={preview.src}
        width={size?.width}
        height={size?.height}
      />
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

function LeadTile({ onOpen, tall }) {
  const L = COPY.lead
  return (
    <article
      className="bx-tile bx-lead"
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
  return (
    <article
      className={`bx-tile bx-t-${tile.id}`}
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
      <FoldBody project={{ outcome: tile.body }} />
      <figure className="bx-fig">
        <ArchitectureDiagram kind={tile.id} tall={tall} inst="tile" />
      </figure>
    </article>
  )
}

function HealthcareTile({ onOpen, tall }) {
  const M = COPY.mychart
  return (
    <article
      className="bx-tile bx-t-mychart"
      data-testid="mychart-tile"
      tabIndex={0}
      role="button"
      aria-haspopup="dialog"
      onClick={(event) => openTile(event, 'mychart', onOpen)}
      onKeyDown={(event) => onTileKey(event, 'mychart', onOpen)}
    >
      <div className="bx-head">
        <div>
          <p className="bx-kicker">{M.kicker}</p>
          <h4 className="bx-h2">{M.title}</h4>
        </div>
        <OpenIcon />
      </div>
      <p className="bx-lede">{M.summary}</p>
      <figure className="bx-fig">
        <ArchitectureDiagram kind="mychart" tall={tall} inst="tile" />
      </figure>
    </article>
  )
}

function MychartPanel({ tall }) {
  const M = COPY.mychart
  return (
    <div className="bx-panel-body bx-mychart-panel">
      <figure className="bx-fig bx-fig-big">
        <ArchitectureDiagram kind="mychart" tall={tall} inst="open" />
        <figcaption>{M.caption}</figcaption>
      </figure>
      <p>{M.open}</p>
    </div>
  )
}

function MoreCard({ item, featured, onOpen }) {
  const industries = item.prev === 'hr-intelligence' ? COPY.industries : null
  const card = (
    <article
      className={`bx-tile bx-card ${featured ? 'bx-feat' : 'bx-row'}`}
      data-testid="project-tile"
      tabIndex={0}
      role="button"
      aria-haspopup="dialog"
      onClick={(event) => openTile(event, item.id, onOpen)}
      onKeyDown={(event) => onTileKey(event, item.id, onOpen)}
    >
      <PreviewMedia item={item} />
      <div className="bx-cardtext">
        <h4 className="bx-h3">{item.title}</h4>
        <FoldBody project={item} />
      </div>
      <OpenIcon />
    </article>
  )
  if (!industries) return card
  return (
    <div className="bx-feat-block">
      {card}
      <p className="bx-industries">
        {industries.line}{' '}
        <a href={industries.href}>{industries.link}</a>
      </p>
    </div>
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

function RowTable({ rows }) {
  if (!rows?.length) return null
  return (
    <div className="bx-rows">
      {rows.map((row) => (
        <div className="bx-rowline" key={row.name}>
          <code>{row.name}</code>
          <p>{row.role}</p>
          {row.schedule ? <span className="bx-rowmeta">{row.schedule}</span> : null}
          {row.status ? <span className="bx-rowstatus">{row.status}</span> : null}
        </div>
      ))}
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

function TilePanel({ tile, tall }) {
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

function MorePanel({ item }) {
  return (
    <div className="bx-panel-body">
      <PreviewMedia item={item} big />
      <p>{item.line}</p>
      <Chips tags={item.tags} />
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
  const open = resolveOpen(openId)
  const modalTitle =
    open.kind === 'lead'
      ? COPY.lead.title
      : open.kind === 'tile'
        ? open.tile.title
        : open.kind === 'mychart'
          ? COPY.mychart.title
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
        : open.kind === 'mychart'
          ? KICKERS.mychart
          : open.kind === 'more'
            ? 'More shipped work'
            : undefined

  return (
    <section className="projects section" id="projects">
      <SectionHeader title={COPY.section.title} subtitle={COPY.section.kicker} />

      <div className="container bx">
        <p className="bx-intro">{COPY.section.intro}</p>
        <div className="bx-grid">
          <LeadTile onOpen={onOpen} tall={tall} />
          {COPY.tiles.map((tile) => (
            <ArchitectureTile key={tile.id} tile={tile} onOpen={onOpen} tall={tall} />
          ))}
        </div>
        <h3 className="bx-morehead">More shipped work</h3>
        <div className="bx-grid bx-more">
          {moreItems.map((item, index) => (
            <Fragment key={item.id}>
              <MoreCard item={item} featured={index === 0} onOpen={onOpen} />
              {index === 0 ? <HealthcareTile onOpen={onOpen} tall={tall} /> : null}
            </Fragment>
          ))}
        </div>
      </div>

      <Modal
        open={Boolean(open.kind)}
        title={modalTitle}
        kicker={modalKicker}
        onClose={() => onOpen(null)}
        className={open.kind && open.kind !== 'legacy' ? 'bx-dialog' : undefined}
      >
        {open.kind === 'lead' ? <LeadPanel tall={tall} /> : null}
        {open.kind === 'tile' ? <TilePanel tile={open.tile} tall={tall} /> : null}
        {open.kind === 'mychart' ? <MychartPanel tall={tall} /> : null}
        {open.kind === 'more' ? <MorePanel item={open.more} /> : null}
        {open.kind === 'legacy' ? <LegacyBody key={open.project.id} project={open.project} onOpen={onOpen} /> : null}
      </Modal>
    </section>
  )
}
