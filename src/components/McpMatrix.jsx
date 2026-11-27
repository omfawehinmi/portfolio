import { useState } from 'react'
import { DIAG } from '../data/architecture.js'
import { PLATFORM } from './diagrams/architectureSvg.js'

const KIND_LABEL = { read: 'read', write: 'write', jobs: 'jobs' }

function Logo({ id }) {
  return PLATFORM[id] ? <img className="bx-mcpm-logo" src={PLATFORM[id].src} alt="" /> : null
}

export default function McpMatrix() {
  const M = DIAG.mcpMatrix
  const [openId, setOpenId] = useState('analyst')
  const totals = Object.fromEntries(
    M.servers.map((s) => [s.id, M.families.reduce((sum, f) => sum + (f.n[s.id] || 0), 0)]),
  )
  const stop = (event) => event.stopPropagation()
  return (
    <div className="bx-mcpm" onClick={stop}>
      <div className="bx-mcpm-who">
        <div className="bx-mcpm-box">
          <p className="bx-mcpm-lab">Clients</p>
          <div className="bx-mcpm-line">
            <Logo id="grokbot" />
            <strong>Grokbot desks</strong>
            {M.desks.map((d) => (
              <span key={d.name} className="bx-mcpm-pill">
                {d.name} · {d.access}
              </span>
            ))}
          </div>
          <div className="bx-mcpm-line">
            {M.otherClients.map((c) => (
              <span key={c.name}>
                <strong>{c.name}</strong> <span className="bx-mcpm-pill">{c.access}</span>
              </span>
            ))}
          </div>
        </div>
        <div className="bx-mcpm-box">
          <p className="bx-mcpm-lab">Cortex agents, called as tools</p>
          <div className="bx-mcpm-line">
            <Logo id="snowflake" />
            {M.cortexAgents.map((a) => (
              <span key={a} className="bx-mcpm-pill is-ok">
                {a}
              </span>
            ))}
          </div>
          <p className="bx-mcpm-note">{M.cortexNote}</p>
        </div>
      </div>
      <div className="bx-mcpm-scroll">
        <table className="bx-mcpm-table">
          <thead>
            <tr>
              <th>Family</th>
              {M.servers.map((s) => (
                <th key={s.id} className="is-num" title={s.name}>
                  {s.label}
                  <small>{s.name.replace('MCP_', '')}</small>
                </th>
              ))}
              <th className="is-kind">Kind</th>
            </tr>
          </thead>
          <tbody>
            {M.families.map((f) => {
              const open = openId === f.id
              return [
                <tr key={f.id} className={open ? 'is-open' : ''}>
                  <td>
                    <button
                      type="button"
                      className="bx-mcpm-fam"
                      aria-expanded={open}
                      onClick={() => setOpenId(open ? null : f.id)}
                    >
                      <span aria-hidden="true">{open ? '▾' : '▸'}</span> {f.t}
                    </button>
                  </td>
                  {M.servers.map((s) => (
                    <td key={s.id} className={`is-num${f.n[s.id] ? '' : ' is-zero'}`}>
                      {f.n[s.id] || '–'}
                    </td>
                  ))}
                  <td className="is-kind">
                    {f.kind.map((k) => (
                      <span key={k} className={`bx-mcpm-kind is-${k}`}>
                        {KIND_LABEL[k]}
                      </span>
                    ))}
                  </td>
                </tr>,
                open ? (
                  <tr key={`${f.id}-tools`} className="is-open">
                    <td colSpan={M.servers.length + 2}>
                      <div className="bx-mcpm-chips">
                        {f.tools.map((t, i) => (
                          <code key={t} className={f.readTools && i < f.readTools ? 'is-read' : ''}>
                            {t}
                          </code>
                        ))}
                      </div>
                      {f.readTools ? (
                        <p className="bx-mcpm-note">The first {f.readTools} are on both ledger servers; the rest are write only.</p>
                      ) : null}
                    </td>
                  </tr>
                ) : null,
              ]
            })}
            <tr className="is-total">
              <td>Total tools</td>
              {M.servers.map((s) => (
                <td key={s.id} className="is-num">
                  {totals[s.id]}
                </td>
              ))}
              <td className="is-kind" />
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  )
}
