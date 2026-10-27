export const mcpSpotlight = {
  kicker: 'Production · MCP',
  title: 'MCP servers connecting AI agents to Snowflake Cortex',
  framing:
    'I expose Cortex agents, Cortex Analyst on semantic views, Cortex Search services and guarded write procedures as MCP tools, so each AI client sees only the tools its server allows.',
  quote: 'MCP is not the agent. It is the API catalog the bots are allowed to see.',
  body: 'Domain analyst agents sit on semantic views, a master orchestrator handles multi-step work, and MCP bridges let other agents reuse that expertise under the customer’s identity and policy.',
  servers: [
    {
      label: 'Finance ledger server · read/write',
      line: 'Cursor/Cowork uses the finance ledger server (read/write): agents, analyst, search, procedures.',
    },
    {
      label: 'Finance ledger server · read-only for AI clients',
      line: 'Read-only server so AI finance bots can ask the ledger without write tools.',
    },
  ],
  tools: [
    { label: 'agent-run tools', opens: 'ledger-tools' },
    { label: 'Cortex Analyst on semantic views', opens: 'ledger-semantics' },
    { label: 'Cortex Search services', opens: 'ledger-semantics' },
    { label: 'guarded write procedures for ingest and mutate', opens: 'ledger-tools' },
  ],
  cta: { label: 'See the MCP tools', opens: 'ledger-tools' },
  diagram: {
    alt: 'AI agents connect through a read/write MCP server or a read-only MCP server to Cortex agents, Cortex Analyst on semantic views, Cortex Search services and, on the read/write server only, guarded write procedures in Snowflake.',
  },
}

function Diagram({ alt }) {
  const targets = [40, 140, 240, 340]
  return (
    <div className="rb-spot__diagram" role="img" aria-label={alt}>
      <svg viewBox="0 0 1050 450" preserveAspectRatio="xMidYMid meet" aria-hidden="true" focusable="false">
        {[90, 225, 360].map((y) => (
          <circle key={y} className="d-agent" cx="110" cy={y} r="42" />
        ))}
        {[90, 225, 360].map((y) =>
          [140, 310].map((sy) => <line key={`${y}-${sy}`} className="d-line" x1="152" y1={y} x2="400" y2={sy} />),
        )}
        {targets.map((ty, index) => (
          <line key={`rw-${ty}`} className="d-line d-line--strong" x1="560" y1="140" x2="760" y2={ty + 35} />
        ))}
        {targets.slice(0, 3).map((ty) => (
          <line key={`ro-${ty}`} className="d-line d-line--dash" x1="560" y1="310" x2="760" y2={ty + 35} />
        ))}
        <rect className="d-server" x="400" y="95" width="160" height="90" rx="14" />
        <rect className="d-server d-server--ro" x="400" y="265" width="160" height="90" rx="14" />
        {targets.map((ty, index) => (
          <rect key={ty} className={index === 3 ? 'd-tool d-tool--guard' : 'd-tool'} x="760" y={ty} width="230" height="70" rx="12" />
        ))}
      </svg>
    </div>
  )
}

export default function McpSpotlight({ data = mcpSpotlight, onOpen }) {
  return (
    <article className="rb-card rb-lift rb-mcp" id="t1-mcp" aria-labelledby="t1-mcp-h">
      <p className="rb-kicker">{data.kicker}</p>
      <h3 className="rb-spot__title" id="t1-mcp-h">{data.title}</h3>
      <p className="rb-spot__framing">{data.framing}</p>
      <blockquote className="rb-spot__quote">{data.quote}</blockquote>
      <p>{data.body}</p>
      <div className="rb-spot__servers">
        {data.servers.map((server) => (
          <div className="rb-spot__server" key={server.label}>
            <p className="rb-spot__server-label">{server.label}</p>
            <p>{server.line}</p>
          </div>
        ))}
      </div>
      <ul className="rb-chips" aria-label="MCP catalog tools">
        {data.tools.map((tool) => (
          <li key={tool.label}>
            <button
              type="button"
              className="rb-chip"
              aria-haspopup="dialog"
              onClick={() => onOpen?.(tool.opens)}
            >
              {tool.label}
            </button>
          </li>
        ))}
      </ul>
      <Diagram alt={data.diagram.alt} />
      <div>
        <button type="button" className="button button--flex rb-btn" aria-haspopup="dialog" onClick={() => onOpen?.(data.cta.opens)}>
          {data.cta.label}
        </button>
      </div>
    </article>
  )
}
