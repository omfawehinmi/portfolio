import CaseTabs from '../ui/CaseTabs.jsx'
import Mermaid from './Mermaid.jsx'

// Source: internal migration docs
// Public page: generic names only.

const dataFlow = (dir) => `flowchart ${dir}
  subgraph sources[Sources]
    direction TB
    s1[Finance export]
    s2[Video stats]
    s3[Mail and docs]
    s4[Chat export]
    s5[Desk captures]
    s1 ~~~ s2 ~~~ s3 ~~~ s4 ~~~ s5
  end
  s1 --> raw[(Raw)]
  s2 --> raw
  s3 --> raw
  s4 --> raw
  s5 --> raw
  raw --> stg[(Staging and<br/>intermediate)]
  stg --> marts[(Analytics<br/>marts)]
  marts --> app[Desktop app,<br/>read-only]
  marts --> mcp[MCP clients,<br/>read-only]
  wk[Airflow on<br/>on-prem host] -->|dbt build| stg
  wk -->|task results| tr[(Test<br/>results)]
  ci[GitHub<br/>Actions] -->|JUnit results| tr
  tr --> dash[Red-alert<br/>dashboard]`

const prodEnv = (dir) => `flowchart ${dir}
  tf[Terraform: one state, plan on every PR] --> prod
  pr[Pipeline role] --> prod
  subgraph prod[PROD: 5 databases]
    direction TB
    p1[(Ledger database)]
    p2[(Projects database)]
    p3[(Agent workforce database)]
    p4[(Control database)]
    p5[(Audit store, append-only)]
    p1 ~~~ p2 ~~~ p3 ~~~ p4 ~~~ p5
  end`

const devEnv = (dir) => `flowchart ${dir}
  tf[Same Terraform state] --> dev
  dr[DEV pipeline role] --> dev
  subgraph dev[DEV: 4 databases, synthetic data only]
    direction TB
    d1[(Ledger DEV)]
    d2[(Projects DEV)]
    d3[(Agent workforce DEV)]
    d4[(Control DEV)]
    d1 ~~~ d2 ~~~ d3 ~~~ d4
  end`

const roles = (dir) => `flowchart ${dir}
  owners[Tier 1: Database owner roles] --> pipe[Tier 2: Pipeline and dbt service roles]
  pipe --> apps[Tier 3: Read-only app, agent and dashboard roles]
  gate[Tier 4: AI gate role] --> ai[AI user roles]
  owners -.->|read grants| apps`

const ciFlow = (dir) => `flowchart ${dir}
  pr[Pull request] --> checks
  subgraph checks[4 required checks]
    direction TB
    c1[Lint and unit tests]
    c2[Full-history secret scan]
    c3[Terraform static checks]
    c4[Snowflake plan, 0-destroy guard]
    c1 ~~~ c2 ~~~ c3 ~~~ c4
  end
  pr -.->|optional| dbt[dbt slim CI in a per-PR schema]
  checks --> merge[Merge to main]
  merge --> apply[Terraform apply: manual dispatch<br/>re-plan and guard]`

const releaseFlow = (dir) => `flowchart ${dir}
  merge[Merge<br/>to main] --> tests[DAG import<br/>tests]
  tests --> tag[Release tag]
  tag --> pull[On-prem host<br/>pulls the tag,<br/>dry run first]
  pull --> swap[Atomic swap<br/>to release]
  swap --> run[Airflow DAGs<br/>and daily dbt]
  run --> tr[(Test<br/>results)]
  swap -.->|import error| back[Redeploy<br/>prior tag]
  ci[CI JUnit<br/>results] -.->|planned| tr`

const Legend = () => (
  <p className="case-legend">Dotted lines: labeled <em>optional</em> (runs when configured) or <em>planned</em> (not built yet).</p>
)

const overview = (
  <>
    <div className="case-stats">
      <div><strong>211</strong><span>PRs</span></div>
      <div><strong>184</strong><span>merged</span></div>
      <div><strong>15</strong><span>decisions</span></div>
      <div><strong>5 + 4</strong><span>PROD + DEV databases</span></div>
      <div><strong>4</strong><span>required checks</span></div>
    </div>
    <h4>Problem</h4>
    <p>My personal data platform ran on Snowflake in a single-schema layout. Routine scripts ran on admin roles, objects were created by hand-run SQL, transforms lived in stored procedures, and orchestration was a mix of scheduled scripts and ad hoc runs. There was only a production environment, and test outcomes lived in logs. It worked, but nothing was reviewable, reproducible or safe to change.</p>
    <h4>Approach</h4>
    <p>I treated this as a data infrastructure project, not a warehouse tweak. The goal was a medallion multi-database layout managed by Terraform, built by dbt, orchestrated by Airflow, with DEV and PROD separated. Five working principles held every change:</p>
    <ul className="case-points">
      <li>Expand-only migrations.</li>
      <li>Plan before apply.</li>
      <li>Least privilege per service user.</li>
      <li>Synthetic data in DEV.</li>
      <li>Rollback written before apply.</li>
    </ul>
  </>
)

const architecture = (
  <>
    <h4>Layers and data flow</h4>
    <p>Sources land in a raw layer, move through staging and intermediate layers, and publish to analytics marts read by a desktop app and MCP read clients. Each domain database follows the same pattern.</p>
    <Mermaid chart={dataFlow} label="Data flow from five sources through raw, staging and marts to read-only clients, with test results feeding a dashboard" />
    <h4>Environments</h4>
    <p>PROD has 5 databases: a finance ledger, a projects domain, an agent workforce domain, a control plane and an audit store. DEV has 4 matching databases with their own warehouse, resource monitor and pipeline role, holding synthetic fixtures only.</p>
    <Mermaid chart={prodEnv} label="Terraform and the pipeline role manage 5 PROD databases" />
    <Mermaid chart={devEnv} label="The same Terraform state and a DEV pipeline role manage 4 DEV databases with synthetic data" />
    <h4>Roles</h4>
    <p>Every pipeline and app has its own service role and key-pair authenticated service user. Access to AI functions flows through a single gate role, so there is one place to audit and revoke. No person runs as a pipeline role.</p>
    <Mermaid chart={roles} label="Four role tiers: owners, pipeline and dbt, read-only apps, and an AI gate granting AI users" />
  </>
)

const decisions = [
  ['Layout (D1)', 'Keep the existing schemas and add a separate audit database, rather than rename schemas and break every fully qualified name. dbt takes over the marts in phases behind compatibility views.'],
  ['Terraform owns objects (D3)', 'Import before manage, expand-only, prevent_destroy, and a 0-destroy guard in CI.'],
  ['No staging environment (D6)', 'Per-database DEV copies plus transient CI schemas cover pre-production checks with fewer moving parts.'],
  ['Synthetic DEV data (D8)', 'No real personal or client data outside production.'],
  ['Pull deploys for Airflow (D13)', 'The on-prem host pulls tagged releases, so nothing needs inbound access to it.'],
  ['One-shot ops (D12)', 'Privileged changes ship as a PR, run once, then a follow-up PR removes the workflow so nothing stays armed.'],
]

const decisionsTab = (
  <>
    <p>Each design choice went into a decision log, 15 decisions in all (D0 to D14), with the alternatives I rejected and why. A few that shaped the result:</p>
    <div className="case-scroll" tabIndex={0} role="region" aria-label="Design decisions table">
    <table className="case-table">
      <tbody>
        {decisions.map(([k, v]) => (
          <tr key={k}><th scope="row">{k}</th><td>{v}</td></tr>
        ))}
      </tbody>
    </table>
    </div>
  </>
)

const delivery = (
  <>
    <h4>Terraform</h4>
    <p>One root module and one state cover DEV and PROD, so a single plan shows every change to both and the 0-destroy guard covers both. Plans run on every PR through a read-only OIDC role. Apply is a manual dispatch on main that re-plans and re-runs the guard. A nightly drift check opens an issue for any difference. Merging never applies.</p>
    <h4>CI checks to merge</h4>
    <Mermaid chart={ciFlow} label="Pull request runs four required checks and optional dbt slim CI, then merges; apply is a separate manual dispatch" />
    <h4>Release and pull deploy</h4>
    <p>Airflow 3 runs on the on-prem host with dbt through Cosmos. A merge runs DAG import tests and cuts a release tag. The deploy runs dry by default, swaps to the release atomically, and rolls back to the prior tag on any import error.</p>
    <Mermaid chart={releaseFlow} label="Release tag pulled by the on-prem host, atomic swap, rollback to prior tag on import error" />
    <Legend />
    <h4>dbt and test results</h4>
    <p>Models run with tests; slim CI builds unit tests and zero-row builds into a per-PR schema, and a sweeper drops stale PR schemas. Prod builds use a dedicated warehouse, separate from CI. Airflow task outcomes, dbt run results and CI results land in one append-only test-results table, with a latest view that feeds a red-alert dashboard. Every test job writes JUnit XML and keeps a test-results artifact for 30 days.</p>
  </>
)

const beforeAfter = [
  ['Roles', 'Routine scripts on admin roles', 'Per-service roles'],
  ['Objects', 'Hand-run SQL', 'Terraform plan and apply with CI'],
  ['Transforms', 'Stored procedures', 'dbt with tests and slim CI'],
  ['Orchestration', 'Scheduled scripts and ad hoc runs', 'Airflow 3 on the on-prem host, tagged releases'],
  ['Environments', 'Prod only', 'DEV and PROD'],
]

const outcomes = (
  <>
    <div className="case-scroll" tabIndex={0} role="region" aria-label="Before and after table">
    <table className="case-table">
      <thead><tr><th>Aspect</th><th>Before</th><th>After</th></tr></thead>
      <tbody>
        {beforeAfter.map(([a, b, c]) => (
          <tr key={a}><th scope="row">{a}</th><td data-label="Before">{b}</td><td data-label="After">{c}</td></tr>
        ))}
      </tbody>
    </table>
    </div>
    <ul className="case-points">
      <li>211 PRs across 4 of 27 repos scanned: 184 merged, 16 open, 8 draft, 3 closed. 26 of them were groundwork that came before the migration plan.</li>
      <li>15 recorded decisions (D0 to D14), each with alternatives and sources.</li>
      <li>Every infrastructure change is reversible by design: a change log entry with exact rollback steps is written first, plans must show 0 destroys, and Airflow rollback is a redeploy of the previous tag.</li>
      <li>Incidents handled in the open: 3 incidents documented with cause, response and learnings, including a partial Terraform apply that stayed safe because applies are create-only and 0-destroy.</li>
      <li>Next: bring dbt build results into the CI JUnit test summaries.</li>
    </ul>
  </>
)

export default function MigrationCaseStudy() {
  return (
    <CaseTabs
      tabs={[
        { id: 'overview', label: 'Overview', content: overview },
        { id: 'architecture', label: 'Architecture', content: architecture },
        { id: 'decisions', label: 'Decisions', content: decisionsTab },
        { id: 'delivery', label: 'Terraform, dbt, Airflow', content: delivery },
        { id: 'outcomes', label: 'Before / after', content: outcomes },
      ]}
    />
  )
}
