// Portfolio Snowflake section rewritten from the as-built in
// myorglab-repos/household-data-architecture (prod-vs-planned, snowflake-objects,
// agents-and-mcp). Real databases, MCP servers, catalogs, EAIs, and DAG names.
// No secrets, PATs, or account locators. Status language matches the as-built:
// production vs planned vs paused. Edit or cut anything you do not want public.

export const COPY = {
  section: {
    kicker: 'Projects',
    title: 'Work that shows the platform',
    intro:
      'The lead project is HOUSEHOLD_LEDGER on Snowflake: 4 production databases, 3 MCP servers, 16 Airflow DAG files on the office-PC scheduler, and dbt targets hl_prod / hl_dev that build ANALYTICS and SV_* catalogs agents actually query. The tiles under it break that stack into MCP, schemas, every DAG, dbt/deploy, feeds, roles, and catalogs. Other shipped work follows in a shorter list.',
  },
  lead: {
    title: 'HOUSEHOLD_LEDGER and MCP',
    subhead: 'Snowflake data architecture and MCP',
    oneLine:
      '4 prod Snowflake databases · 3 MCP servers · 16 Airflow DAG files (13 prod-shaped + 3 *_dev) · 3 ledger feeds (Monarch, Gmail, Drive) plus YouTube · 15 SV_* Analyst views · 6 Cortex Search services. Agents hit MCP catalogs, not an open warehouse login.',
    tags: [
      'Snowflake',
      'HOUSEHOLD_LEDGER',
      'Airflow',
      'dbt',
      'Terraform',
      'Cortex',
      'MCP',
      'Monarch',
      'Gmail',
    ],
    stats: [
      { n: '4', label: 'prod databases' },
      { n: '3', label: 'MCP servers' },
      { n: '16', label: 'Airflow DAG files' },
      { n: '3', label: 'ledger source feeds' },
      { n: '15', label: 'SV_* Analyst views' },
      { n: '6', label: 'CS_* search services' },
    ],
    intro:
      'This is the household data platform I build and run. Snowflake holds HOUSEHOLD_LEDGER (finance), MOONSHOTS (YouTube), AGENTIC_WORKFORCE, and CONTROL — layers as schemas (RAW, CORE, CTRL, ANALYTICS, AI), not one database per layer. Airflow on the office PC is the production scheduler (pinned releases such as airflow-20261107-02cbc74); Snowflake tasks are retired. Monarch GraphQL, Gmail, and Drive feed the ledger through EAIs. Grok bots and Cursor attach to MCP servers that bind Analyst on SV_*, Cortex Search on CS_*, and approved procedures — never a general warehouse login.',
  },
  tiles: [
    {
      id: 'mcp',
      n: '2',
      title: 'MCP_HOUSEHOLD_LEDGER split',
      svg: 'mcp',
      body:
        'HOUSEHOLD_LEDGER.CTRL hosts two of three MCP servers. MCP_HOUSEHOLD_LEDGER_GROK is the read catalog Grok desks attach to: household_finance_agent, ledger_ops_agent, analyst_* tools bound to the 15 SV_* views, search over CS_DOCUMENTS / CS_EMAIL_MESSAGES / CS_GAPS_IDEA / CS_PARSE_REVIEW, plus document stage list and download. Money answers are a staff rule for Stephen even though the SQL spec lists analyst tools on the read server. MCP_HOUSEHOLD_LEDGER is the write catalog Cursor / Eta use: the same read surface plus proc_monarch_extract, proc_monarch_mutate, proc_ledger_mutate, proc_gmail_ingest, proc_email_pipeline, proc_reconcile_pending, proc_document_extract, proc_rebuild_analytics, and related pipeline procs. MOONSHOTS.CTRL has MCP_MOONSHOTS for the YouTube catalog only (Cynthia). Auth is a Snowflake PAT under role MCP_LEDGER_CLIENT. A read client never holds write tools.',
      caption:
        'Read path through MCP_HOUSEHOLD_LEDGER_GROK to SV_* and CS_*. Write path through MCP_HOUSEHOLD_LEDGER into approved CORE / pipeline procedures.',
    },
    {
      id: 'domains',
      n: '3',
      title: 'Databases and schemas',
      svg: 'erd',
      body:
        'Production databases: HOUSEHOLD_LEDGER, MOONSHOTS, AGENTIC_WORKFORCE, CONTROL (plus *_DEV twins with synthetic fixtures). HOUSEHOLD_LEDGER schemas: PUBLIC, RAW, CORE, CTRL, ANALYTICS, AI, with transient STAGING and INTERMEDIATE. CONTROL holds PROC, QUEUE, STAGE, FLAGS, TASKS, MCP, SECURITY, TESTS. A RAW row is one landed payload. A CORE row is one business object. ANALYTICS is reporting grain and the SV_* Analyst catalogs. Several landing procedures write RAW and CORE in the same call, so CORE is not only a later rebuild of RAW. dbt reads CORE into staging/marts (and Cosmos dbt_daily targets hl_prod / ANALYTICS_DBT) — see the dbt tile for the deploy path.',
      caption:
        'Product databases hold layers as schemas. Example documents path: DOCUMENTS_RAW → DOCUMENTS → FACT_DOCUMENT_STANDING → SV_DOCUMENT_STANDING / CS_DOCUMENTS.',
    },
    {
      id: 'airflow',
      n: '4',
      title: 'Airflow on the office PC — every DAG',
      svg: 'airflow',
      body:
        '16 DAG files in household-platform Airflow. Production scheduling is the office-PC Docker pin (airflow-20261107-02cbc74 on the 2026-11-07 check; later pins exist in the change log). A file in git is not the same as an unpaused job: only moonshots_yt_sync_daily and moonshots_eyl_sync were confirmed unpaused in that check. Open the tile for the full inventory — schedule, responsibility, and prod status for each DAG.',
      caption:
        '16 DAG files · 13 prod-shaped · 3 *_dev · 2 confirmed unpaused on the 2026-11-07 pin. Pause unknown ≠ running.',
      rows: [
        {
          name: 'moonshots_yt_sync_daily',
          role: 'YouTube catalog sync into MOONSHOTS',
          schedule: '40 7 * * *',
          status: 'Production — unpaused on 2026-11-07 pin',
        },
        {
          name: 'moonshots_eyl_sync',
          role: 'EYL / YouTube companion sync',
          schedule: '10 7,23 * * *',
          status: 'Production — unpaused on 2026-11-07 pin',
        },
        {
          name: 'moonshots_transcript_backfill',
          role: 'Transcript backfill (Airflow port of Windows job)',
          schedule: 'none',
          status: 'Not running — stayed paused',
        },
        {
          name: 'monarch_load',
          role: 'Monarch GraphQL → HOUSEHOLD_LEDGER RAW/CORE via EAI_MONARCH',
          schedule: '0 7 * * *',
          status: 'Prod-shaped (fixed prod). Pause unknown',
        },
        {
          name: 'ledger_email_alerts',
          role: 'Gmail card/bank alerts → RAW/CORE via EAI_GMAIL',
          schedule: '15 7,12,17 * * *',
          status: 'Code on pin. Pause unknown',
        },
        {
          name: 'ledger_email_pull_poll',
          role: 'Poll / follow-up for the email pipeline',
          schedule: '*/5 * * * *',
          status: 'Code on pin. Pause unknown',
        },
        {
          name: 'drive_poll',
          role: 'Google Drive poll → document stages',
          schedule: '0 * * * *',
          status: 'Code on pin. Pause unknown',
        },
        {
          name: 'doc_extract',
          role: 'PC OCR (pypdf then Tesseract) → CORE document facts',
          schedule: 'triggered',
          status: 'Do not mark unpaused — quota pause in change log',
        },
        {
          name: 'ledger_dbt_daily',
          role: 'Bash dbt for ledger models (LEDGER_WH / COMPUTE_WH, staging + marts)',
          schedule: '0 6 * * *',
          status: 'Prod-shaped (fixed prod). Pause unknown',
        },
        {
          name: 'dbt_daily',
          role: 'Astronomer Cosmos dbt, target hl_prod / ANALYTICS_DBT, per-model tasks when manifest exists',
          schedule: '0 6 * * *',
          status: 'Prod-shaped Cosmos. Pause unknown',
        },
        {
          name: 'agentic_workforce_dbt_daily',
          role: 'dbt for AGENTIC_WORKFORCE analytics',
          schedule: '15 7 * * *',
          status: 'Code on pin. Pause unknown',
        },
        {
          name: 'hr_dbt_daily',
          role: 'HR domain dbt blueprint',
          schedule: '30 6 * * *',
          status: 'Code on pin — HR account still a blueprint',
        },
        {
          name: 'platform_smoke',
          role: 'Manual smoke / health checks',
          schedule: 'none',
          status: 'Manual only',
        },
        {
          name: 'ledger_dbt_daily_dev',
          role: 'Isolated ledger dbt twin',
          schedule: 'none',
          status: 'Not production',
        },
        {
          name: 'dbt_daily_dev',
          role: 'Isolated Cosmos dbt twin',
          schedule: 'none',
          status: 'Not production',
        },
        {
          name: 'monarch_load_dev',
          role: 'Isolated Monarch load twin',
          schedule: 'none',
          status: 'Not production',
        },
      ],
    },
    {
      id: 'dbt',
      n: '5',
      title: 'dbt, schemas, and prod deploys',
      svg: 'dbt',
      body:
        'RAW lands payloads. CORE holds business objects (often written in the same procedure pass as RAW). dbt builds STAGING / marts / ANALYTICS on top of CORE: ledger_dbt_daily runs Bash dbt on LEDGER_WH or COMPUTE_WH; dbt_daily runs Cosmos against target hl_prod (docstring: ANALYTICS_DBT) with per-model tasks when a manifest is present. Dev twins force hl_dev / *_DEV and have no schedule. PR CI can use schema HOUSEHOLD_LEDGER_CI.DBT_PR_<n> when secrets exist, otherwise it skips. Terraform plan/apply on main (apply=true, environment production) owns DBs, schemas, roles, warehouses; procedures stay loader-owned. Required CI checks in the migration chapters: ruff + pytest, gitleaks, static, plan. Production environment required reviewer is not enforced in those chapters — Michael’s go for apply is the team rule. Airflow reaches the PC by tagged release + manual pull deploy, not by Snowflake tasks.',
      caption:
        'RAW → CORE → dbt STAGING/marts → ANALYTICS + SV_*. hl_prod vs hl_dev. PR CI schema, then merge/apply to prod.',
      rows: [
        {
          name: 'RAW',
          role: 'One landed page, file, or API payload (Monarch, Gmail MIME, staged docs)',
          status: 'Landing schema',
        },
        {
          name: 'CORE',
          role: 'One business object — often written with RAW in the same procedure',
          status: 'Conformed schema',
        },
        {
          name: 'STAGING / marts',
          role: 'dbt factory schemas on the ledger path',
          status: 'Transform',
        },
        {
          name: 'ANALYTICS + SV_*',
          role: 'Reporting grain and Cortex Analyst contracts (15 semantic views)',
          status: 'Serve / MCP catalogs',
        },
        {
          name: 'hl_prod / hl_dev',
          role: 'dbt targets — fixed-prod DAGs refuse DEV_MIGRATION_WH',
          status: 'Environment split',
        },
        {
          name: 'HOUSEHOLD_LEDGER_CI.DBT_PR_<n>',
          role: 'Per-PR dbt CI schema when secrets exist',
          status: 'Not prod',
        },
        {
          name: 'Terraform apply on main',
          role: 'Manual dispatch apply=true → environment production',
          status: 'Infra deploy',
        },
        {
          name: 'Airflow pin pull',
          role: 'Tagged release + manual PC pull deploy',
          status: 'Scheduler deploy',
        },
      ],
    },
    {
      id: 'pipes',
      n: '6',
      title: 'Source feeds into Snowflake',
      svg: 'pipes',
      body:
        'Three ledger feeds hit Airflow, then HOUSEHOLD_LEDGER: Monarch via EAI_MONARCH (monarch_load / proc_monarch_extract), Gmail via EAI_GMAIL (ledger_email_alerts → classify → reconcile_pending), and Drive via drive_poll → doc_extract (pypdf then local Tesseract — Cortex is not used for OCR). YouTube feeds MOONSHOTS through the two confirmed-unpaused Moonshots DAGs. Shape is extract/land → transform (dbt) → test. No S3/Snowpipe path — that design was dropped.',
      caption:
        'Monarch / Gmail / Drive → Airflow → RAW/CORE → dbt → ANALYTICS → MCP. YouTube → MOONSHOTS.',
    },
    {
      id: 'security',
      n: '7',
      title: 'Roles and Cortex gate',
      svg: 'security',
      body:
        'Access is split by role and by which MCP server a client is attached to. Pipeline work uses R_PIPELINE / R_PC_PIPELINE; interactive and agent clients use MCP_LEDGER_CLIENT against the MCP path, not a shared open warehouse login. R_CORTEX_GATE is the sole holder of SNOWFLAKE.CORTEX_USER. Answers are supposed to come from certified SV_* views and CS_* search services. Public copy stops at that architecture: no account locators, hostnames, or PATs.',
      caption:
        'Role separation plus MCP_HOUSEHOLD_LEDGER vs MCP_HOUSEHOLD_LEDGER_GROK, with Cortex gated behind R_CORTEX_GATE.',
    },
    {
      id: 'stack',
      n: '8',
      title: 'Stack',
      svg: 'stack',
      body:
        'Snowflake holds the data and the catalogs. Airflow on the office PC schedules the 16 DAG files. dbt builds tested models (hl_prod / hl_dev; ANALYTICS_DBT). Terraform under infra/terraform/snowflake owns databases, schemas, roles, warehouses, and monitors; procedures stay loader-owned. GitHub Actions is the CI path (ruff + pytest, gitleaks, static, plan). Cortex covers Agents, Analyst on SV_*, and Search on CS_*. Warehouses include LEDGER_WH, COMPUTE_WH, INGEST_WH, YT_SYNC_WH, DEV_MIGRATION_WH, ADHOC_WH. Same control plane pattern I use in MyOrgLab customer deployments.',
      caption:
        'Airflow, Snowflake, dbt, Terraform, GitHub Actions, with MCP as the agent interface.',
    },
    {
      id: 'features',
      n: '9',
      title: 'Catalogs and Snowflake features',
      svg: 'features',
      body:
        '15 semantic views are the Analyst contracts: SV_SPENDING, SV_CASHFLOW, SV_BALANCES_NET_WORTH, SV_LIABILITIES_DUES, SV_EARLY_ACTIVITY, SV_BUDGET_VARIANCE, SV_INVESTMENTS, SV_OBLIGATIONS, SV_DOCUMENT_STANDING, SV_OPS_FRESHNESS, SV_TRANSACTIONS, SV_RECURRING_FIXED, SV_DEBT_PAYOFF, SV_ACCOUNTS_RECEIVABLE, SV_SAVINGS_GOALS. Cortex Search: CS_DOCUMENTS, CS_EMAIL_MESSAGES, CS_GAPS_IDEA, CS_PARSE_REVIEW on the ledger; CS_TRANSCRIPTS and CS_VIDEOS on Moonshots (6 services). Cortex agents: AGENT_HOUSEHOLD_FINANCE, AGENT_HOUSEHOLD_FINANCE_RO, AGENT_LEDGER_OPS. MCP servers in CTRL expose those catalogs as read or write. Warehouse work is batched into a load window; interactive and agent work can sit on a separate warehouse.',
      caption:
        '15 SV_* and 6 CS_* catalogs. MCP_HOUSEHOLD_LEDGER_GROK and MCP_HOUSEHOLD_LEDGER sit in front of them.',
    },
  ],
  more: [
    {
      prev: 'hr-intelligence',
      title: 'HR Intelligence',
      line:
        'Workday and SuccessFactors landed as governed Snowflake semantics, the same grain agents and dashboards read. At Novanta I was the sole engineer on the Workday-to-Snowflake integration and put Cortex on that grain so recurring Tier 0 and Tier 1 reporting queues went away. At NetApp I land SuccessFactors, Fieldglass, Eightfold, and Visier into Snowflake and One Model.',
      tags: [
        'Workday',
        'SuccessFactors',
        'Fieldglass',
        'Eightfold',
        'Visier',
        'One Model',
        'Snowflake',
        'Cortex',
      ],
    },
    {
      prev: 'myorglab',
      title: 'MyOrgLab',
      line:
        'Customer-facing practice built from the same pattern as HOUSEHOLD_LEDGER: in-account Cortex agents, SV_* semantic views, and MCP bridges — not a multi-tenant copy of customer data.',
      tags: ['Snowflake', 'Cortex', 'semantic views', 'MCP', 'Streamlit', 'Power BI'],
    },
    {
      prev: 'sof-portal',
      title: 'SOF client portal',
      line:
        'The portal clients still log into: document vault, messaging, and live status, with MFA, on AWS.',
      tags: ['React', 'Flask', 'AWS', 'MFA'],
    },
    {
      prev: 'palms',
      title: 'Robotic punching bag',
      line:
        'PALMS I1: three crown cameras find the opponent, then the arms rotate to strike. The punch clip is the mechanism in Blender, not live closed-loop firmware.',
      tags: ['YOLOv8', 'Raspberry Pi Pico W', 'BLE', 'steppers', 'Blender'],
    },
    {
      prev: 'camera',
      title: 'Camera tracking',
      line:
        'The live vision loop the bag is built to take: YOLOv8 body and head detection on a Pico W that drives the steppers.',
      tags: ['YOLOv8', 'Raspberry Pi Pico W', 'BLE'],
    },
    {
      prev: 'inpainting',
      title: 'Inpainting detection',
      line: 'WPI coursework, not a product: CNN and ensemble snippets on CIFAKE and DeepFake Art.',
      tags: ['CNN', 'Hyperopt', 'CIFAKE', 'DeepFake Art'],
    },
  ],
}

// Diagram labels use the same real object names. Layer label strings are consumed by architectureSvg.js — keep Clients / MCP servers / Cortex / Catalogs / Snowflake.
export const DIAG = {
  lead: {
    title:
      'Lead diagram: agents through MCP_HOUSEHOLD_LEDGER_GROK and MCP_HOUSEHOLD_LEDGER into SV_* / CS_* and HOUSEHOLD_LEDGER schemas',
    layers: [
      {
        label: 'Clients',
        nodes: [
          { t: 'Grok bots / read clients', s: 'MCP_HOUSEHOLD_LEDGER_GROK' },
          { t: 'Cursor / write client', s: 'MCP_HOUSEHOLD_LEDGER' },
        ],
      },
      {
        label: 'MCP servers',
        nodes: [
          { t: 'MCP_HOUSEHOLD_LEDGER_GROK', s: 'read catalog, no write tools', e: 1 },
          { t: 'MCP_HOUSEHOLD_LEDGER', s: 'read tools + mutate / pipeline procs', e: 1 },
        ],
      },
      {
        label: 'Cortex',
        nodes: [
          { t: 'AGENT_HOUSEHOLD_FINANCE' },
          { t: 'Analyst', s: 'text to SQL on SV_*' },
          { t: 'Search', s: 'CS_* services' },
        ],
      },
      {
        label: 'Catalogs',
        nodes: [
          { t: 'SV_* semantic views', e: 1 },
          { t: 'CS_* Cortex Search', e: 1 },
        ],
      },
      {
        label: 'Snowflake',
        nodes: [
          { t: 'ANALYTICS', s: 'one reporting grain' },
          { t: 'CORE', s: 'one business object' },
        ],
      },
    ],
    edges: [
      [0, 0, 1, 0],
      [0, 1, 1, 1],
      [1, 0, 2, 0],
      [1, 0, 2, 1],
      [1, 0, 2, 2],
      [2, 0, 3, 0],
      [2, 0, 3, 1],
      [2, 1, 3, 0],
      [2, 2, 3, 1],
      [3, 0, 4, 0],
      [3, 1, 4, 1],
      [1, 1, 4, 1],
    ],
  },
  pipes: {
    title: 'Platform flow: Monarch, Gmail, Drive through Airflow into HOUSEHOLD_LEDGER',
    layers: [
      {
        label: 'Sources',
        nodes: [{ t: 'Monarch GraphQL' }, { t: 'Gmail (EAI_GMAIL)' }, { t: 'Drive docs' }],
      },
      {
        label: 'Extract and land',
        nodes: [
          {
            t: 'Airflow on office PC',
            s: 'production scheduler (pinned releases)',
            e: 1,
          },
          {
            t: 'doc_extract',
            s: 'pypdf then Tesseract; no Cortex OCR',
          },
        ],
      },
      {
        label: 'Snowflake',
        nodes: [{ t: 'RAW' }, { t: 'CORE' }],
      },
      {
        label: 'Transform',
        nodes: [{ t: 'dbt / procedures', e: 1 }],
      },
      {
        label: 'Test',
        nodes: [{ t: 'Tests / QA procs' }],
      },
      {
        label: 'Serve',
        nodes: [{ t: 'ANALYTICS + SV_*' }, { t: 'MCP servers', e: 1 }],
      },
    ],
    edges: [
      [0, 0, 1, 0],
      [0, 1, 1, 0],
      [0, 2, 1, 1],
      [1, 0, 2, 0],
      [1, 0, 2, 1],
      [1, 1, 2, 0],
      [2, 0, 3, 0],
      [2, 1, 3, 0],
      [3, 0, 4, 0],
      [4, 0, 5, 0],
      [4, 0, 5, 1],
    ],
  },
  security: {
    title: 'Access split by role and MCP server',
    layers: [
      {
        label: 'Roles',
        nodes: [
          { t: 'R_PIPELINE / R_PC_PIPELINE' },
          { t: 'Interactive roles' },
          { t: 'MCP_LEDGER_CLIENT' },
        ],
      },
      {
        label: 'MCP split',
        nodes: [
          { t: 'MCP_HOUSEHOLD_LEDGER', e: 1 },
          { t: 'MCP_HOUSEHOLD_LEDGER_GROK', s: 'no write tools', e: 1 },
        ],
      },
      {
        label: 'Cortex gate',
        nodes: [{ t: 'R_CORTEX_GATE', s: 'sole SNOWFLAKE.CORTEX_USER' }],
      },
      {
        label: 'Catalogs',
        nodes: [{ t: 'SV_* certified views' }, { t: 'CS_* search services' }],
      },
    ],
  },
  stack: {
    title: 'Stack diagram',
    layers: [
      { label: 'Agent interface', nodes: [{ t: 'MCP in CTRL', e: 1 }] },
      { label: 'Transform', nodes: [{ t: 'dbt', s: 'hl_prod / hl_dev' }] },
      {
        label: 'Warehouse and catalogs',
        nodes: [
          {
            t: 'Snowflake',
            s: 'HOUSEHOLD_LEDGER, MOONSHOTS, AGENTIC_WORKFORCE, CONTROL',
            e: 1,
          },
        ],
      },
      { label: 'Scheduler', nodes: [{ t: 'Airflow', s: 'office PC Docker pin' }] },
      {
        label: 'Infrastructure and CI',
        nodes: [{ t: 'Terraform' }, { t: 'GitHub Actions' }],
      },
    ],
  },
  features: {
    title: 'Catalogs and MCP split',
    layers: [
      {
        label: 'MCP servers in CTRL',
        nodes: [
          { t: 'MCP_HOUSEHOLD_LEDGER_GROK', e: 1 },
          { t: 'MCP_HOUSEHOLD_LEDGER', e: 1 },
        ],
      },
      {
        label: 'Catalogs',
        nodes: [
          { t: 'SV_*', s: 'Analyst catalogs' },
          { t: 'CS_*', s: 'Search catalogs' },
        ],
      },
      {
        label: 'Schemas',
        nodes: [{ t: 'RAW / CORE / CTRL / ANALYTICS', s: 'defined row meaning' }],
      },
      {
        label: 'Warehouses',
        nodes: [
          { t: 'LEDGER_WH / INGEST_WH', s: 'batched load window' },
          { t: 'Interactive / agents', s: 'separate warehouse' },
        ],
      },
    ],
  },
  erd: {
    title:
      'Schema ERD for the documents path in HOUSEHOLD_LEDGER: RAW → CORE → ANALYTICS, plus catalogs',
    areas: ['HOUSEHOLD_LEDGER', 'MOONSHOTS', 'AGENTIC_WORKFORCE', 'CONTROL'],
    schemas: ['RAW', 'CORE', 'CTRL', 'ANALYTICS'],
    entities: [
      {
        layer: 'LANDING: RAW',
        name: 'DOCUMENTS_RAW',
        rows: [
          ['LOAD_ID', 'PK'],
          ['DOCUMENT_ID', 'PK'],
          ['STAGE_PATH'],
          ['SOURCE_CHANNEL'],
        ],
        meaning: 'One landed file or API payload',
      },
      {
        layer: 'CONFORMED: CORE',
        name: 'DOCUMENTS',
        rows: [
          ['DOCUMENT_ID', 'PK'],
          ['RAW_LOAD_ID', '', 'ref'],
          ['DOCUMENT_KIND', 'FK'],
          ['TEXT_EXTRACT'],
        ],
        meaning: 'One business document',
      },
      {
        layer: 'ANALYTICS',
        name: 'FACT_DOCUMENT_STANDING',
        rows: [
          ['AS_OF_DATE', 'PK'],
          ['DOCUMENT_ID', 'PK'],
          ['STATUS'],
          ['DAYS_TO_EXPIRY'],
        ],
        meaning: 'One reporting grain',
      },
      {
        layer: 'SEMANTIC VIEW',
        name: 'SV_DOCUMENT_STANDING',
        rows: [['facts'], ['dimensions']],
        meaning: 'Analyst catalog',
      },
      {
        layer: 'CORTEX SEARCH',
        name: 'CS_DOCUMENTS',
        rows: [['SEARCH_TEXT'], ['attributes']],
        meaning: 'Search catalog (serving may be suspended; indexing active)',
      },
    ],
  },
}

DIAG.lead.airflow = {
  t: 'Airflow on office PC',
  s: '16 DAG files · 2 confirmed unpaused on 2026-11-07 pin',
}
DIAG.airflow = {
  title: 'Airflow DAG inventory on the office-PC pin',
  headline: '16 files in git',
  counts: [
    { n: '13', label: 'prod-shaped' },
    { n: '3', label: '*_dev' },
    { n: '2', label: 'confirmed unpaused' },
  ],
}
DIAG.dbt = {
  title: 'dbt path from RAW/CORE to ANALYTICS and prod apply',
  layers: [
    { t: 'RAW', s: 'landed payloads' },
    { t: 'CORE', s: 'business objects' },
    { t: 'dbt STAGING / marts', s: 'hl_prod · ledger_dbt_daily / dbt_daily' },
    { t: 'ANALYTICS + SV_*', s: '15 Analyst views' },
    { t: 'CI → merge → apply', s: 'PR schema · Terraform · Airflow pin pull' },
  ],
}
DIAG.erd.searchNote =
  'CS_DOCUMENTS indexes conformed document text. Object names are real. No row data is shown.'

DIAG.code = {
  raw: 'RAW',
  core: 'CORE',
  analytics: 'ANALYTICS',
  readSrv: 'MCP_HOUSEHOLD_LEDGER_GROK',
  writeSrv: 'MCP_HOUSEHOLD_LEDGER',
  svOps: 'CTRL.SV_OPS_FRESHNESS',
  svDoc: 'SV_DOCUMENT_STANDING',
  cs: 'CS_DOCUMENTS',
  airflow: '16 DAGs · pin airflow-20261107-02cbc74',
  land: 'monarch_load, ledger_email_alerts',
  ocr: 'doc_extract (paused for WH quota)',
  dbt: 'ledger_dbt_daily, dbt_daily → hl_prod',
}

Object.assign(DIAG.pipes, {
  airflow: DIAG.pipes.layers[1].nodes[0],
  ocr: DIAG.pipes.layers[1].nodes[1],
  dbt: DIAG.pipes.layers[3].nodes[0],
})

Object.assign(DIAG.security, {
  roles: DIAG.security.layers[0].nodes.map((n) => n.t),
  write: 'MCP_HOUSEHOLD_LEDGER',
  read: { t: 'MCP_HOUSEHOLD_LEDGER_GROK', s: 'no write tools' },
  gate: { t: 'R_CORTEX_GATE', s: 'sole SNOWFLAKE.CORTEX_USER' },
  cat: ['SV_* certified views', 'CS_* search services'],
})

DIAG.features.title = 'Snowflake capability matrix'
DIAG.features.rows = [
  {
    label: 'Access',
    z: 'agent',
    cells: [
      { i: 'hex', t: 'MCP_HOUSEHOLD_LEDGER_GROK', s: 'no write tools' },
      { i: 'hex', t: 'MCP_HOUSEHOLD_LEDGER', s: 'mutate + pipeline procs' },
    ],
  },
  {
    label: 'Catalogs',
    z: 'data',
    cells: [
      { i: 'grid', t: 'SV_*', s: 'Analyst catalogs' },
      { i: 'search', t: 'CS_*', s: 'Search catalogs' },
    ],
  },
  {
    label: 'Data',
    z: 'data',
    cells: [{ i: 'layers', t: 'RAW / CORE / CTRL / ANALYTICS', s: 'schemas in product DBs' }],
  },
  {
    label: 'Warehouses',
    cells: [
      { i: 'clock', t: 'LEDGER_WH / INGEST_WH', s: 'batched load window' },
      { i: 'split', t: 'Interactive / agents', s: 'separate warehouse' },
    ],
  },
]

COPY.mychart = {
  id: 'mychart',
  kicker: 'Healthcare data ingestion',
  title: 'MyChart data in Snowflake',
  summary:
    'SMART on FHIR R4 path from Epic MyChart into a planned HOUSEHOLD_HEALTH database (RAW / STAGING / CORE / CTRL). App registration, JWKS, and sandbox token exchange exist; production patient loads are not live yet.',
  open:
    'Designed a SMART on FHIR R4 ingestion path from an Epic MyChart patient portal toward Snowflake. Patient authorization is OAuth; tokens are kept in a token vault on AWS Lambda, DynamoDB, and SSM Parameter Store. A scheduled Airflow load is designed to pull FHIR R4 resources onto a Snowflake internal stage, then stored procedures load VARIANT raw tables. Infrastructure is Terraform. Weekly Monday load inside pipeline_load_window was decided; it is not built. HOUSEHOLD_HEALTH objects are not in production. No patient data is shown.',
  caption:
    'Ingestion path from patient authorization to Snowflake raw tables. Built through sandbox auth; awaiting production consent and first load. No patient data is shown.',
  status: 'Sandbox auth tested; prod load not built',
  chip: 'HOUSEHOLD_HEALTH planned',
  nodes: [
    'MyChart (FHIR R4)',
    'Patient authorization (OAuth)',
    'Token vault (Lambda, DynamoDB, SSM)',
    'Scheduled load (Airflow)',
    'Internal stage',
    'Load procedures',
    'HOUSEHOLD_HEALTH.RAW (VARIANT)',
  ],
}

COPY.industries = {
  line: 'The same pattern also covers Healthcare, Logistics, Legal, Finance and Operations, and Public Sector.',
  link: 'See all industries',
  href: 'https://www.myorglab.com/intelligence',
}

DIAG.mychart = COPY.mychart
