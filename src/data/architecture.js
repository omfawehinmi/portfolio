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
      'The lead project is HOUSEHOLD_LEDGER on Snowflake: Airflow on my office PC schedules production work, dbt and procedures transform landed data, and separate MCP servers expose governed catalogs to agents. The tiles under it are how that platform is shaped. Other shipped work follows in a shorter list.',
  },
  lead: {
    title: 'HOUSEHOLD_LEDGER and MCP',
    subhead: 'Snowflake data architecture and MCP',
    oneLine:
      'A production Snowflake platform I operate: HOUSEHOLD_LEDGER, MOONSHOTS, AGENTIC_WORKFORCE, and CONTROL; Airflow as the scheduler; MCP_HOUSEHOLD_LEDGER (write) and MCP_HOUSEHOLD_LEDGER_GROK (read) so agents use semantic views and Cortex Search instead of an open warehouse login.',
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
    intro:
      'This is the household data platform I build and run. Snowflake holds four production databases: HOUSEHOLD_LEDGER (finance), MOONSHOTS (YouTube catalog), AGENTIC_WORKFORCE, and CONTROL. Layers are schemas inside those databases (RAW, CORE, CTRL, ANALYTICS, AI), not one database per layer. Self-hosted Airflow on the office PC replaced Snowflake tasks as the production scheduler. Monarch GraphQL, Gmail, and Google Drive feed the ledger through External Access Integrations. Agents never get a general warehouse login: they call MCP servers that bind Analyst views, Cortex Search, and approved procedures.',
  },
  tiles: [
    {
      id: 'mcp',
      n: '2',
      title: 'MCP_HOUSEHOLD_LEDGER split',
      svg: 'mcp',
      body:
        'HOUSEHOLD_LEDGER.CTRL hosts two MCP servers. MCP_HOUSEHOLD_LEDGER_GROK is the read catalog: household_finance_agent, ledger_ops_agent, analyst_* tools bound to SV_* views, search over CS_DOCUMENTS / CS_EMAIL_MESSAGES / CS_GAPS_IDEA / CS_PARSE_REVIEW, plus document stage list and download. MCP_HOUSEHOLD_LEDGER is the write catalog: the same read surface plus proc_monarch_extract, proc_monarch_mutate, proc_ledger_mutate, proc_gmail_ingest, proc_email_pipeline, proc_reconcile_pending, proc_document_extract, proc_rebuild_analytics, and related pipeline procs. MOONSHOTS.CTRL has a third server, MCP_MOONSHOTS, for the YouTube catalog only. Clients authenticate with a Snowflake PAT under role MCP_LEDGER_CLIENT. A read client never holds write tools.',
      caption:
        'Read path through MCP_HOUSEHOLD_LEDGER_GROK to SV_* and CS_*. Write path through MCP_HOUSEHOLD_LEDGER into approved CORE / pipeline procedures.',
    },
    {
      id: 'domains',
      n: '3',
      title: 'Databases and schemas',
      svg: 'erd',
      body:
        'Production databases: HOUSEHOLD_LEDGER, MOONSHOTS, AGENTIC_WORKFORCE, CONTROL (plus *_DEV twins). HOUSEHOLD_LEDGER schemas: PUBLIC, RAW, CORE, CTRL, ANALYTICS, AI, with transient STAGING and INTERMEDIATE. CONTROL holds PROC, QUEUE, STAGE, FLAGS, TASKS, MCP, SECURITY, TESTS. A RAW row is one landed payload (email MIME, Monarch GraphQL page, staged document). A CORE row is one business object. ANALYTICS is reporting grain and the SV_* Analyst catalogs. Several landing procedures write RAW and CORE in the same call, so CORE is not only a later rebuild of RAW. Dev uses synthetic fixtures, not a copy of production data.',
      caption:
        'Product databases hold layers as schemas. Example documents path: DOCUMENTS_RAW → DOCUMENTS → FACT_DOCUMENT_STANDING → SV_DOCUMENT_STANDING / CS_DOCUMENTS.',
    },
    {
      id: 'security',
      n: '5',
      title: 'Roles and Cortex gate',
      svg: 'security',
      body:
        'Access is split by role and by which MCP server a client is attached to. Pipeline work uses R_PIPELINE / R_PC_PIPELINE; interactive and agent clients use MCP_LEDGER_CLIENT against the MCP path, not a shared open warehouse login. R_CORTEX_GATE is the sole holder of SNOWFLAKE.CORTEX_USER. Answers are supposed to come from certified SV_* views and CS_* search services. Public copy stops at that architecture: no account locators, hostnames, or PATs.',
      caption:
        'Role separation plus MCP_HOUSEHOLD_LEDGER vs MCP_HOUSEHOLD_LEDGER_GROK, with Cortex gated behind R_CORTEX_GATE.',
    },
    {
      id: 'pipes',
      n: '4',
      title: 'Airflow pipelines and feeds',
      svg: 'pipes',
      body:
        'Production scheduling runs on Airflow on the office PC (release pins such as airflow-20261107-02cbc74), not on Snowflake tasks. Sources: Monarch via EAI_MONARCH (proc_monarch_extract), Gmail via EAI_GMAIL (proc_gmail_ingest → classify → reconcile_pending), Drive documents via drive_poll then doc_extract (pypdf text layer, then local Tesseract OCR — Cortex is not called for OCR). dbt runs through ledger_dbt_daily / dbt_daily / domain twins. Document S3 + Snowpipe intake was designed and dropped; intake stays on the Airflow path. Some DAGs are live (moonshots_yt_sync_daily, moonshots_eyl_sync confirmed unpaused on the 2026-11-07 check); others exist in git with live pause unconfirmed or intentionally paused (doc_extract pause for warehouse quota, moonshots_transcript_backfill stayed paused).',
      caption:
        'Monarch / Gmail / Drive → Airflow → HOUSEHOLD_LEDGER RAW/CORE → dbt ANALYTICS → MCP. Snowflake tasks are retired as the scheduler.',
    },
    {
      id: 'stack',
      n: '6',
      title: 'Stack',
      svg: 'stack',
      body:
        'Snowflake holds the data and the catalogs. Airflow on the office PC schedules production DAGs. dbt builds tested models (hl_prod / hl_dev targets; ANALYTICS_DBT). Terraform under infra/terraform/snowflake owns databases, schemas, roles, warehouses, and monitors; procedures stay loader-owned. GitHub Actions is the CI path. Cortex covers Agents, Analyst on SV_*, and Search on CS_*. Warehouses include LEDGER_WH, COMPUTE_WH, INGEST_WH, YT_SYNC_WH, DEV_MIGRATION_WH, ADHOC_WH. The same Terraform / dbt / Airflow / CI family is the pattern I also use in customer deployments through MyOrgLab.',
      caption:
        'Airflow, Snowflake, dbt, Terraform, GitHub Actions, with MCP as the agent interface.',
    },
    {
      id: 'features',
      n: '7',
      title: 'Catalogs and Snowflake features',
      svg: 'features',
      body:
        'Semantic views are the Analyst contracts: SV_SPENDING, SV_CASHFLOW, SV_BALANCES_NET_WORTH, SV_LIABILITIES_DUES, SV_EARLY_ACTIVITY, SV_BUDGET_VARIANCE, SV_INVESTMENTS, SV_OBLIGATIONS, SV_DOCUMENT_STANDING, SV_OPS_FRESHNESS, SV_TRANSACTIONS, SV_RECURRING_FIXED, SV_DEBT_PAYOFF, SV_ACCOUNTS_RECEIVABLE, SV_SAVINGS_GOALS. Cortex Search services: CS_DOCUMENTS, CS_EMAIL_MESSAGES, CS_GAPS_IDEA, CS_PARSE_REVIEW on the ledger; CS_TRANSCRIPTS and CS_VIDEOS on Moonshots. Cortex agents include AGENT_HOUSEHOLD_FINANCE, AGENT_HOUSEHOLD_FINANCE_RO, and AGENT_LEDGER_OPS. MCP servers in CTRL expose those catalogs as a read path or a write path. Warehouse work is batched so a warehouse wakes once per load window; interactive and agent work can sit on a warehouse separate from production loads.',
      caption:
        'SV_* and CS_* are the catalogs. MCP_HOUSEHOLD_LEDGER_GROK and MCP_HOUSEHOLD_LEDGER sit in front of them.',
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
      prev: 'rearch',
      title: 'Snowflake platform re-architecture',
      line:
        'Lift and shift of a Snowflake data platform into Terraform-managed objects, dbt models, and Airflow DAGs — the same control plane HOUSEHOLD_LEDGER runs on.',
      tags: ['Terraform', 'dbt', 'Airflow', 'Snowflake'],
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
          {
            t: 'S3 and Snowpipe',
            s: 'dropped, not running',
            d: 1,
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
  s: 'production scheduler (pinned releases)',
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
  airflow: 'drive_poll, monarch_load, ledger_dbt_daily',
  land: 'monarch_load, ledger_email_alerts',
  ocr: 'doc_extract (paused for WH quota)',
  dbt: 'ledger_dbt_daily, dbt_daily',
}

Object.assign(DIAG.pipes, {
  airflow: DIAG.pipes.layers[1].nodes[0],
  ocr: DIAG.pipes.layers[1].nodes[1],
  dropped: DIAG.pipes.layers[1].nodes[2],
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
