// Portfolio Snowflake section rewritten from the as-built in
// myorglab-repos/household-data-architecture (synced to household-platform
// fdff221 via HDA PR #12 - 20 DAG pages). Real databases, MCP servers, catalogs,
// EAIs, and DAG names. No secrets, PATs, or account locators. Status language
// matches the as-built: production vs paused; source-data gaps called out honestly.

export const COPY = {
  section: {
    kicker: 'Snowflake, agents, and MCP as personal infrastructure',
    title: 'Built for life, not just the office',
    intro:
      'The lead project is the household data platform I run on Snowflake. It has five production databases for finance, YouTube, AI agents, shared operations and health records. Assistant tools can only reach approved data, and every change takes a careful path from pull request to production. The tiles below show how changes ship, how data feeds land, how models and schedules run, how access and cost are controlled, and how agents connect. A shorter list of other shipped work follows.',
  },
  lead: {
    title: 'Household Snowflake Data Architecture and MCP',
    oneLine:
      '5 prod databases · 3 MCP servers · 20 Airflow DAG files · Monarch, Gmail, Drive, YouTube, Epic MyChart · AI chat history in AGENTIC_WORKFORCE · 15 semantic views · 6 Cortex Search services. Agents hit MCP catalogs, not an open warehouse login.',
    tags: [
      'Snowflake',
      'HOUSEHOLD_LEDGER',
      'AGENTIC_WORKFORCE',
      'Airflow',
      'dbt',
      'Terraform',
      'Cortex',
      'MCP',
      'Monarch',
      'Gmail',
    ],
    stats: [
      { n: '5', label: 'prod databases' },
      { n: '3', label: 'MCP servers' },
      { n: '20', label: 'Airflow DAG files' },
      { n: '4', label: 'warehouses' },
      { n: '15', label: 'semantic views' },
      { n: '6', label: 'Cortex Search services' },
    ],
    intro:
      'This is the household data platform I build and run on Snowflake. It has five production databases. HOUSEHOLD_LEDGER holds finance. MOONSHOTS holds the YouTube catalog for Moonshots by Peter Diamandis and Market Mondays by Earn Your Leisure. AGENTIC_WORKFORCE holds AI agents and the chat history captured across every platform I use. CONTROL holds shared procedures, queues and tests. HOUSEHOLD_HEALTH holds Epic MyChart records, loaded every Monday through SMART on FHIR R4. Each database keeps its layers as schemas instead of splitting them into separate databases. Airflow on my office PC runs every schedule, and Snowflake tasks are retired. Monarch, Gmail and Google Drive feed the ledger. Every bank account is set to send a Gmail alert for each transaction, so new charges show up early. YouTube feeds Moonshots. Grok bots and Cursor never get an open warehouse login: they connect through MCP servers that only expose semantic views, Cortex Search services and approved procedures.',
  },
  tiles: [
    {
      id: 'workflow',
      n: '2',
      title: 'Ship path: one trunk, a release path per platform',
      svg: 'workflow',
      body:
        'Every change follows the same trunk: a branch, a pull request, the required checks, then a merge that only lands code. After that, each platform has its own way to production. Snowflake infrastructure changes through Terraform. The plan runs on the pull request with a guard that fails on any delete, and the apply is a manual run against production after my explicit go. dbt checks run on DuckDB and in a temporary per-PR schema, and the models run in production from Airflow’s 06:00 ET chain. Airflow cuts a release tag after its tests pass, and a person pulls that tag onto the office PC, which swaps back on its own if anything fails to import. The websites I manage ship through the same trunk, and the one shown here, the MyOrgLab public site, deploys only from a manual dispatch with a required reviewer. Pick a platform chip to see its full flow.',
    },
    {
      id: 'pipes',
      choices: [
        { t: 'Local OCR instead of Cortex', d: 'doc_extract reads text with pypdf, then Tesseract, so reading a document never spends Cortex credits.' },
        { t: 'One morning load window', d: 'INGEST_WH wakes once at 07:15 ET for every feed and suspends when the window ends, instead of waking for each feed.' },
      ],
      n: '3',
      title: 'Source feeds into Snowflake',
      svg: 'pipes',
      body:
        'Each feed lands in its own product database, not a shared dump. Morning ingest runs once at 07:15 ET on INGEST_WH. Monarch and email load every day. YouTube loads on Mondays and then starts the sync for Moonshots by Peter Diamandis and Market Mondays by Earn Your Leisure. Epic MyChart loads on Mondays in the same window. Monarch comes in through EAI_MONARCH. Every bank account is set to send a Gmail alert for each transaction; those emails come in through EAI_GMAIL, get classified, and are matched to transactions by reconcile_pending. drive_poll picks up new Google Drive documents at the same minute without waking a warehouse, then doc_extract reads the text with pypdf and local Tesseract instead of Cortex OCR. AI chat history from every platform lands in AGENTIC_WORKFORCE through a daily capture and a nightly chat backup. Epic MyChart SMART on FHIR R4 records land in HOUSEHOLD_HEALTH, whose database, schemas, roles and grants are managed in Terraform. Every feed follows the same shape: extract and land, transform with dbt, then test. There is no S3 or Snowpipe path; that design was dropped.',
    },
    {
      id: 'airflow',
      choices: [
        { t: 'Airflow instead of Snowflake tasks', d: 'Scheduling runs as Airflow DAGs instead of Snowflake tasks, which keeps warehouse time down.' },
        { t: 'Drive polling on the load window', d: 'drive_poll runs on the 07:15 ET window, so new documents share the one INGEST_WH wake instead of starting their own.' },
      ],
      n: '4',
      title: 'Airflow: the production DAGs',
      svg: 'airflow',
      body:
        'Airflow on the office PC runs every production schedule, and Snowflake tasks are retired. Of 20 DAG files, 16 run in production and 3 are dev twins with no schedule. Four schedules start the day. At 06:00 ET ledger_dbt_daily builds the ledger models on LEDGER_WH, then runs hr_dbt_daily and agentic_workforce_dbt_daily and waits for each, so all three share one warehouse window. At 07:15 ET pipeline_load_window wakes INGEST_WH once and runs the Gmail, Monarch and, on Mondays, Epic MyChart and Earn Your Leisure loads back to back before suspending it, then starts moonshots_yt_sync_daily on Mondays. monarch_load, ledger_email_alerts, epic_fhir_load_manual and moonshots_eyl_sync also stay as DAGs for on-demand runs. Also at 07:15 ET, drive_poll checks Google Drive. drive_poll has no Snowflake connection. It asks the Drive Changes API for everything since a saved page token, keeps PDFs and images in the watched snowflake_docs folders, and skips Google-native files. If nothing changed, it moves the token and stops, so an empty poll costs no warehouse time. If files changed, it starts doc_extract with their IDs and checksums, and moves the token only after that trigger succeeds, so a failed or missed run picks up the same changes next time. doc_extract downloads the files, reads the PDF text layer or runs Tesseract on the PC, and loads the text through two owner\'s-rights procedures on INGEST_WH. Two office-PC checks run without a warehouse: nightly_app_backup at 19:33 ET and pc_usage_check three times on weekdays. platform_smoke, moonshots_transcript_backfill and ledger_email_pull_poll run only by hand. Code reaches the scheduler only from a tagged release that a person pulls onto the PC.',
    },
    {
      id: 'dbt',
      n: '5',
      title: 'dbt: layers and what starts them',
      svg: 'dbt',
      body:
        'RAW holds data exactly as it landed. CORE holds clean business objects and is often written in the same step as RAW. dbt builds staging models, marts and ANALYTICS on top of CORE, and the 15 semantic views sit on ANALYTICS. Three things start a dbt run. The schedule: at 06:00 ET, ledger_dbt_daily runs dbt on LEDGER_WH, then triggers hr_dbt_daily and agentic_workforce_dbt_daily and waits for both, so everything fits in one LEDGER_WH window. GitHub: every pull request builds the project on DuckDB and in a temporary per-PR schema, and a merge to main publishes the production manifest so later pull requests only build what changed. Manual runs: dev copies target hl_dev and the *_DEV databases and never run on a schedule. The databases, schemas and warehouse dbt builds into come from Terraform, which has its own block.',
    },
    {
      id: 'terraform',
      n: '6',
      title: 'Terraform: Snowflake infrastructure and recovery',
      svg: 'terraform',
      body:
        'Terraform in infra/terraform/snowflake owns the shape of the Snowflake account: databases, schemas, roles and grants, warehouses, resource monitors and policies. That includes HOUSEHOLD_HEALTH in household_health.tf: the database, its schemas, roles and grants, and the RAW and CTRL tables that the Monday Epic MyChart load writes to. Procedures and tasks stay with the loaders that run them. Dev and prod share one state, so a single plan shows both. Every pull request runs static checks and a read-only plan, and a guard fails the job if the plan would delete or replace anything. Merging lands code only; a manual run on main re-plans, re-runs the guard and applies after my explicit go. A nightly plan opens an issue when the account drifts from the code. Recovery has three layers: the state lives in an S3 backend with versioning and a lock table, Snowflake Time Travel and UNDROP bring back dropped or changed objects, and every apply re-plans first, so a bad change is reverted by applying the previous code.',
    },
    {
      id: 'domains',
      n: '6',
      title: 'Databases: columns, tables, modeling',
      svg: 'erd',
      body:
        'Five production databases, each with a *_DEV twin that has the same schemas, tables and columns. Open a database below to see its schemas, example tables and grain. HOUSEHOLD_LEDGER has PUBLIC, RAW, CORE, CTRL, ANALYTICS and AI, plus temporary STAGING and INTERMEDIATE schemas. MOONSHOTS has RAW, CORE, CTRL and AI. AGENTIC_WORKFORCE has AGENTS, CONVERSATIONS, GRAPH and CAPTURE, plus STAGING, INTERMEDIATE, ANALYTICS, AI, TRANSFORM and SECURITY. It is where AI chat history from every platform collects, through a daily capture and a nightly chat backup, before agentic_workforce_dbt_daily builds ANALYTICS. CONTROL has PROC, QUEUE, STAGE, FLAGS, TASKS, MCP, SECURITY and TESTS. HOUSEHOLD_HEALTH is the Epic MyChart database, with RAW, STAGING, CORE and CTRL, kept apart so no ledger, Moonshots or agent role reaches health data through inheritance. In every database, a RAW row is one landed payload, a CORE row is one business object, and ANALYTICS holds reporting tables and semantic views.',
      databases: [
        {
          id: 'ledger',
          name: 'HOUSEHOLD_LEDGER',
          status: 'Production',
          oneLine: 'Finance ledger from Monarch, Gmail and Google Drive Docs',
          schemas: ['PUBLIC', 'RAW', 'CORE', 'CTRL', 'ANALYTICS', 'AI', 'STAGING', 'INTERMEDIATE'],
          detail:
            'Landing and conformed finance data. CTRL hosts MCP_HOUSEHOLD_LEDGER_GROK and MCP_HOUSEHOLD_LEDGER. ANALYTICS holds reporting grain and the 15 semantic views; Cortex Search CS_DOCUMENTS, CS_EMAIL_MESSAGES, CS_GAPS_IDEA, CS_PARSE_REVIEW sit on ledger text. Example path: DOCUMENTS_RAW → DOCUMENTS → FACT_DOCUMENT_STANDING → SV_DOCUMENT_STANDING / CS_DOCUMENTS.',
          objects: [
            { layer: 'RAW', name: 'DOCUMENTS_RAW', note: 'One landed file / payload' },
            { layer: 'CORE', name: 'DOCUMENTS', note: 'One business document' },
            { layer: 'ANALYTICS', name: 'FACT_DOCUMENT_STANDING', note: 'Reporting grain' },
            { layer: 'Catalog', name: 'semantic views / Cortex Search services', note: 'Analyst + Search' },
          ],
        },
        {
          id: 'moonshots',
          name: 'MOONSHOTS',
          status: 'Production',
          oneLine: 'YouTube catalog for Moonshots by Peter Diamandis and Market Mondays by Earn Your Leisure',
          schemas: ['RAW', 'CORE', 'CTRL', 'AI'],
          detail:
            'YouTube domain for Moonshots by Peter Diamandis and Market Mondays by Earn Your Leisure, shortened to EYL. Monday pipeline_load_window triggers moonshots_yt_sync_daily after the INGEST_WH work; EYL captions write in that window. CTRL hosts MCP_MOONSHOTS for Cynthia (Moonshot and architecture advisory). CS_TRANSCRIPTS and CS_VIDEOS are the Moonshots Cortex Search services. Phase 5 added ANALYTICS views over CORE for reporting reads.',
          objects: [
            { layer: 'Scheduler', name: 'moonshots_yt_sync_daily', note: 'Monday window trigger' },
            { layer: 'Scheduler', name: 'moonshots_eyl_sync', note: 'Monday window step' },
            { layer: 'CTRL', name: 'MCP_MOONSHOTS', note: 'YouTube MCP catalog' },
            { layer: 'Search', name: 'CS_TRANSCRIPTS / CS_VIDEOS', note: 'Cortex Search' },
          ],
        },
        {
          id: 'workforce',
          name: 'AGENTIC_WORKFORCE',
          status: 'Production',
          oneLine: 'Cross-platform AI chat history, with agents, conversations, graph and capture',
          schemas: [
            'AGENTS',
            'CONVERSATIONS',
            'GRAPH',
            'CAPTURE',
            'STAGING',
            'INTERMEDIATE',
            'ANALYTICS',
            'AI',
            'TRANSFORM',
            'SECURITY',
          ],
          detail:
            'Where multi-platform AI history accumulates. A daily capture routine MERGEs agents, conversations, graph, and capture tables. A nightly chat backup MERGEs RAW.CHAT_MESSAGES as SVC_BOX_PIPELINE under R_PIPELINE. agentic_workforce_dbt_daily builds tag:agentic_workforce into AGENTIC_WORKFORCE.ANALYTICS, with AGENTIC_WORKFORCE_DEV as its dev twin. It has no cron of its own and starts after hr_dbt_daily in the 06:00 ledger_dbt_daily chain. Desk marts for the Desks app are still queued and not counted as shipped.',
          objects: [
            { layer: 'Capture', name: 'AGENTS / CONVERSATIONS / GRAPH', note: 'Daily MERGE' },
            { layer: 'Landing', name: 'RAW.CHAT_MESSAGES', note: 'Nightly chat backup' },
            { layer: 'dbt', name: 'agentic_workforce_dbt_daily', note: '→ ANALYTICS · 06:00 chain' },
            { layer: 'Planned', name: 'AW desk marts', note: 'Queued, not prod' },
          ],
        },
        {
          id: 'control',
          name: 'CONTROL',
          status: 'Production',
          oneLine: 'Shared control plane for procs, queues, MCP and tests',
          schemas: ['PROC', 'QUEUE', 'STAGE', 'FLAGS', 'TASKS', 'MCP', 'SECURITY', 'TESTS'],
          detail:
            'Cross-product control database. Procedures, queues, stages, feature flags, retired Snowflake tasks inventory, MCP registration, security objects, and test-result tables used by the test-results dashboard. It is not a customer domain database. It is the ops and governance plane for the account.',
          objects: [
            { layer: 'PROC', name: 'Pipeline procedures', note: 'Loader-owned' },
            { layer: 'QUEUE / STAGE', name: 'Work queues + stages', note: 'Ingest support' },
            { layer: 'MCP', name: 'Server registration', note: 'Catalog bind' },
            { layer: 'TESTS', name: 'Test-result tables', note: 'Dashboard reads' },
          ],
        },
        {
          id: 'health',
          name: 'HOUSEHOLD_HEALTH',
          status: 'Production',
          oneLine: 'Epic MyChart FHIR R4 · Terraform-managed · weekly load',
          schemas: ['RAW', 'STAGING', 'CORE', 'CTRL'],
          detail:
            'Production healthcare database on the same product-database pattern as the ledger. Terraform in household_health.tf owns the database, its schemas, roles, grants, RAW.FHIR_RESOURCES and the CTRL run log. SMART on FHIR R4 from Epic MyChart uses a registered patient app, a hosted OAuth callback, and refresh tokens kept in SSM. Every Monday the epic_fhir step of pipeline_load_window pulls each connected org, lands the resources in one PUT and COPY, runs dbt for the health models, and runs watchers that email counts only. epic_fhir_load_manual runs the same path on demand. Patient values never appear on this site.',
          objects: [
            { layer: 'Auth', name: 'SMART app, hosted callback, SSM tokens', note: 'Epic app 64675' },
            { layer: 'Scheduler', name: 'pipeline_load_window, Mondays', note: 'Production step, plus an on-demand rerun' },
            { layer: 'Stage', name: 'RAW.FHIR_LAND_STAGE', note: 'One PUT and COPY per load' },
            { layer: 'RAW', name: 'RAW.FHIR_RESOURCES', note: 'Terraform-managed · dbt builds STAGING and CORE' },
          ],
        },
      ],
    },
    {
      id: 'mcp',
      n: '7',
      title: 'MCP servers, agents, and tools',
      svg: 'mcp',
      body:
        'Three MCP servers connect agents to tools. MCP_HOUSEHOLD_LEDGER_GROK is the read server that Grok desks use. It gives household_finance_agent and ledger_ops_agent analyst tools on the 15 semantic views, search over CS_DOCUMENTS, CS_EMAIL_MESSAGES, CS_GAPS_IDEA and CS_PARSE_REVIEW, and document listing and download. MCP_HOUSEHOLD_LEDGER is the write server that Cursor and Eta (Household COO) use. It has the same reads plus approved procedures for Monarch updates, ledger changes, the Gmail and document pipelines, reconciliation, analytics rebuilds, freshness checks and metal prices. MCP_MOONSHOTS is a separate YouTube server for Cynthia (Moonshot and architecture advisory), with video and channel lookups, transcript and video search over CS_TRANSCRIPTS and CS_VIDEOS, sync status, and sync and embedding jobs. The Cortex agents AGENT_HOUSEHOLD_FINANCE, AGENT_HOUSEHOLD_FINANCE_RO and AGENT_LEDGER_OPS sit behind these servers on the same certified data. Clients sign in with a Snowflake access token under the MCP_LEDGER_CLIENT role, and a read client never gets write tools.',
      rows: [
        {
          name: 'MCP_HOUSEHOLD_LEDGER_GROK',
          role: 'Read server for the Grok desks. Agents: household_finance_agent, ledger_ops_agent. Tools: analyst_* on the 15 semantic views, search_* on 4 Cortex Search services, document stage list and download.',
          status: 'Read only',
        },
        {
          name: 'MCP_HOUSEHOLD_LEDGER',
          role: 'Write server for Cursor and Eta (Household COO). Same read families plus proc_monarch_*, proc_ledger_mutate, proc_gmail_ingest, proc_email_pipeline, proc_reconcile_pending, document land/extract/jobs, proc_rebuild_analytics, freshness / QA, proc_metal_spot.',
          status: 'Read + write',
        },
        {
          name: 'MCP_MOONSHOTS',
          role: 'YouTube server for Cynthia (Moonshot and architecture advisory). Catalog get and list, transcript and video search by keyword, semantic match or Cortex Search, sync status, and sync, embed and search refresh jobs. No household money.',
          status: 'Domain MCP',
        },
        {
          name: 'Agent layer',
          role: 'MCP agents household_finance_agent + ledger_ops_agent; Cortex agents AGENT_HOUSEHOLD_FINANCE, AGENT_HOUSEHOLD_FINANCE_RO, AGENT_LEDGER_OPS on semantic views / Cortex Search services.',
          status: 'Orchestration',
        },
        {
          name: 'Ledger tool families',
          role: 'analyst_* · search_* · document_* · monarch_* · ledger_mutate · gmail/email pipeline · reconcile · rebuild_analytics · freshness/QA · metal_spot',
          status: 'Summarized',
        },
        {
          name: 'Moonshots tool families',
          role: 'get/list catalog · search transcripts/videos · CS_TRANSCRIPTS / CS_VIDEOS · youtube_sync · embed_transcripts · refresh CS',
          status: 'Summarized',
        },
      ],
    },
    {
      id: 'security',
      choices: [
        { t: 'Batch warehouse work', d: 'Every desk follows one standard: wake a warehouse once for a batch of work, skip row-by-row inserts, and tag every query.' },
        { t: 'A separate warehouse for bot questions', d: 'Grok service users default to GROK_WH under its own RM_GROK monitor, so a question from a bot never wakes the ledger warehouse.' },
        { t: 'CI results in one insert', d: 'CI test results are written in one INSERT per run instead of one per test.' },
        { t: 'One ledger warehouse', d: 'LEDGER_WH is not split into a second warehouse yet, so ledger work still wakes and bills a single warehouse.' },
      ],
      n: '8',
      title: 'Roles, warehouses & cost',
      svg: 'security',
      body:
        'Each kind of work gets its own warehouse, so cost stays easy to see and easy to cap. LEDGER_WH runs the morning dbt builds for the R_PIPELINE pipelines. INGEST_WH wakes once at 07:15 ET for the office-PC loads under R_PC_PIPELINE, and YT_SYNC_WH runs the YouTube sync under the same role. GROK_WH is the default for the Grok bots, so a question from a bot never wakes the ledger warehouse. Every warehouse goes to sleep within a minute of going idle and has its own monitor. Account-wide caps sit on top: an account monitor, a Cortex AI budget that cuts off agent access at 100%, and one role, R_CORTEX_GATE, that is allowed to call Cortex. People and bots reach data through MCP servers, never an open warehouse login. Reading documents and audio happens on the office PC, so it stays off the Snowflake bill. The table lists every role and what it is for.',
    },
    {
      id: 'grokbot',
      n: '9',
      title: 'Grokbot architecture, agents per MCP',
      svg: 'grokbot',
      body:
        'This shows who talks to the household data and through which door. Stephen (Household CFO) and the Grok finance desks use the read server, MCP_HOUSEHOLD_LEDGER_GROK, for spending, cash flow, balances and document search, and Stephen is the only one who answers money questions. Eta (Household COO) reads too, and she uses the write server, MCP_HOUSEHOLD_LEDGER, when she relays Michael’s go for a change. Cursor uses the write server for ledger and Monarch changes, pipeline runs and rebuilds. Cynthia (Moonshot advisory) uses MCP_MOONSHOTS for the YouTube catalog and transcript search. Behind the ledger servers, three Cortex agents answer from 15 semantic views and 6 Cortex Search services.',
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
        'Customer-facing practice built from the same pattern as HOUSEHOLD_LEDGER: in-account Cortex agents, semantic views, and MCP bridges. It is not a multi-tenant copy of customer data.',
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

// Diagram labels use the same real object names. Layer label strings are consumed by architectureSvg.js - keep Clients / MCP servers / Cortex / Catalogs / Snowflake.
export const DIAG = {
  lead: {
    title:
      'Lead diagram: agents through MCP_HOUSEHOLD_LEDGER_GROK and MCP_HOUSEHOLD_LEDGER into semantic views / Cortex Search services and HOUSEHOLD_LEDGER schemas',
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
          { t: 'Analyst', s: 'text to SQL on semantic views' },
          { t: 'Search', s: 'Cortex Search services' },
        ],
      },
      {
        label: 'Catalogs',
        nodes: [
          { t: 'semantic views', e: 1 },
          { t: 'Cortex Search services', e: 1 },
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
        nodes: [{ t: 'Monarch GraphQL' }, { t: 'Gmail · EAI_GMAIL' }, { t: 'Google Drive Docs' }],
      },
      {
        label: 'Extract and land',
        nodes: [
          {
            t: 'Airflow',
            s: 'production scheduler on pinned releases',
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
        nodes: [{ t: 'ANALYTICS + semantic views' }, { t: 'MCP servers', e: 1 }],
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
    title: 'Warehouses by job, with the roles that use them and how each stops spending',
    lanes: [
      { wh: { t: 'LEDGER_WH', s: 'ledger and dbt work', logo: 'snowflake', hero: true }, who: { t: 'Pipelines · R_PIPELINE', s: 'ledger_dbt_daily at 06:00 ET builds the ledger models', logo: 'dbt' }, stop: { t: 'Sleeps after 30s', target: 'LEDGER_WH_MONITOR' } },
      { wh: { t: 'INGEST_WH', s: 'morning loads', logo: 'snowflake', hero: true }, who: { t: 'Office PC · R_PC_PIPELINE', s: 'one 07:15 ET window for email, Monarch and documents', logo: 'airflow' }, stop: { t: 'Sleeps after 60s', target: 'INGEST_MONITOR' } },
      { wh: { t: 'GROK_WH', s: 'bot questions', logo: 'snowflake' }, who: { t: 'Grok bots · SVC_GROK_*', s: 'the read, write and YouTube connectors default here', logo: 'grokbot' }, stop: { t: 'Own monitor', target: 'RM_GROK' } },
      { wh: { t: 'YT_SYNC_WH', s: 'YouTube sync', logo: 'snowflake' }, who: { t: 'Office PC · R_PC_PIPELINE', s: 'moonshots_yt_sync_daily and transcript embeddings', logo: 'youtube' }, stop: { t: 'Sleeps after 60s', target: 'YT_SYNC_MONITOR' } },
    ],
    caps: [
      { t: 'Account monitor', s: 'one credit cap over the whole account' },
      { t: 'Cortex AI budget', s: 'cuts off agent access at 100%', gate: 'BUDGET_CORTEX_AI_20' },
      { t: 'One Cortex role', s: 'only R_CORTEX_GATE can call Cortex' },
      { t: 'Dev stays apart', s: 'R_DEV_PIPELINE runs on DEV_MIGRATION_WH' },
    ],
    roles: [
      { group: 'Pipelines', items: [
        ['R_PIPELINE', 'Airflow pipelines on the box and office PC: loads and dbt', 'LEDGER_WH'],
        ['R_PC_PIPELINE', 'Office-PC loads: the morning window and YouTube sync', 'INGEST_WH, YT_SYNC_WH'],
      ] },
      { group: 'Owners', items: [
        ['LEDGER_OWNER', 'Owns the household ledger tables', 'LEDGER_WH'],
        ['MOONSHOTS_OWNER', 'Owns the YouTube research tables', 'LEDGER_WH'],
        ['R_CTRL_OWNER', 'Owns control procedures and tasks; no person holds it', 'Not set'],
      ] },
      { group: 'Bots and MCP', items: [
        ['R_GROK_LEDGER_RO', 'Grok read connector for Stephen and the finance desks', 'GROK_WH'],
        ['R_GROK_LEDGER_RW', 'Grok write connector for Eta', 'GROK_WH'],
        ['MCP_LEDGER_CLIENT', 'Signs in to the ledger MCP servers', 'LEDGER_WH'],
        ['MCP_MOONSHOTS_CLIENT', 'YouTube MCP connector used by Cynthia', 'GROK_WH'],
      ] },
      { group: 'Cortex', items: [
        ['R_CORTEX_GATE', 'The only role allowed to call Cortex AI', 'Not set'],
        ['R_AI_USER', 'Read-only analytics for MCP and agents, through the gate', 'Not set'],
        ['R_AI_ADMIN', 'Manages Cortex objects', 'Not set'],
        ['R_MOONSHOTS_AI_ADMIN', 'Manages the YouTube AI objects', 'Not set'],
      ] },
      { group: 'Builds and admin', items: [
        ['R_DBT_PROD', 'Production dbt builds into ANALYTICS', 'Not set'],
        ['R_DBT_CI', 'Pull-request dbt builds in temporary schemas', 'Not set'],
        ['R_TF_ADMIN', 'Applies Terraform changes', 'TF_WH'],
        ['R_SEC_ADMIN', 'Manages security policies', 'Not set'],
      ] },
      { group: 'Audit and readers', items: [
        ['R_AUDIT_OWNER', 'Owns the audit trail; no person holds it', 'Not set'],
        ['R_AUDIT_READ', 'Reads the audit trail for reviews', 'Not set'],
        ['R_TEST_RESULTS_READ', 'Reads CI test results', 'GROK_WH'],
        ['R_AW_DESKS_READ', 'Reads agent workforce analytics', 'Not set'],
        ['R_EYL_SVC', 'Read-only service access', 'Not set'],
      ] },
      { group: 'Dev', items: [
        ['R_DEV_PIPELINE', 'Dev pipelines into the _DEV databases', 'DEV_MIGRATION_WH'],
        ['R_AW_DESKS_READ_DEV', 'Dev agent workforce analytics', 'DEV_MIGRATION_WH'],
        ['R_TEST_RESULTS_READ_DEV', 'Dev test results', 'Not set'],
      ] },
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
      { label: 'Scheduler', nodes: [{ t: 'Airflow', s: 'Docker release pin' }] },
      {
        label: 'Infrastructure and CI',
        nodes: [{ t: 'Terraform' }, { t: 'GitHub Actions' }],
      },
    ],
  },

  mcpTools: {
    title: 'MCP servers, agents, and tool families',
    note: '',
    servers: [
      {
        t: 'MCP_HOUSEHOLD_LEDGER_GROK',
        s: 'Read · Grok desks',
        families: 'agents · analyst_* · search_* · docs',
        z: 'agent',
      },
      {
        t: 'MCP_HOUSEHOLD_LEDGER',
        s: 'Write · Cursor / Eta (Household COO)',
        families: 'read families + mutate / pipeline procs',
        z: 'agent',
      },
      {
        t: 'MCP_MOONSHOTS',
        s: 'YouTube · Cynthia (Moonshot advisory)',
        families: 'catalog · search · sync / embed',
        z: 'src',
      },
    ],
    agents: [
      ['household_finance_agent', 'ledger_ops_agent'],
      ['same agents + write procs', 'Cortex AGENT_* behind'],
      ['Cynthia (Moonshot advisory)', 'YouTube catalog tools'],
    ],
  },
  grokbot: {
    title: 'Who uses each MCP server, what it offers, and the Cortex agents behind it',
    people: [
      { t: 'Stephen', s: 'Household CFO', logo: 'grokbot', to: [0] },
      { t: 'Grok desks', s: 'finance', logo: 'grokbot', to: [0] },
      { t: 'Eta', s: 'Household COO', logo: 'grokbot', to: [0, 1] },
      { t: 'Michael', s: 'gives the go', logo: 'person', go: true },
      { t: 'Cursor', s: 'coding agent', logo: 'person', to: [1] },
      { t: 'Cynthia', s: 'Moonshot advisory', logo: 'grokbot', to: [2] },
    ],
    servers: [
      { t: 'Read server', s: 'MCP_HOUSEHOLD_LEDGER_GROK', logo: 'mcp', hero: true, users: 'Stephen, Grok desks, Eta' },
      { t: 'Write server', s: 'MCP_HOUSEHOLD_LEDGER', logo: 'mcp', hero: true, users: 'Eta, Cursor' },
      { t: 'YouTube server', s: 'MCP_MOONSHOTS', logo: 'mcp', users: 'Cynthia' },
    ],
    families: [
      { t: 'Finance and ops answers', s: 'household_finance_agent, ledger_ops_agent, analyst and search tools' },
      { t: 'The same reads, plus changes', s: 'ledger and Monarch changes, pipeline runs, rebuilds' },
      { t: 'YouTube research', s: 'catalog, transcript search, sync and embed', logo: 'youtube' },
    ],
    cortex: { t: 'Three Cortex agents', s: 'AGENT_HOUSEHOLD_FINANCE, its read-only twin and AGENT_LEDGER_OPS, on 15 semantic views and 6 Cortex Search services', logo: 'snowflake' },
  },
  erd: {
    title:
      'Five prod databases including HOUSEHOLD_HEALTH. Select a database in the open panel for its schemas',
    areas: [
      'HOUSEHOLD_LEDGER',
      'MOONSHOTS',
      'AGENTIC_WORKFORCE',
      'CONTROL',
      'HOUSEHOLD_HEALTH',
    ],
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
        meaning: 'Search catalog. Indexing stays on even when serving is suspended',
      },
    ],
  },
  workflow: {
    title: 'Ship path: a shared pull request trunk, then one release path per platform',
    trunk: [
      { t: 'Branch + pull request', s: 'household-platform · website repos' },
      { t: 'Required checks', s: 'ruff + pytest · gitleaks · static · plan' },
      { t: 'Merge to main', s: 'lands code only' },
    ],
    platforms: [
      {
        id: 'terraform',
        label: 'Terraform · Snowflake',
        logo: 'terraform',
        path: 'infra/terraform/snowflake',
        merge: 'code only · nightly drift check',
        summary: 'static checks and a read-only plan on the PR, a 0-destroy guard, then a manual apply on main that re-plans and guards again.',
        pr: [
          { t: 'static', s: 'fmt · validate · tflint · migration check' },
          { t: 'plan', s: 'read-only OIDC role · plan posted to the PR' },
          { t: '0-destroy guard', s: 'any delete or replace fails the job', gate: 'override needs a go' },
        ],
        post: [
          { t: 'Manual apply', s: 'apply=true · production · re-plan and guard again', gate: 'Michael’s go' },
        ],
      },
      {
        id: 'dbt',
        label: 'dbt',
        logo: 'dbt',
        path: 'dbt/ models, tests, seeds',
        merge: 'publishes the prod manifest',
        summary: 'PR CI on DuckDB and a per-PR schema, merge publishes the prod manifest, models run in the 06:00 ET chain from the release the PC pulled.',
        pr: [
          { t: 'DuckDB build', s: 'synthetic seeds · no Snowflake access' },
          { t: 'Per-PR schema', s: 'HOUSEHOLD_LEDGER_CI.DBT_PR_<n> · unit tests' },
          { t: 'Schema cleanup', s: 'dropped on close · nightly sweep' },
        ],
        post: [
          { t: 'Prod manifest', s: 'dbt parse on main for slim CI' },
          { t: 'ledger_dbt_daily', s: '06:00 ET in Airflow · then HR and workforce' },
        ],
      },
      {
        id: 'airflow',
        label: 'Airflow',
        logo: 'airflow',
        path: 'airflow/dags and the PC image',
        merge: 'tags a release when tests pass',
        summary: 'DAG and image tests, release tag on main, a person pulls the tag onto the office PC, auto swap back on import errors.',
        pr: [
          { t: 'DAG import tests', s: 'every DAG must import cleanly' },
          { t: 'PC image check', s: 'PC Dockerfile built · no import errors' },
        ],
        post: [
          { t: 'Release tag', s: 'airflow-YYYYMMDD-sha after both tests pass' },
          { t: 'PC pull', s: 'deploy_airflow.ps1 · dry run unless -Apply', gate: 'run by a person' },
          { t: 'Swap and check', s: 'import errors swap back · rollback is the last tag' },
        ],
      },
      {
        id: 'website',
        label: 'Websites',
        logo: 'github',
        path: 'website repos · MyOrgLab shown',
        merge: 'builds and tests, no deploy',
        summary: 'shown for MyOrgLab. Lint, build and consent tests on PRs, and deploys only by dispatch through a production environment with a required reviewer.',
        pr: [
          { t: 'Lint and build', s: 'lint · production build' },
          { t: 'Consent tests', s: 'Jest · Playwright end to end' },
        ],
        post: [
          { t: 'Deploy dispatch', s: 'production environment · OIDC role', gate: 'required reviewer' },
        ],
      },
    ],
  },
}

DIAG.lead.airflow = {
  t: 'Airflow',
  s: '16 production DAGs · 3 dev twins',
}
DIAG.airflow = {
  title: 'All 16 production Airflow DAGs, one lane per warehouse',
  lanes: [
    { id: 'ledger', t: 'LEDGER_WH', s: '06:00 ET dbt window', logo: 'snowflake' },
    { id: 'ingest', t: 'INGEST_WH', s: '07:15 ET load window', logo: 'snowflake' },
    { id: 'yt', t: 'YT_SYNC_WH', s: 'Mondays', logo: 'snowflake' },
    { id: 'docs', side: 1, t: 'Google Drive documents', s: 'poll needs no warehouse' },
    { id: 'pc', side: 1, t: 'Office PC, no warehouse', s: 'checks and backups' },
    { id: 'manual', side: 1, t: 'Manual only', s: 'no schedule' },
  ],
  dags: [
    { id: 'ledger_dbt_daily', lane: 'ledger', col: 0, at: '06:00', trigger: 'schedule', by: 'cron 0 6 * * *', runs: 'Daily', writes: 'ledger models, ANALYTICS', logo: 'dbt' },
    { id: 'hr_dbt_daily', lane: 'ledger', col: 1, ncol: 1, after: 'ledger_dbt_daily', trigger: 'parent DAG', by: 'ledger_dbt_daily', runs: 'Daily', writes: 'HR marts', logo: 'dbt' },
    { id: 'agentic_workforce_dbt_daily', lane: 'ledger', col: 2, ncol: 1, after: 'hr_dbt_daily', trigger: 'parent DAG', by: 'hr_dbt_daily', runs: 'Daily', writes: 'AGENTIC_WORKFORCE analytics', logo: 'dbt' },
    { id: 'pipeline_load_window', lane: 'ingest', col: 0, at: '07:15', trigger: 'schedule', by: 'cron 15 7 * * *', runs: 'Daily', writes: 'one wake, then suspend', logo: 'airflow' },
    { id: 'ledger_email_alerts', lane: 'ingest', col: 1, ncol: 1, after: 'pipeline_load_window', trigger: 'window step', by: 'pipeline_load_window', runs: 'Daily', writes: 'HOUSEHOLD_LEDGER RAW, CORE', logo: 'gmail' },
    { id: 'monarch_load', lane: 'ingest', col: 1, ncol: 1, after: 'pipeline_load_window', trigger: 'window step', by: 'pipeline_load_window', runs: 'Daily', writes: 'HOUSEHOLD_LEDGER RAW, CORE', logo: 'monarch' },
    { id: 'moonshots_eyl_sync', lane: 'ingest', col: 1, ncol: 1, after: 'pipeline_load_window', trigger: 'window step', by: 'pipeline_load_window', runs: 'Mondays', writes: 'MOONSHOTS captions', logo: 'youtube' },
    { id: 'epic_fhir_load_manual', lane: 'ingest', col: 1, ncol: 1, after: 'pipeline_load_window', trigger: 'window step', by: 'pipeline_load_window', runs: 'Mondays', also: 'also on demand', writes: 'HOUSEHOLD_HEALTH RAW, then dbt' },
    { id: 'moonshots_yt_sync_daily', lane: 'yt', col: 2, ncol: 1, after: 'pipeline_load_window', trigger: 'parent DAG', by: 'pipeline_load_window', runs: 'Mondays', writes: 'MOONSHOTS, embeddings', logo: 'youtube' },
    { id: 'drive_poll', lane: 'docs', col: 0, at: '07:15', trigger: 'schedule', by: 'cron 15 7 * * *', runs: 'Daily', writes: 'starts doc_extract if new', logo: 'drive' },
    { id: 'doc_extract', lane: 'docs', col: 1, ncol: 1, after: 'drive_poll', trigger: 'parent DAG', by: 'drive_poll', runs: 'New files', writes: 'document text on INGEST_WH', logo: 'drive' },
    { id: 'nightly_app_backup', lane: 'pc', col: 0, at: '19:33', trigger: 'schedule', by: 'cron 33 19 * * *', runs: 'Daily', writes: 'pushes the PC bundles' },
    { id: 'pc_usage_check', lane: 'pc', col: 1, ncol: 1, at: '09:17', trigger: 'schedule', by: '13:17 · 17:17', runs: 'Weekdays', writes: 'emails on a threshold' },
    { id: 'platform_smoke', lane: 'manual', col: 0, trigger: 'manual', by: 'read-only connection check', runs: 'On demand', writes: 'counts only' },
    { id: 'moonshots_transcript_backfill', lane: 'manual', col: 1, ncol: 1, trigger: 'manual', by: 'one day per run, LEDGER_WH', runs: 'On demand', writes: 'MOONSHOTS transcripts', logo: 'youtube' },
    { id: 'ledger_email_pull_poll', lane: 'manual', col: 0, ncol: 1, trigger: 'manual', by: 'metadata count', runs: 'On demand', writes: 'no warehouse' },
  ],
}
DIAG.mcpMatrix = {
  servers: [
    { id: 'read', label: 'Read', name: 'MCP_HOUSEHOLD_LEDGER_GROK', users: ['Stephen', 'Eta'] },
    { id: 'write', label: 'Write', name: 'MCP_HOUSEHOLD_LEDGER', users: ['Eta', 'Cursor'] },
    { id: 'yt', label: 'YouTube', name: 'MCP_MOONSHOTS', users: ['Cynthia'] },
  ],
  desks: [
    { name: 'Eta (Household COO)', access: 'read and write' },
    { name: 'Stephen (Household CFO)', access: 'read' },
    { name: 'Cynthia (Moonshot and architecture advisory)', access: 'Moonshots' },
  ],
  otherClients: [{ name: 'Cursor', access: 'write server' }],
  cortexAgents: ['AGENT_HOUSEHOLD_FINANCE', 'AGENT_HOUSEHOLD_FINANCE_RO', 'AGENT_LEDGER_OPS'],
  cortexNote: 'Called as tools through household_finance_agent and ledger_ops_agent, on the 15 semantic views and Cortex Search.',
  families: [
    { id: 'agents', t: 'Cortex agent calls', kind: ['read'], n: { read: 2, write: 2 }, tools: ['household_finance_agent', 'ledger_ops_agent'] },
    { id: 'analyst', t: 'Analyst on semantic views', kind: ['read'], n: { read: 15, write: 15 }, tools: ['analyst_spending', 'analyst_transactions', 'analyst_cashflow', 'analyst_balances_net_worth', 'analyst_liabilities_dues', 'analyst_early_activity', 'analyst_budget_variance', 'analyst_investments', 'analyst_obligations', 'analyst_document_standing', 'analyst_recurring_fixed', 'analyst_accounts_receivable', 'analyst_debt_payoff', 'analyst_savings_goals', 'analyst_ops_freshness'] },
    { id: 'search', t: 'Cortex Search', kind: ['read'], n: { read: 4, write: 4 }, tools: ['search_documents', 'search_emails', 'search_gaps', 'search_parse_review'] },
    { id: 'docs', t: 'Documents', kind: ['read', 'write'], n: { read: 2, write: 8 }, tools: ['proc_document_stage_list', 'proc_document_download', 'proc_document_pipeline_folders', 'proc_drive_docs_ingest', 'proc_document_extract', 'proc_document_ingest_and_extract', 'proc_enqueue_doc_job', 'proc_get_doc_job'], readTools: 2 },
    { id: 'monarch', t: 'Monarch', kind: ['write'], n: { write: 4 }, tools: ['proc_monarch_extract', 'proc_monarch_holdings_extract', 'proc_monarch_mutate', 'proc_enqueue_monarch_extract_followup'] },
    { id: 'ledger', t: 'Ledger and reconcile', kind: ['write'], n: { write: 2 }, tools: ['proc_ledger_mutate', 'proc_reconcile_pending'] },
    { id: 'email', t: 'Email pipeline', kind: ['write'], n: { write: 2 }, tools: ['proc_gmail_ingest', 'proc_email_pipeline'] },
    { id: 'ops', t: 'Freshness, QA and rebuild', kind: ['write'], n: { write: 3 }, tools: ['proc_get_freshness', 'proc_qa_ledger_health', 'proc_rebuild_analytics'] },
    { id: 'market', t: 'Market prices', kind: ['write'], n: { write: 1 }, tools: ['proc_metal_spot'] },
    { id: 'async', t: 'Async agent runs', kind: ['read'], n: { write: 3 }, tools: ['ASYNC_AGENT_RUN_STATUS', 'AGENT_RUN_RESULT', 'ASYNC_AGENT_RUN_CANCEL'] },
    { id: 'ytcat', t: 'YouTube catalog', kind: ['read'], n: { yt: 6 }, tools: ['get_channel', 'list_videos', 'get_video', 'get_video_topics', 'list_transactions', 'get_catalog_counts'] },
    { id: 'ytsearch', t: 'Transcript and video search', kind: ['read'], n: { yt: 6 }, tools: ['search_videos', 'search_transcripts', 'search_transcripts_semantic', 'search_videos_semantic', 'search_cs_transcripts', 'search_cs_videos'] },
    { id: 'ytstatus', t: 'Sync status', kind: ['read'], n: { yt: 2 }, tools: ['get_youtube_sync_status', 'get_transcript_coverage'] },
    { id: 'ytjobs', t: 'Sync and embedding jobs', kind: ['jobs'], n: { yt: 3 }, tools: ['proc_youtube_sync', 'proc_embed_transcripts', 'proc_refresh_cs_transcripts'] },
  ],
}
DIAG.dbt = {
  title: 'What starts dbt and the layers it builds',
  triggers: [
    { t: 'Schedule', s: 'ledger_dbt_daily at 06:00 ET', target: 'hl_prod', logo: 'airflow', hero: true },
    { t: 'GitHub', s: 'pull request on DuckDB and a per-PR schema · main publishes the manifest', target: 'DBT_PR_<n>', logo: 'github' },
    { t: 'Manual', s: 'dev runs into the *_DEV databases, never scheduled', target: 'hl_dev', logo: 'person' },
  ],
  layers: [
    { t: 'RAW', s: 'data as it landed', cyl: true },
    { t: 'CORE', s: 'business objects, often written with RAW', cyl: true },
    { t: 'STAGING and marts', s: 'dbt models on CORE', dbt: true },
    { t: 'ANALYTICS and semantic views', s: '15 Analyst views', dbt: true, hero: true },
  ],
}
DIAG.terraform = {
  title: 'Terraform for Snowflake: what it owns, how it applies, how it recovers',
  apply: [
    { t: 'plan', s: 'read-only OIDC role · dev and prod in one plan' },
    { t: '0-destroy guard', s: 'any delete or replace fails the job', gate: 'override needs a go' },
    { t: 'Manual apply', s: 'on main · re-plan and guard again', gate: 'Michael’s go', hero: true },
  ],
  recover: [
    { t: 'Versioned state', s: 'S3 backend with versioning and a lock table' },
    { t: 'Time Travel and UNDROP', s: 'dropped or changed objects come back' },
    { t: 'Nightly drift check', s: 'opens an issue when Snowflake and code disagree' },
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
  airflow: '16 production DAGs · runs the pulled release tag',
  land: 'pipeline_load_window → monarch + email',
  ocr: 'doc_extract · local OCR on the office PC',
  dbt: 'ledger_dbt_daily → hr → agentic',
}

Object.assign(DIAG.pipes, {
  airflow: DIAG.pipes.layers[1].nodes[0],
  ocr: DIAG.pipes.layers[1].nodes[1],
  dbt: DIAG.pipes.layers[3].nodes[0],
})

COPY.industries = {
  line: 'The same pattern also covers Healthcare, Logistics, Legal, Finance and Operations, and Public Sector.',
  link: 'See all industries',
  href: 'https://www.myorglab.com/intelligence',
}

DIAG.databases = Object.fromEntries(
  (COPY.tiles.find((t) => t.id === 'domains')?.databases || []).map((db) => [db.id, db]),
)
