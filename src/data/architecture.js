// Portfolio Snowflake section rewritten from the as-built in
// myorglab-repos/household-data-architecture (synced to household-platform
// fdff221 via HDA PR #12 — 20 DAG pages). Real databases, MCP servers, catalogs,
// EAIs, and DAG names. No secrets, PATs, or account locators. Status language
// matches the as-built: production vs paused; source-data gaps called out honestly.

export const COPY = {
  section: {
    kicker: 'Projects',
    title: 'Work that shows the platform',
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
      'This is the household data platform I build and run on Snowflake. It has five production databases. HOUSEHOLD_LEDGER holds finance. MOONSHOTS holds the YouTube catalog for Moonshots by Peter Diamandis and Market Mondays by Earn Your Leisure. AGENTIC_WORKFORCE holds AI agents and the chat history captured across every platform I use. CONTROL holds shared procedures, queues and tests. HOUSEHOLD_HEALTH is built for Epic MyChart records; its model is in production, but no patient data is loaded yet. Each database keeps its layers as schemas instead of splitting them into separate databases. Airflow on my office PC runs every schedule, and Snowflake tasks are retired. Monarch, Gmail and Google Drive feed the ledger. Every bank account is set to send a Gmail alert for each transaction, so new charges show up early. YouTube feeds Moonshots. Grok bots and Cursor never get an open warehouse login: they connect through MCP servers that only expose semantic views, Cortex Search services and approved procedures.',
  },
  tiles: [
    {
      id: 'workflow',
      choices: [
        { t: 'Merge, apply and deploy are separate gates', d: 'Merging only lands code. Snowflake changes apply from a manual Terraform run, and Airflow changes deploy only from a tagged release pulled onto the PC. A merge alone never reaches production.' },
        { t: 'An explicit go for risky changes', d: 'Unpausing jobs, granting access and dropping objects each need Michael’s go. Eta (Household COO) relays it, and a plan is not production until then.' },
      ],
      n: '2',
      title: 'Ship path: one trunk, a release path per platform',
      svg: 'workflow',
      body:
        'Every change follows the same trunk: a branch, a pull request, the required checks, then a merge that only lands code. After that, each platform has its own way to production. Snowflake infrastructure changes through Terraform. The plan runs on the pull request with a guard that fails on any delete, and the apply is a manual run against production after my explicit go. dbt checks run on DuckDB and in a temporary per-PR schema, and the models run in production from Airflow’s 06:00 ET chain. Airflow cuts a release tag after its tests pass, and a person pulls that tag onto the office PC, which swaps back on its own if anything fails to import. The public website deploys only from a manual dispatch with a required reviewer. Pick a platform chip to see its full flow.',
      caption:
        'The shared trunk sits top right. The selected platform’s flow runs below it, and the dashed cards summarize the other three.',
      rows: [
        {
          name: 'Shared trunk',
          role: 'Branch and pull request, required checks, then a merge that lands code only',
          status: 'Every platform',
        },
        {
          name: 'Terraform · Snowflake',
          role: 'static and plan on the pull request with a 0-destroy guard, nightly drift check, manual apply after Michael’s go',
          status: 'Manual apply',
        },
        {
          name: 'dbt',
          role: 'DuckDB build and a per-PR schema on the pull request, prod manifest on merge, production runs in Airflow',
          status: 'Runs in Airflow',
        },
        {
          name: 'Airflow',
          role: 'DAG and image tests, a release tag on main, then a person pulls the tag onto the office PC',
          status: 'Pulled release',
        },
        {
          name: 'Website',
          role: 'Lint, build and consent tests on the pull request, deploy by dispatch with a required reviewer',
          status: 'Reviewed deploy',
        },
      ],
    },
    {
      id: 'pipes',
      choices: [
        { t: 'Office-PC Airflow instead of S3 and Snowpipe', d: 'Snowpipe was planned after a Drive token failure, then dropped. Documents come in through drive_poll and doc_extract on the office PC. S3 stays a future option.' },
        { t: 'Local OCR instead of Cortex', d: 'doc_extract reads text with pypdf, then Tesseract, so reading a document never spends Cortex credits.' },
        { t: 'One morning load window', d: 'INGEST_WH wakes once at 07:15 ET for every feed and suspends when the window ends, instead of waking for each feed.' },
      ],
      n: '3',
      title: 'Source feeds into Snowflake',
      svg: 'pipes',
      body:
        'Each feed lands in its own product database, not a shared dump. Morning ingest runs once at 07:15 ET on INGEST_WH. Monarch and email load every day. YouTube loads on Mondays and then starts the sync for Moonshots by Peter Diamandis and Market Mondays by Earn Your Leisure. Epic MyChart loads once health data is switched on. Monarch comes in through EAI_MONARCH. Every bank account is set to send a Gmail alert for each transaction; those emails come in through EAI_GMAIL, get classified, and are matched to transactions by reconcile_pending. drive_poll picks up new Google Drive documents at the same minute without waking a warehouse, then doc_extract reads the text with pypdf and local Tesseract instead of Cortex OCR. AI chat history from every platform lands in AGENTIC_WORKFORCE through a daily capture and a nightly chat backup. Epic MyChart SMART on FHIR R4 data is meant for HOUSEHOLD_HEALTH; the model is ready, but no patient data is loaded yet. Every feed follows the same shape: extract and land, transform with dbt, then test. There is no S3 or Snowpipe path; that design was dropped.',
      caption:
        'The morning load window feeds the ledger, plus YouTube on Mondays. Google Drive Docs go through OCR. Chat capture lands in AGENTIC_WORKFORCE. MyChart is modeled in HOUSEHOLD_HEALTH and waiting on source data.',
    },
    {
      id: 'airflow',
      choices: [
        { t: 'Airflow instead of Snowflake tasks', d: 'Scheduling moved off Snowflake tasks for cost and control. Five old CTRL tasks were dropped, and their work now runs as Airflow DAGs.' },
        { t: 'Hourly Drive polling', d: 'drive_poll went from every 15 minutes to hourly to cut schedule cost.' },
        { t: 'One deploy path', d: 'A pinned release tag plus a pull on the PC is the only way code reaches the running Airflow.' },
      ],
      n: '4',
      title: 'Airflow: the production DAGs',
      svg: 'airflow',
      body:
        'Airflow runs every production schedule, and Snowflake tasks are retired. Two clocks drive the day. At 06:00 ET ledger_dbt_daily rebuilds the ledger models, then runs the HR and agentic workforce builds. At 07:15 ET pipeline_load_window wakes INGEST_WH once, calls the Monarch and Gmail loads, and on Mondays starts the YouTube syncs for Moonshots by Peter Diamandis and Market Mondays by Earn Your Leisure. drive_poll checks Google Drive for new documents every hour without waking a warehouse. Code reaches the scheduler only from a tagged release that a person pulls onto the PC. Paused, manual and dev DAGs are left out here.',
      caption:
        'The two clocks that start the day and the production DAGs each one calls.',
      rows: [
        {
          name: 'ledger_dbt_daily',
          role: 'dbt for the ledger on LEDGER_WH, then the HR and agentic workforce builds',
          schedule: '0 6 * * *',
          status: 'Daily 06:00 ET',
        },
        {
          name: 'agentic_workforce_dbt_daily',
          role: 'dbt for AGENTIC_WORKFORCE analytics, at the end of the morning chain',
          schedule: 'triggered',
          status: 'Started by ledger_dbt_daily',
        },
        {
          name: 'pipeline_load_window',
          role: 'One INGEST_WH window for email and Monarch every day, YouTube on Mondays',
          schedule: '15 7 * * *',
          status: 'Daily 07:15 ET',
        },
        {
          name: 'monarch_load',
          role: 'Monarch GraphQL into HOUSEHOLD_LEDGER RAW and CORE through EAI_MONARCH',
          schedule: 'triggered',
          status: 'Called by the window',
        },
        {
          name: 'ledger_email_alerts',
          role: 'Gmail card and bank alerts into RAW and CORE through EAI_GMAIL',
          schedule: 'triggered',
          status: 'Called by the window',
        },
        {
          name: 'moonshots_yt_sync_daily',
          role: 'YouTube catalog sync into MOONSHOTS',
          schedule: 'triggered',
          status: 'Mondays, from the window',
        },
        {
          name: 'moonshots_eyl_sync',
          role: 'Market Mondays captions and companion data',
          schedule: 'triggered',
          status: 'Mondays, from the window',
        },
        {
          name: 'drive_poll',
          role: 'Google Drive poll into document stages, no warehouse',
          schedule: 'hourly',
          status: 'Hourly',
        },
      ],
    },
    {
      id: 'dbt',
      choices: [
        { t: 'RAW and CORE in one pass', d: 'Several procedures write RAW and CORE in the same call, so CORE is not only a later rebuild of RAW.' },
        { t: 'Cosmos on one DAG only', d: 'Astronomer Cosmos turns dbt into one Airflow task per model, but only dbt_daily uses it. The daily ledger chain calls dbt directly.' },
      ],
      n: '5',
      title: 'dbt layers and targets',
      svg: 'dbt',
      body:
        'RAW holds data exactly as it landed. CORE holds clean business objects and is often written in the same step as RAW. dbt builds staging models, marts and ANALYTICS on top of CORE. At 06:00 ET, ledger_dbt_daily runs dbt on LEDGER_WH or COMPUTE_WH, then triggers hr_dbt_daily and agentic_workforce_dbt_daily and waits for both, so everything fits in one LEDGER_WH window. A separate dbt_daily job has its own 06:00 Cosmos schedule targeting hl_prod and ANALYTICS_DBT, but it is created paused and is not part of that chain. Dev copies always target hl_dev and the *_DEV databases and never run on a schedule. How a pull request becomes a Terraform apply or an Airflow release is on the Ship path tile; this tile covers the transform layers.',
      caption:
        'RAW → CORE → dbt staging and marts → ANALYTICS and semantic views. Production targets hl_prod; dev targets hl_dev.',
      rows: [
        {
          name: 'RAW',
          role: 'One landed page, file or API payload, such as Monarch, Gmail MIME or a staged document',
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
          name: 'ANALYTICS + semantic views',
          role: 'Reporting grain and the 15 semantic views Cortex Analyst reads',
          status: 'Serve / MCP catalogs',
        },
        {
          name: 'hl_prod / hl_dev',
          role: 'dbt targets — fixed-prod DAGs refuse DEV_MIGRATION_WH',
          status: 'Environment split',
        },
        {
          name: 'HOUSEHOLD_LEDGER_CI.DBT_PR_<n>',
          role: 'Per-PR dbt CI schema when secrets are set. The Ship path tile covers it',
          status: 'CI only',
        },
      ],
    },
    {
      id: 'domains',
      choices: [
        { t: 'One database per product, not per layer', d: 'Each product area gets its own database, and RAW, CORE and ANALYTICS live as schemas inside it.' },
        { t: 'CONTROL stands on its own', d: 'Shared procedures, queues, flags and tests sit in their own CONTROL database instead of inside any one product.' },
      ],
      n: '6',
      title: 'Databases: columns, tables, modeling',
      svg: 'erd',
      body:
        'Five production databases, each with a *_DEV twin filled with synthetic test data. Open a database below to see its schemas, example tables and grain. HOUSEHOLD_LEDGER has PUBLIC, RAW, CORE, CTRL, ANALYTICS and AI, plus temporary STAGING and INTERMEDIATE schemas. MOONSHOTS has RAW, CORE, CTRL and AI. AGENTIC_WORKFORCE has AGENTS, CONVERSATIONS, GRAPH and CAPTURE, plus STAGING, INTERMEDIATE, ANALYTICS, AI, TRANSFORM and SECURITY. It is where AI chat history from every platform collects, through a daily capture and a nightly chat backup, before agentic_workforce_dbt_daily builds ANALYTICS. CONTROL has PROC, QUEUE, STAGE, FLAGS, TASKS, MCP, SECURITY and TESTS. HOUSEHOLD_HEALTH is the Epic MyChart database; its design and model are in production, but no patient data is loaded yet. In every database, a RAW row is one landed payload, a CORE row is one business object, and ANALYTICS holds reporting tables and semantic views.',
      caption:
        'Select a database for schemas and modeling grain. Prod: HOUSEHOLD_LEDGER, MOONSHOTS, AGENTIC_WORKFORCE, CONTROL, HOUSEHOLD_HEALTH.',
      databases: [
        {
          id: 'ledger',
          name: 'HOUSEHOLD_LEDGER',
          status: 'Production',
          oneLine: 'Finance ledger — Monarch, Gmail, Google Drive Docs',
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
            'YouTube domain for Moonshots by Peter Diamandis and Market Mondays by Earn Your Leisure, shortened to EYL. Monday pipeline_load_window triggers moonshots_yt_sync_daily after the INGEST_WH work; EYL captions write in that window. moonshots_yt_sync_daily and moonshots_eyl_sync were the two DAGs confirmed unpaused at the last check. CTRL hosts MCP_MOONSHOTS for Cynthia (Moonshot and architecture advisory). CS_TRANSCRIPTS and CS_VIDEOS are the Moonshots Cortex Search services. Phase 5 added ANALYTICS views over CORE for reporting reads.',
          objects: [
            { layer: 'Scheduler', name: 'moonshots_yt_sync_daily', note: 'Monday window trigger · unpaused' },
            { layer: 'Scheduler', name: 'moonshots_eyl_sync', note: 'Unscheduled · unpaused' },
            { layer: 'CTRL', name: 'MCP_MOONSHOTS', note: 'YouTube MCP catalog' },
            { layer: 'Search', name: 'CS_TRANSCRIPTS / CS_VIDEOS', note: 'Cortex Search' },
          ],
        },
        {
          id: 'workforce',
          name: 'AGENTIC_WORKFORCE',
          status: 'Production',
          oneLine: 'Cross-platform AI chat history — agents, conversations, graph, capture',
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
            { layer: 'dbt', name: 'agentic_workforce_dbt_daily', note: '→ ANALYTICS · created paused' },
            { layer: 'Planned', name: 'AW desk marts', note: 'Queued, not prod' },
          ],
        },
        {
          id: 'control',
          name: 'CONTROL',
          status: 'Production',
          oneLine: 'Shared control plane — procs, queues, MCP, tests',
          schemas: ['PROC', 'QUEUE', 'STAGE', 'FLAGS', 'TASKS', 'MCP', 'SECURITY', 'TESTS'],
          detail:
            'Cross-product control database. Procedures, queues, stages, feature flags, retired Snowflake tasks inventory, MCP registration, security objects, and test-result tables used by the test-results dashboard. Not a customer domain database — the ops and governance plane for the account.',
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
          oneLine: 'Epic MyChart FHIR — architecture + model built; source data pending',
          schemas: ['RAW', 'STAGING', 'CORE', 'CTRL'],
          detail:
            'Production healthcare database on the same product-database pattern as the ledger. The architecture and Snowflake data model are built. SMART on FHIR R4 from Epic MyChart uses OAuth patient sign-in and a token vault on Lambda, DynamoDB and SSM. Airflow loads onto an internal stage, then procedures write VARIANT RAW. A weekly Monday step inside pipeline_load_window is wired, but no MyChart patient data is loaded yet, so no patient rows are shown.',
          objects: [
            { layer: 'Auth', name: 'OAuth + token vault', note: 'Built' },
            { layer: 'Scheduler', name: 'pipeline_load_window, Mondays', note: 'Wired · waiting on source data' },
            { layer: 'Stage', name: 'Internal stage', note: 'FHIR resources' },
            { layer: 'RAW', name: 'VARIANT tables', note: 'Model ready; no patient data yet' },
          ],
        },
      ],
    },
    {
      id: 'mcp',
      choices: [
        { t: 'Separate read and write servers', d: 'Grok desks get a read-only server. Changes go through a different server that only exposes approved procedures.' },
        { t: 'Writes go through one desk', d: 'Among staff, Eta (Household COO) calls the write tools on MCP_HOUSEHOLD_LEDGER. Other bots may see them, but their instructions forbid using them.' },
        { t: 'YouTube kept apart from money', d: 'Cynthia (Moonshot and architecture advisory) uses MCP_MOONSHOTS, a separate catalog from the household ledger.' },
      ],
      n: '7',
      title: 'MCP servers, agents, and tools',
      svg: 'mcp',
      body:
        'Three MCP servers connect agents to tools. MCP_HOUSEHOLD_LEDGER_GROK is the read server that Grok desks use. It gives household_finance_agent and ledger_ops_agent analyst tools on the 15 semantic views, search over CS_DOCUMENTS, CS_EMAIL_MESSAGES, CS_GAPS_IDEA and CS_PARSE_REVIEW, and document listing and download. MCP_HOUSEHOLD_LEDGER is the write server that Cursor and Eta (Household COO) use. It has the same reads plus approved procedures for Monarch updates, ledger changes, the Gmail and document pipelines, reconciliation, analytics rebuilds, freshness checks and metal prices. MCP_MOONSHOTS is a separate YouTube server for Cynthia (Moonshot and architecture advisory), with video and channel lookups, transcript and video search over CS_TRANSCRIPTS and CS_VIDEOS, sync status, and sync and embedding jobs. The Cortex agents AGENT_HOUSEHOLD_FINANCE, AGENT_HOUSEHOLD_FINANCE_RO and AGENT_LEDGER_OPS sit behind these servers on the same certified data. Clients sign in with a Snowflake access token under the MCP_LEDGER_CLIENT role, and a read client never gets write tools.',
      caption:
        'Servers, then agents, then tool families. Grok uses the read server, Cursor and Eta (Household COO) use the write server, and YouTube work uses the Moonshots server.',
      rows: [
        {
          name: 'MCP_HOUSEHOLD_LEDGER_GROK',
          role: 'Read server for the Grok desks. Agents: household_finance_agent, ledger_ops_agent. Tools: analyst_* on the 15 semantic views, search_* on 4 Cortex Search services, document stage list and download.',
          status: 'Read only',
        },
        {
          name: 'MCP_HOUSEHOLD_LEDGER',
          role: 'Write server — Cursor / Eta (Household COO). Same read families plus proc_monarch_*, proc_ledger_mutate, proc_gmail_ingest, proc_email_pipeline, proc_reconcile_pending, document land/extract/jobs, proc_rebuild_analytics, freshness / QA, proc_metal_spot.',
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
        { t: 'A separate warehouse for agent reads', d: 'Ad-hoc and agent work runs on ADHOC_WH under its own monitor, and Grok service users default to it, so a question never wakes the ledger warehouse.' },
        { t: 'CI results in one insert', d: 'CI test results are written in one INSERT per run instead of one per test.' },
        { t: 'No extra ledger warehouse for now', d: 'The team decided not to split LEDGER_WH into a separate warehouse yet.' },
      ],
      n: '8',
      title: 'Roles, warehouses & cost',
      svg: 'security',
      body:
        'Access and cost are controlled in the same place. Pipelines run under R_PIPELINE and R_PC_PIPELINE. People and agents come in through MCP_LEDGER_CLIENT on the MCP path, never an open warehouse login. Only R_CORTEX_GATE holds SNOWFLAKE.CORTEX_USER, so answers come from certified semantic views and Cortex Search. Warehouses are split by job. LEDGER_WH runs dbt and ledger work and suspends after 30 seconds. INGEST_WH runs the 07:15 ET load window and suspends after 60 seconds. ADHOC_WH handles agent and desk reads, so questions from Grok never wake LEDGER_WH. YT_SYNC_WH runs the Moonshots sync and embeddings. Each warehouse has its own resource monitor. Cost rules live in git: batch work instead of row-by-row inserts, tag every query, account monitors at 40 and 50, and a Cortex AI budget that revokes agent access at 100%. OCR and transcription run locally with Tesseract and Whisper instead of Cortex, Airflow replaces Snowflake tasks, drive_poll runs hourly, and production jobs never use DEV_MIGRATION_WH.',
      caption:
        'Roles, the MCP read and write split, and the Cortex gate on top. The four warehouses, their monitors and cost rules below.',
      rows: [
        {
          name: 'LEDGER_WH',
          role: 'X-Small · auto_suspend 30s · LEDGER_WH_MONITOR · ledger_dbt_daily / COMPUTE_WH path · STD-WH-BATCH',
          status: 'Prod batch',
        },
        {
          name: 'INGEST_WH',
          role: 'auto_suspend 60s · INGEST_MONITOR · pipeline_load_window at 07:15 ET for email and Monarch, plus YouTube on Mondays',
          status: 'Load window',
        },
        {
          name: 'ADHOC_WH',
          role: 'RM_ADHOC · SVC_GROK_* default · agent and desk reads · QUERY_TAG, then suspend',
          status: 'Agents',
        },
        {
          name: 'YT_SYNC_WH',
          role: 'moonshots_yt_sync_daily · PROC_EMBED_TRANSCRIPTS · QUERY_TAG embed vs sync',
          status: 'Moonshots',
        },
        {
          name: 'Caps & savings',
          role: 'BUDGET_CORTEX_AI_20 → PROC_CORTEX_REVOKE_AGENTS · ACCOUNT_MONITOR_* · PC OCR · Airflow not tasks · no DEV_MIGRATION_WH on prod',
          status: 'Controls',
        },
      ],
    },
    {
      id: 'features',
      n: '9',
      title: 'Catalogs and Snowflake features',
      svg: 'features',
      body:
        '15 semantic views define what agents may total: SV_SPENDING, SV_CASHFLOW, SV_BALANCES_NET_WORTH, SV_LIABILITIES_DUES, SV_EARLY_ACTIVITY, SV_BUDGET_VARIANCE, SV_INVESTMENTS, SV_OBLIGATIONS, SV_DOCUMENT_STANDING, SV_OPS_FRESHNESS, SV_TRANSACTIONS, SV_RECURRING_FIXED, SV_DEBT_PAYOFF, SV_ACCOUNTS_RECEIVABLE and SV_SAVINGS_GOALS. Six Cortex Search services handle text: CS_DOCUMENTS, CS_EMAIL_MESSAGES, CS_GAPS_IDEA and CS_PARSE_REVIEW on the ledger, and CS_TRANSCRIPTS and CS_VIDEOS on Moonshots. The Cortex agents are AGENT_HOUSEHOLD_FINANCE, AGENT_HOUSEHOLD_FINANCE_RO and AGENT_LEDGER_OPS. MCP servers expose these as read or write. Warehouse work is batched into one load window, and agent questions run on a separate warehouse.',
      caption:
        '15 semantic views and 6 Cortex Search services. MCP_HOUSEHOLD_LEDGER_GROK and MCP_HOUSEHOLD_LEDGER sit in front of them.',
    },
    {
      id: 'grokbot',
      choices: [
        { t: 'One voice for money', d: 'Stephen (Household CFO) is the only bot that answers money and spend questions. Other desks route those questions to him.' },
        { t: 'Approvals are relayed, not assumed', d: 'Eta (Household COO) speaks for Michael when she relays a go. Without that go, a plan stays a plan.' },
        { t: 'Cortex agents are not Grok bots', d: 'AGENT_HOUSEHOLD_FINANCE, AGENT_HOUSEHOLD_FINANCE_RO and AGENT_LEDGER_OPS are Cortex agents defined in SQL, separate from the Grok bot staff.' },
      ],
      n: '10',
      title: 'Grokbot architecture — agents per MCP',
      svg: 'grokbot',
      body:
        'This shows which agents use each MCP server and what for. Stephen (Household CFO) and the Grok finance desks use the read server, MCP_HOUSEHOLD_LEDGER_GROK. household_finance_agent covers spending, cash flow, balances, documents and ops search. ledger_ops_agent covers freshness and warehouse health. Analyst and search tools return certified totals and find emails and documents. Cursor and Eta (Household COO) use the write server, MCP_HOUSEHOLD_LEDGER, for the same reads plus Monarch and ledger changes, the Gmail and document pipelines, and analytics rebuilds. Cynthia (Moonshot and architecture advisory) uses MCP_MOONSHOTS for the YouTube catalog, transcript search, and sync and embedding jobs. The Cortex agents sit behind these servers on semantic views and Cortex Search services. Even with analyst tools available, money answers on the read path still follow a staff rule.',
      caption:
        'Grok uses the read server, Cursor and Eta (Household COO) the write server, and Cynthia (Moonshot and architecture advisory) the Moonshots server. Each line shows the agent, its server and what that access is for.',
      rows: [
        {
          name: 'Grok finance desks',
          role: '→ MCP_HOUSEHOLD_LEDGER_GROK — household_finance_agent, ledger_ops_agent, analyst_*, search_*, document stage/download',
          status: 'Read',
        },
        {
          name: 'Cursor / Eta (Household COO)',
          role: '→ MCP_HOUSEHOLD_LEDGER — same read families + monarch/ledger mutate, Gmail/docs pipelines, rebuild_analytics',
          status: 'Write',
        },
        {
          name: 'Cynthia (Moonshot and architecture advisory)',
          role: '→ MCP_MOONSHOTS — catalog get/list, transcript/video search, youtube_sync / embed / CS refresh',
          status: 'Domain',
        },
        {
          name: 'Uses of that access',
          role: 'Money Q&A and ops freshness on ledger semantic views / Cortex Search services; document standing; YouTube research on CS_TRANSCRIPTS / CS_VIDEOS',
          status: 'Technical',
        },
      ],
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
        'Customer-facing practice built from the same pattern as HOUSEHOLD_LEDGER: in-account Cortex agents, semantic views, and MCP bridges — not a multi-tenant copy of customer data.',
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
    title: 'Roles, warehouses, and cost controls',
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
        nodes: [{ t: 'certified semantic views' }, { t: 'Cortex Search services' }],
      },
    ],
    warehouses: [
      { t: 'LEDGER_WH', s: '30s · LEDGER_WH_MONITOR' },
      { t: 'INGEST_WH', s: '60s · load window' },
      { t: 'ADHOC_WH', s: 'RM_ADHOC · agents' },
      { t: 'YT_SYNC_WH', s: 'embed QUERY_TAG' },
    ],
    cost: [
      { t: 'STD-WH-BATCH', s: 'batch wake-ups' },
      { t: 'BUDGET_CORTEX_AI_20', s: 'revoke at 100%' },
      { t: 'PC OCR / Whisper', s: 'off Cortex bill' },
      { t: 'Airflow not tasks', s: 'scheduler cost' },
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
          { t: 'Semantic views', s: 'Analyst catalogs' },
          { t: 'Cortex Search', s: 'Search catalogs' },
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
    title: 'Grokbot architecture — agents per MCP',
    note: 'Cortex AGENT_HOUSEHOLD_FINANCE / _RO / AGENT_LEDGER_OPS sit on semantic views and Cortex Search services behind these MCP catalogs.',
    rows: [
      {
        agent: 'Grok finance desks',
        mcp: 'MCP_HOUSEHOLD_LEDGER_GROK',
        use: 'spend · cashflow · docs · ops freshness',
        z: 'agent',
      },
      {
        agent: 'Cursor / Eta (Household COO)',
        mcp: 'MCP_HOUSEHOLD_LEDGER',
        use: 'mutate · ingest · rebuild analytics',
        z: 'agent',
      },
      {
        agent: 'Cynthia (Moonshot advisory)',
        mcp: 'MCP_MOONSHOTS',
        use: 'YouTube catalog · transcripts · sync',
        z: 'src',
      },
    ],
  },
  erd: {
    title:
      'Five prod databases including HOUSEHOLD_HEALTH — select a database in the open panel for its schemas',
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
      { t: 'Branch + pull request', s: 'household-platform · website repo' },
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
        label: 'Website',
        logo: 'github',
        path: 'myorglab-public-website',
        merge: 'builds and tests, no deploy',
        summary: 'lint, build and consent tests on PRs, deploy only by dispatch through a production environment with a required reviewer.',
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
  s: '20 DAG files · 2 confirmed unpaused',
}
DIAG.airflow = {
  title: 'Production Airflow DAGs by trigger',
  counts: [
    { n: '8', label: 'production DAGs' },
    { n: '06:00', label: 'ET dbt chain' },
    { n: '07:15', label: 'ET load window' },
  ],
  rows: [
    ['ledger_dbt_daily → agentic_workforce', '06:00 ET dbt chain'],
    ['pipeline_load_window', '07:15 ET · one INGEST_WH window'],
    ['monarch_load · ledger_email_alerts', 'called by the window'],
    ['moonshots_yt_sync_daily · eyl_sync', 'Mondays, from the window'],
    ['drive_poll', 'hourly · no warehouse'],
  ],
}
DIAG.dbt = {
  title: 'dbt path from RAW/CORE to ANALYTICS and prod apply',
  layers: [
    { t: 'RAW', s: 'landed payloads' },
    { t: 'CORE', s: 'business objects' },
    { t: 'dbt STAGING / marts', s: 'hl_prod · ledger_dbt_daily / dbt_daily' },
    { t: 'ANALYTICS + semantic views', s: '15 Analyst views' },
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
  airflow: '20 DAGs · runs the pulled release tag',
  land: 'pipeline_load_window → monarch + email',
  ocr: 'doc_extract · paused for warehouse quota',
  dbt: 'ledger_dbt_daily → hr → agentic · dbt_daily paused',
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
  cat: ['certified semantic views', 'Cortex Search services'],
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
      { i: 'grid', t: 'Semantic views', s: 'Analyst catalogs' },
      { i: 'search', t: 'Cortex Search', s: 'Search catalogs' },
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

COPY.industries = {
  line: 'The same pattern also covers Healthcare, Logistics, Legal, Finance and Operations, and Public Sector.',
  link: 'See all industries',
  href: 'https://www.myorglab.com/intelligence',
}

DIAG.databases = Object.fromEntries(
  (COPY.tiles.find((t) => t.id === 'domains')?.databases || []).map((db) => [db.id, db]),
)
