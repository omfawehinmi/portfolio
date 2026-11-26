// Portfolio Snowflake section rewritten from the as-built in
// myorglab-repos/household-data-architecture (synced to household-platform
// fdff221 via HDA PR #12 — 20 DAG pages). Real databases, MCP servers, catalogs,
// EAIs, and DAG names. No secrets, PATs, or account locators. Status language
// matches the as-built: production vs planned vs paused.

export const COPY = {
  section: {
    kicker: 'Projects',
    title: 'Work that shows the platform',
    intro:
      'The lead project is the household Snowflake platform: 4 production databases (HOUSEHOLD_LEDGER, MOONSHOTS, AGENTIC_WORKFORCE, CONTROL), 3 MCP servers, 20 Airflow DAG files, and a PR → CI → merge → Terraform apply → Airflow pin ship path. Tiles under the lead walk CI/CD with source feeds, schemas and modeling, Airflow + dbt, MCP and roles, catalogs, then Grokbot agents per MCP. Other shipped work follows in a shorter list.',
  },
  lead: {
    title: 'HOUSEHOLD_LEDGER and MCP',
    subhead: 'Snowflake data architecture and MCP',
    oneLine:
      '4 prod databases · 3 MCP servers · 20 Airflow DAG files · Monarch, Gmail, Drive, YouTube (plus planned Epic MyChart) · AI chat history in AGENTIC_WORKFORCE · 15 SV_* · 6 CS_*. Agents hit MCP catalogs, not an open warehouse login.',
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
    intro:
      'This is the household data platform I build and run. Snowflake holds HOUSEHOLD_LEDGER (finance), MOONSHOTS (YouTube), AGENTIC_WORKFORCE (agent roster, conversations, graph, and chat-history capture across desks), and CONTROL (procs, queues, MCP, tests) — layers as schemas, not one database per layer. HOUSEHOLD_HEALTH is the planned Epic MyChart database, not a separate product card. Airflow on the office PC is the production scheduler; Snowflake tasks are retired. Monarch, Gmail, and Drive feed the ledger; YouTube feeds Moonshots; workforce capture and nightly chat backup land AI history. Grok bots and Cursor attach to MCP servers bound to SV_*, CS_*, and approved procedures.',
  },
  tiles: [
    {
      id: 'workflow',
      n: '2',
      title: 'Ship path: PR → prod',
      svg: 'workflow',
      body:
        'Change reaches Snowflake and Airflow in separate gates — merge never applies and never deploys. Branch work opens a PR. Required CI in the migration chapters: ruff + pytest, gitleaks, static, and Terraform plan. dbt PR CI can use HOUSEHOLD_LEDGER_CI.DBT_PR_<n> when secrets exist; otherwise it skips. Merge lands code on main only. Snowflake infra (databases, schemas, roles, warehouses, monitors) applies from a manual GitHub Actions dispatch with apply=true against the production environment. Production environment required reviewer is not enforced in those chapters — Michael’s go for apply is the team rule. Airflow reaches the office PC only after a tagged release and a manual PC pull (deploy_airflow.ps1). Last recorded PC apply in household-tracker is airflow-20261113-d64a1c8; git tags continue through airflow-20261123-b6a02d7 on household-platform fdff221 — a tag is not a live pin. PLATFORM_ENV and PLATFORM_STUB on the pin decide prod vs stub. Unpause, grants, and drops also need an explicit go.',
      caption:
        'PR CI → merge (code only) → Terraform apply (Snowflake) → Airflow tag + PC pull. Merge ≠ apply ≠ deploy.',
      rows: [
        {
          name: '1. Branch + PR',
          role: 'Feature work opens a pull request on household-platform',
          status: 'Author',
        },
        {
          name: '2. PR CI',
          role: 'ruff + pytest, gitleaks, static, Terraform plan; optional dbt in HOUSEHOLD_LEDGER_CI.DBT_PR_<n>',
          status: 'GitHub Actions',
        },
        {
          name: '3. Merge to main',
          role: 'Code only — merge does not apply Terraform and does not pull Airflow',
          status: 'Gate',
        },
        {
          name: '4. Terraform apply',
          role: 'Manual workflow_dispatch apply=true → environment production (Michael’s go)',
          status: 'Snowflake infra',
        },
        {
          name: '5. Airflow release tag',
          role: 'Annotated pin (e.g. airflow-20261113-d64a1c8); later tags through 20261123 exist in git',
          status: 'Scheduler artifact',
        },
        {
          name: '6. Office-PC pull',
          role: 'Manual deploy_airflow.ps1 on the Docker worker — git tag ≠ live until pulled',
          status: 'Prod scheduler',
        },
        {
          name: '7. Unpause / grants',
          role: 'Separate go for unpause, grants, and drops (doc 17 approval gates)',
          status: 'Ops',
        },
      ],
    },
    {
      id: 'pipes',
      n: '3',
      title: 'Source feeds into Snowflake',
      svg: 'pipes',
      body:
        'Live feeds land in product database schemas — not a shared dump. Morning ingest is one pipeline_load_window at 07:15 ET on INGEST_WH (email + Monarch daily; YouTube Monday, then trigger moonshots_yt_sync_daily; Epic FHIR only when EPIC_FHIR_LOAD_ENABLED). Monarch via EAI_MONARCH (monarch_load / proc_monarch_extract); Gmail via EAI_GMAIL (ledger_email_alerts → classify → reconcile_pending); Drive via drive_poll (same 07:15 minute, no warehouse) → doc_extract (pypdf then local Tesseract — no Cortex OCR). Workforce AI history lands through capture MERGE and the nightly chat backup into AGENTIC_WORKFORCE (and RAW.CHAT_MESSAGES). Planned feed: Epic MyChart SMART on FHIR R4 into HOUSEHOLD_HEALTH (in git, not recorded as applied). Shape is extract/land → transform (dbt) → test. No S3/Snowpipe path — that design was dropped.',
      caption:
        'pipeline_load_window → ledger + Monday YouTube. Drive → OCR. Chat capture → AGENTIC_WORKFORCE. MyChart FHIR → planned HOUSEHOLD_HEALTH.',
    },
    {
      id: 'airflow',
      n: '4',
      title: 'Airflow on the office PC — every DAG',
      svg: 'airflow',
      body:
        '20 DAG files in household-platform Airflow (household-platform fdff221 / HDA PR #12 — one docs page per DAG). Code on main is not the running pin: last office-PC apply recorded in household-tracker is airflow-20261113-d64a1c8 (2026-11-13); later git tags exist through airflow-20261123-b6a02d7. A file in git is not an unpaused job: only moonshots_yt_sync_daily and moonshots_eyl_sync were confirmed unpaused on the 2026-11-07 pin check (not re-read since). Open the tile for the full inventory — schedule, responsibility, and prod status for each DAG.',
      caption:
        '20 DAG files · 17 prod-shaped · 3 *_dev · 2 confirmed unpaused on the 2026-11-07 pin. Last PC apply: airflow-20261113-d64a1c8. Pause unknown ≠ running.',
      rows: [
        {
          name: 'pipeline_load_window',
          role: '07:15 ET INGEST_WH window — email + Monarch daily; YouTube Mondays; optional Epic',
          schedule: '15 7 * * *',
          status: 'Prod-shaped ingest window. Created paused',
        },
        {
          name: 'ledger_dbt_daily',
          role: 'Bash dbt for ledger (LEDGER_WH / COMPUTE_WH); triggers HR then agentic workforce',
          schedule: '0 6 * * *',
          status: 'Prod-shaped (fixed prod). Pause unknown',
        },
        {
          name: 'hr_dbt_daily',
          role: 'HR domain dbt blueprint — started by ledger_dbt_daily',
          schedule: 'none',
          status: 'No cron — HR account still a blueprint',
        },
        {
          name: 'agentic_workforce_dbt_daily',
          role: 'dbt for AGENTIC_WORKFORCE analytics — after HR in the morning chain',
          schedule: 'none',
          status: 'No cron. Started after HR. Pause unknown',
        },
        {
          name: 'dbt_daily',
          role: 'Astronomer Cosmos dbt, target hl_prod / ANALYTICS_DBT — not in the morning chain',
          schedule: '0 6 * * *',
          status: 'Created paused. Not the morning chain',
        },
        {
          name: 'drive_poll',
          role: 'Google Drive poll → document stages (same minute as load window)',
          schedule: '15 7 * * *',
          status: 'Created paused. No warehouse',
        },
        {
          name: 'doc_extract',
          role: 'PC OCR (pypdf then Tesseract) → CORE document facts',
          schedule: 'none, triggered',
          status: 'Do not mark unpaused — quota pause in change log',
        },
        {
          name: 'monarch_load',
          role: 'Monarch GraphQL → HOUSEHOLD_LEDGER RAW/CORE via EAI_MONARCH (called by window)',
          schedule: 'none',
          status: 'Prod-shaped (fixed prod / INGEST_WH). Created paused',
        },
        {
          name: 'ledger_email_alerts',
          role: 'Gmail card/bank alerts → RAW/CORE via EAI_GMAIL (called by window)',
          schedule: 'none',
          status: 'Called by window. Created paused',
        },
        {
          name: 'ledger_email_pull_poll',
          role: 'Metadata count for email pull requests — window loads the requests',
          schedule: 'none',
          status: 'Metadata only. Pause unknown',
        },
        {
          name: 'moonshots_yt_sync_daily',
          role: 'YouTube catalog sync into MOONSHOTS (triggered by Monday window)',
          schedule: 'none',
          status: 'Triggered Mondays · unpaused on 2026-11-07 pin',
        },
        {
          name: 'moonshots_eyl_sync',
          role: 'EYL captions / companion — Monday work is the load window',
          schedule: 'none',
          status: 'Unscheduled · unpaused on 2026-11-07 pin',
        },
        {
          name: 'moonshots_transcript_backfill',
          role: 'Transcript backfill (Airflow port of Windows job)',
          schedule: 'none',
          status: 'Not running — stayed paused on 2026-11-07 check',
        },
        {
          name: 'nightly_app_backup',
          role: 'Office-PC nightly bundle push (no Snowflake)',
          schedule: '33 19 * * *',
          status: 'Created paused. No Snowflake',
        },
        {
          name: 'pc_usage_check',
          role: 'Weekday PC usage / My Machines snapshot thresholds (no Snowflake)',
          schedule: '17 9,13,17 * * 1-5',
          status: 'Created paused. No Snowflake',
        },
        {
          name: 'epic_fhir_load_manual',
          role: 'Manual Epic FHIR → HOUSEHOLD_HEALTH (needs EPIC_FHIR_LOAD_ENABLED)',
          schedule: 'none',
          status: 'Manual only — not a production load; DB not recorded applied',
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
      title: 'dbt layers and targets',
      svg: 'dbt',
      body:
        'RAW lands payloads. CORE holds business objects (often written in the same procedure pass as RAW). dbt builds STAGING / marts / ANALYTICS on top of CORE: at 06:00 ET ledger_dbt_daily runs Bash dbt on LEDGER_WH or COMPUTE_WH, then triggers hr_dbt_daily, then agentic_workforce_dbt_daily, and waits — one LEDGER_WH window. dbt_daily keeps its own 06:00 Cosmos cron (target hl_prod / ANALYTICS_DBT) and is created paused — not in that chain. Dev twins force hl_dev / *_DEV and have no schedule. How a PR becomes a Terraform apply or an Airflow pin lives on the Ship path tile — this tile is the transform contract.',
      caption:
        'RAW → CORE → dbt STAGING/marts → ANALYTICS + SV_*. hl_prod vs hl_dev. PR CI schema is on the Ship path tile.',
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
          role: 'Per-PR dbt CI schema when secrets exist (see Ship path)',
          status: 'CI only',
        },
      ],
    },
    {
      id: 'domains',
      n: '6',
      title: 'Databases: columns, tables, modeling',
      svg: 'erd',
      body:
        'Four production databases plus *_DEV twins with synthetic fixtures. Open a database below for schemas, example tables, and grain. HOUSEHOLD_LEDGER: PUBLIC, RAW, CORE, CTRL, ANALYTICS, AI, transient STAGING / INTERMEDIATE. MOONSHOTS: RAW, CORE, CTRL, AI. AGENTIC_WORKFORCE: AGENTS, CONVERSATIONS, GRAPH, CAPTURE, plus STAGING, INTERMEDIATE, ANALYTICS, AI, TRANSFORM, SECURITY — this is where desk AI history accumulates (capture MERGE + nightly RAW.CHAT_MESSAGES backup), then agentic_workforce_dbt_daily builds ANALYTICS. CONTROL: PROC, QUEUE, STAGE, FLAGS, TASKS, MCP, SECURITY, TESTS. HOUSEHOLD_HEALTH is planned (Epic MyChart FHIR), not production — same product-database idea, not a separate portfolio card. A RAW row is one landed payload; a CORE row is one business object; ANALYTICS is reporting grain and SV_* catalogs.',
      caption:
        'Select a database for schemas and modeling grain. Prod: HOUSEHOLD_LEDGER, MOONSHOTS, AGENTIC_WORKFORCE, CONTROL. Planned: HOUSEHOLD_HEALTH.',
      databases: [
        {
          id: 'ledger',
          name: 'HOUSEHOLD_LEDGER',
          status: 'Production',
          oneLine: 'Finance ledger — Monarch, Gmail, Drive documents',
          schemas: ['PUBLIC', 'RAW', 'CORE', 'CTRL', 'ANALYTICS', 'AI', 'STAGING', 'INTERMEDIATE'],
          detail:
            'Landing and conformed finance data. CTRL hosts MCP_HOUSEHOLD_LEDGER_GROK and MCP_HOUSEHOLD_LEDGER. ANALYTICS holds reporting grain and the 15 SV_* Analyst views; Cortex Search CS_DOCUMENTS, CS_EMAIL_MESSAGES, CS_GAPS_IDEA, CS_PARSE_REVIEW sit on ledger text. Example path: DOCUMENTS_RAW → DOCUMENTS → FACT_DOCUMENT_STANDING → SV_DOCUMENT_STANDING / CS_DOCUMENTS.',
          objects: [
            { layer: 'RAW', name: 'DOCUMENTS_RAW', note: 'One landed file / payload' },
            { layer: 'CORE', name: 'DOCUMENTS', note: 'One business document' },
            { layer: 'ANALYTICS', name: 'FACT_DOCUMENT_STANDING', note: 'Reporting grain' },
            { layer: 'Catalog', name: 'SV_* / CS_*', note: 'Analyst + Search' },
          ],
        },
        {
          id: 'moonshots',
          name: 'MOONSHOTS',
          status: 'Production',
          oneLine: 'YouTube catalog — sync DAGs + MCP_MOONSHOTS',
          schemas: ['RAW', 'CORE', 'CTRL', 'AI'],
          detail:
            'YouTube / EYL domain. Monday pipeline_load_window triggers moonshots_yt_sync_daily after the INGEST_WH work; EYL captions write in that window. moonshots_yt_sync_daily and moonshots_eyl_sync were the two confirmed-unpaused DAGs on the 2026-11-07 pin (not re-read). CTRL hosts MCP_MOONSHOTS for Cynthia. CS_TRANSCRIPTS and CS_VIDEOS are the Moonshots Cortex Search services. Phase 5 added ANALYTICS views over CORE for reporting reads.',
          objects: [
            { layer: 'Scheduler', name: 'moonshots_yt_sync_daily', note: 'Monday window trigger · unpaused on pin' },
            { layer: 'Scheduler', name: 'moonshots_eyl_sync', note: 'Unscheduled · unpaused on pin' },
            { layer: 'CTRL', name: 'MCP_MOONSHOTS', note: 'YouTube MCP catalog' },
            { layer: 'Search', name: 'CS_TRANSCRIPTS / CS_VIDEOS', note: 'Cortex Search' },
          ],
        },
        {
          id: 'workforce',
          name: 'AGENTIC_WORKFORCE',
          status: 'Production',
          oneLine: 'AI desk history — agents, conversations, graph, capture',
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
            'Where multi-platform AI history accumulates. A daily capture routine MERGEs agents, conversations, graph, and capture tables. A nightly chat backup MERGEs RAW.CHAT_MESSAGES (SVC_BOX_PIPELINE / R_PIPELINE). agentic_workforce_dbt_daily builds tag:agentic_workforce into AGENTIC_WORKFORCE.ANALYTICS (dev twin AGENTIC_WORKFORCE_DEV); no cron of its own — started after hr_dbt_daily in the 06:00 ledger_dbt_daily chain. Desk marts for the Desks app (AW-MART-INFRA) remain queued — not treated as shipped.',
          objects: [
            { layer: 'Capture', name: 'AGENTS / CONVERSATIONS / GRAPH', note: 'Daily MERGE' },
            { layer: 'Landing', name: 'RAW.CHAT_MESSAGES', note: 'Nightly chat backup' },
            { layer: 'dbt', name: 'agentic_workforce_dbt_daily', note: '→ ANALYTICS (paused on create)' },
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
          status: 'Planned — not production',
          oneLine: 'Epic MyChart FHIR — sandbox auth exists; prod load not built',
          schemas: ['RAW', 'STAGING', 'CORE', 'CTRL'],
          detail:
            'Planned healthcare database, same product-DB pattern as the ledger — not a separate “healthcare project” card. SMART on FHIR R4 from Epic MyChart: OAuth patient auth, token vault on Lambda / DynamoDB / SSM, designed Airflow load onto an internal stage, then procedures into VARIANT RAW. Weekly Monday step inside pipeline_load_window was decided; first production sign-in has not succeeded. No patient data is shown.',
          objects: [
            { layer: 'Auth', name: 'OAuth + token vault', note: 'Sandbox tested' },
            { layer: 'Scheduler', name: 'pipeline_load_window (Mon)', note: 'Decided, not built' },
            { layer: 'Stage', name: 'Internal stage', note: 'FHIR resources' },
            { layer: 'RAW', name: 'VARIANT tables', note: 'Not in production' },
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
        'Three MCP servers bind agents and tools — the household read/write split is only the first cut. MCP_HOUSEHOLD_LEDGER_GROK (HOUSEHOLD_LEDGER.CTRL) is the read catalog Grok desks attach to: household_finance_agent, ledger_ops_agent, analyst_* tools on the 15 SV_* views, search over CS_DOCUMENTS / CS_EMAIL_MESSAGES / CS_GAPS_IDEA / CS_PARSE_REVIEW, plus document stage list and download. MCP_HOUSEHOLD_LEDGER is the write catalog Cursor / Eta use: the same read surface plus mutate and pipeline procs (Monarch extract/mutate/holdings, ledger_mutate, Gmail ingest and email pipeline, reconcile_pending, document land/extract/jobs, rebuild_analytics, freshness / QA, metal spot). MCP_MOONSHOTS on MOONSHOTS.CTRL is a separate YouTube catalog for Cynthia: list/get video and channel, transcript and video search (ILIKE, semantic, CS_TRANSCRIPTS / CS_VIDEOS), sync status, and procs for youtube_sync, embed_transcripts, and CS refresh. Cortex agents AGENT_HOUSEHOLD_FINANCE, AGENT_HOUSEHOLD_FINANCE_RO, and AGENT_LEDGER_OPS sit on the certified catalogs behind those MCP surfaces. Auth is a Snowflake PAT under role MCP_LEDGER_CLIENT. A read client never holds write tools. Summarized families only — not a private tool dump.',
      caption:
        'Servers → agents → tool families. Read MCP for Grok; write MCP for Cursor/Eta; Moonshots MCP for YouTube. No full catalog dump.',
      rows: [
        {
          name: 'MCP_HOUSEHOLD_LEDGER_GROK',
          role: 'Read server — Grok desks. Agents: household_finance_agent, ledger_ops_agent. Tools: analyst_* (15 SV_*), search_* (4 CS_*), document stage list / download.',
          status: 'Read only',
        },
        {
          name: 'MCP_HOUSEHOLD_LEDGER',
          role: 'Write server — Cursor / Eta. Same read families plus proc_monarch_*, proc_ledger_mutate, proc_gmail_ingest, proc_email_pipeline, proc_reconcile_pending, document land/extract/jobs, proc_rebuild_analytics, freshness / QA, proc_metal_spot.',
          status: 'Read + write',
        },
        {
          name: 'MCP_MOONSHOTS',
          role: 'YouTube server — Cynthia. Catalog get/list, transcript/video search (ILIKE + semantic + CS_*), sync status, proc_youtube_sync / embed / CS refresh. Not household money.',
          status: 'Domain MCP',
        },
        {
          name: 'Agent layer',
          role: 'MCP agents household_finance_agent + ledger_ops_agent; Cortex agents AGENT_HOUSEHOLD_FINANCE, AGENT_HOUSEHOLD_FINANCE_RO, AGENT_LEDGER_OPS on SV_* / CS_*.',
          status: 'Orchestration',
        },
        {
          name: 'Tool families (ledger)',
          role: 'analyst_* · search_* · document_* · monarch_* · ledger_mutate · gmail/email pipeline · reconcile · rebuild_analytics · freshness/QA · metal_spot',
          status: 'Summarized',
        },
        {
          name: 'Tool families (Moonshots)',
          role: 'get/list catalog · search transcripts/videos · CS_TRANSCRIPTS / CS_VIDEOS · youtube_sync · embed_transcripts · refresh CS',
          status: 'Summarized',
        },
      ],
    },
    {
      id: 'security',
      n: '8',
      title: 'Roles and Cortex gate',
      svg: 'security',
      body:
        'Access is split by role and by which MCP server a client is attached to. Pipeline work uses R_PIPELINE / R_PC_PIPELINE; interactive and agent clients use MCP_LEDGER_CLIENT against the MCP path, not a shared open warehouse login. R_CORTEX_GATE is the sole holder of SNOWFLAKE.CORTEX_USER. Answers are supposed to come from certified SV_* views and CS_* search services. Public copy stops at that architecture: no account locators, hostnames, or PATs.',
      caption:
        'Role separation plus MCP_HOUSEHOLD_LEDGER vs MCP_HOUSEHOLD_LEDGER_GROK, with Cortex gated behind R_CORTEX_GATE.',
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
    {
      id: 'grokbot',
      n: '10',
      title: 'Grokbot architecture — agents per MCP',
      svg: 'grokbot',
      body:
        'Only the agent paths that attach to each MCP server, plus a few technical uses of that surface — not a broader platform tour. Stephen / Grok finance desks use MCP_HOUSEHOLD_LEDGER_GROK (read): household_finance_agent for spend, cashflow, balances, documents, and ops search; ledger_ops_agent for freshness and warehouse health; analyst_* and search_* tools for certified totals and mailbox/document retrieval. Cursor / Eta use MCP_HOUSEHOLD_LEDGER (write) for the same reads plus Monarch and ledger mutations, Gmail/document pipelines, and analytics rebuild. Cynthia uses MCP_MOONSHOTS for YouTube catalog, transcript search, and sync/embed jobs. Cortex agents AGENT_HOUSEHOLD_FINANCE / _RO and AGENT_LEDGER_OPS sit on SV_* and CS_* behind those MCP catalogs. Money answers stay a staff rule on the read path even when analyst tools are listed.',
      caption:
        'Grok → read MCP. Cursor/Eta → write MCP. Cynthia → Moonshots MCP. Each line is agent + MCP + what that access is for.',
      rows: [
        {
          name: 'Grok finance desks',
          role: '→ MCP_HOUSEHOLD_LEDGER_GROK — household_finance_agent, ledger_ops_agent, analyst_*, search_*, document stage/download',
          status: 'Read',
        },
        {
          name: 'Cursor / Eta',
          role: '→ MCP_HOUSEHOLD_LEDGER — same read families + monarch/ledger mutate, Gmail/docs pipelines, rebuild_analytics',
          status: 'Write',
        },
        {
          name: 'Cynthia (YouTube)',
          role: '→ MCP_MOONSHOTS — catalog get/list, transcript/video search, youtube_sync / embed / CS refresh',
          status: 'Domain',
        },
        {
          name: 'Uses of that access',
          role: 'Money Q&A and ops freshness on ledger SV_*/CS_*; document standing; YouTube research on CS_TRANSCRIPTS / CS_VIDEOS',
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
  mcpTools: {
    title: 'MCP servers, agents, and tool families',
    note: 'Read server never holds write tools. Families summarized — not a private catalog dump.',
    servers: [
      {
        t: 'MCP_HOUSEHOLD_LEDGER_GROK',
        s: 'Read · Grok desks',
        families: 'agents · analyst_* · search_* · docs',
        z: 'agent',
      },
      {
        t: 'MCP_HOUSEHOLD_LEDGER',
        s: 'Write · Cursor / Eta',
        families: 'read families + mutate / pipeline procs',
        z: 'agent',
      },
      {
        t: 'MCP_MOONSHOTS',
        s: 'YouTube · Cynthia',
        families: 'catalog · search · sync / embed',
        z: 'src',
      },
    ],
    agents: [
      ['household_finance_agent', 'ledger_ops_agent'],
      ['same agents + write procs', 'Cortex AGENT_* behind'],
      ['Cynthia desk', 'YouTube catalog tools'],
    ],
  },
  grokbot: {
    title: 'Grokbot architecture — agents per MCP',
    note: 'Cortex AGENT_HOUSEHOLD_FINANCE / _RO / AGENT_LEDGER_OPS sit on SV_* and CS_* behind these MCP catalogs.',
    rows: [
      {
        agent: 'Grok finance desks',
        mcp: 'MCP_HOUSEHOLD_LEDGER_GROK',
        use: 'spend · cashflow · docs · ops freshness',
        z: 'agent',
      },
      {
        agent: 'Cursor / Eta',
        mcp: 'MCP_HOUSEHOLD_LEDGER',
        use: 'mutate · ingest · rebuild analytics',
        z: 'agent',
      },
      {
        agent: 'Cynthia',
        mcp: 'MCP_MOONSHOTS',
        use: 'YouTube catalog · transcripts · sync',
        z: 'src',
      },
    ],
  },
  erd: {
    title:
      'Four prod databases plus planned HOUSEHOLD_HEALTH — select a database in the open panel for its schemas',
    areas: [
      'HOUSEHOLD_LEDGER',
      'MOONSHOTS',
      'AGENTIC_WORKFORCE',
      'CONTROL',
      'HOUSEHOLD_HEALTH (planned)',
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
        meaning: 'Search catalog (serving may be suspended; indexing active)',
      },
    ],
  },
  workflow: {
    title: 'Ship path from PR to production Snowflake and Airflow',
    steps: [
      { t: 'Branch + PR', s: 'household-platform' },
      { t: 'PR CI', s: 'ruff · gitleaks · static · plan' },
      { t: 'Merge', s: 'code only' },
      { t: 'Terraform apply', s: 'apply=true · production' },
      { t: 'Airflow tag', s: 'release pin' },
      { t: 'PC pull', s: 'deploy_airflow.ps1' },
    ],
  },
}

DIAG.lead.airflow = {
  t: 'Airflow on office PC',
  s: '20 DAG files · 2 confirmed unpaused on 2026-11-07 pin',
}
DIAG.airflow = {
  title: 'Airflow DAG inventory on the office-PC pin',
  headline: '20 files in git (fdff221)',
  counts: [
    { n: '17', label: 'prod-shaped' },
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
  airflow: '20 DAGs · last PC apply airflow-20261113-d64a1c8',
  land: 'pipeline_load_window → monarch + email',
  ocr: 'doc_extract (paused for WH quota)',
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

COPY.industries = {
  line: 'The same pattern also covers Healthcare, Logistics, Legal, Finance and Operations, and Public Sector.',
  link: 'See all industries',
  href: 'https://www.myorglab.com/intelligence',
}

DIAG.databases = Object.fromEntries(
  (COPY.tiles.find((t) => t.id === 'domains')?.databases || []).map((db) => [db.id, db]),
)
