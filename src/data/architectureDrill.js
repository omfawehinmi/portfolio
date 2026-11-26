// Drill catalog synced from myorglab-repos/household-data-architecture
// (architecture, pipelines, services-and-components, cost-assessment).
// Metadata and object names only — no credentials, PATs, account locators,
// or row/column values. Dollar spend stays blocked (not shown).

/** @typedef {{ id: string, label: string, blurb: string, diagram: string, detail: string, steps?: {t:string,s?:string}[], meta?: {k:string,v:string}[] }} DrillItem */

/** High-level entry areas shown on the map layer. */
export const DRILL_AREAS = [
  {
    id: 'feeds',
    label: 'Source feeds',
    mapHint: 'Monarch · Gmail · Drive · YouTube · MyChart',
    defaultItem: 'monarch',
  },
  {
    id: 'databases',
    label: 'Databases',
    mapHint: '5 prod DBs + *_DEV twins',
    defaultItem: 'ledger',
  },
  {
    id: 'warehouses',
    label: 'Warehouses & cost',
    mapHint: 'LEDGER_WH · INGEST_WH · ADHOC_WH · caps',
    defaultItem: 'ledger_wh',
  },
  {
    id: 'tools',
    label: 'MCP & tools',
    mapHint: '3 servers · agents · tool families',
    defaultItem: 'read_mcp',
  },
  {
    id: 'workflows',
    label: 'Workflows',
    mapHint: 'Ship path · ingest window · dbt chain',
    defaultItem: 'ship',
  },
]

/** @type {Record<string, DrillItem[]>} */
export const DRILL_ITEMS = {
  feeds: [
    {
      id: 'monarch',
      label: 'Monarch → HOUSEHOLD_LEDGER',
      blurb: 'GraphQL EAI extract into RAW/CORE on the morning INGEST_WH window.',
      diagram: 'feed',
      detail:
        'EAI_MONARCH + PROC_MONARCH_EXTRACT / monarch_load. MERGE into HOUSEHOLD_LEDGER RAW and CORE in one procedure pass where the map says so — CORE is not always a pure dbt transform of RAW. Called by pipeline_load_window (07:15 ET). Prod-shaped; created paused on many pins. Fixed-prod warehouse selection (never DEV_MIGRATION_WH). Community Monarch MCP is a client fallback only — warehouse extract is system of record.',
      steps: [
        { t: 'Monarch GraphQL', s: 'EAI_MONARCH' },
        { t: 'PROC_MONARCH_EXTRACT', s: 'Snowpark proc' },
        { t: 'RAW + CORE MERGE', s: 'HOUSEHOLD_LEDGER' },
        { t: 'dbt / ANALYTICS', s: 'ledger_dbt_daily chain' },
        { t: 'SV_* / MCP read', s: 'certified catalogs' },
      ],
      meta: [
        { k: 'DAG', v: 'monarch_load · called by pipeline_load_window' },
        { k: 'Warehouse', v: 'INGEST_WH (window) / LEDGER_WH (dbt)' },
        { k: 'EAI', v: 'EAI_MONARCH' },
        { k: 'Target DB', v: 'HOUSEHOLD_LEDGER' },
      ],
    },
    {
      id: 'gmail',
      label: 'Gmail → early card alerts',
      blurb: 'EAI_GMAIL ingest → classify → reconcile_pending into ledger RAW/CORE.',
      diagram: 'feed',
      detail:
        'ledger_email_alerts (and pull_poll) call TASK_GMAIL_INGEST through SP_RUN_EMAIL_TASK. Rules classify first; PROC_EMAIL_ALERT_CLASSIFY / AI_CLASSIFY handles unclear mail (Cortex inside Snowflake, not Tesseract). Lands early activity before Monarch posts. Schedules in code: 15 7,12,17 * * * for alerts; */5 for pull_poll. Pause state often unknown on the pin.',
      steps: [
        { t: 'Gmail', s: 'EAI_GMAIL' },
        { t: 'SP_RUN_EMAIL_TASK', s: 'ingest' },
        { t: 'Rules + AI_CLASSIFY', s: 'unclear mail' },
        { t: 'RAW / CORE', s: 'early activity' },
        { t: 'reconcile_pending', s: 'write MCP procs' },
      ],
      meta: [
        { k: 'DAGs', v: 'ledger_email_alerts · ledger_email_pull_poll' },
        { k: 'EAI', v: 'EAI_GMAIL' },
        { k: 'Cortex', v: 'AI_CLASSIFY on unclear mail only' },
        { k: 'Target', v: 'HOUSEHOLD_LEDGER early-activity grain' },
      ],
    },
    {
      id: 'drive',
      label: 'Drive → OCR → documents',
      blurb: 'drive_poll (no WH) → doc_extract on PC (pypdf then Tesseract) → SP_PC_DOC_LOAD.',
      diagram: 'feed',
      detail:
        'drive_poll uses the Drive Changes API with no Snowflake connection (hourly after schedule was slowed from every 15 min for cost). doc_extract downloads bytes, runs pypdf then local Tesseract/pdf2image — not Cortex AI_PARSE_DOCUMENT — then CTRL.SP_PC_DOC_LOAD. Often paused for warehouse quota on pins. Cost choice: PC OCR vs Cortex parse (architecture decisions).',
      steps: [
        { t: 'Google Drive API', s: 'drive_poll · no SF WH' },
        { t: 'Download bytes', s: 'doc_extract' },
        { t: 'pypdf → Tesseract', s: 'PC OCR' },
        { t: 'SP_PC_DOC_LOAD', s: 'CTRL proc' },
        { t: 'DOCUMENTS_RAW → CORE', s: 'HOUSEHOLD_LEDGER' },
      ],
      meta: [
        { k: 'DAGs', v: 'drive_poll · doc_extract' },
        { k: 'OCR', v: 'Tesseract on office PC (not Cortex)' },
        { k: 'Warehouse', v: 'Load via LEDGER/CTRL procs; poll itself uses none' },
        { k: 'Cost note', v: 'Hourly poll; OCR off Cortex bill' },
      ],
    },
    {
      id: 'youtube',
      label: 'YouTube → MOONSHOTS',
      blurb: 'Monday window triggers moonshots_yt_sync_daily; EYL captions; optional Whisper on PC.',
      diagram: 'feed',
      detail:
        'pipeline_load_window Mondays trigger moonshots_yt_sync_daily after INGEST_WH work. Official YouTube Data API; caption scraping off; audio fallback only if YT_AUDIO_FALLBACK_ENABLED. faster-whisper (small, int8, CPU) on PC for audio path — not Cortex. Embeddings via PROC_EMBED_TRANSCRIPTS on YT_SYNC_WH with query tag moonshots_yt_sync:embed when MOONSHOTS_EMBED_ENABLED. CS_TRANSCRIPTS / CS_VIDEOS are Moonshots search services. moonshots_yt_sync_daily and moonshots_eyl_sync were the two confirmed-unpaused DAGs on the 2026-11-07 pin.',
      steps: [
        { t: 'YouTube Data API', s: 'PC DAG' },
        { t: 'Captions / Whisper', s: 'PC · optional' },
        { t: 'MOONSHOTS RAW/CORE', s: 'YT_SYNC_WH' },
        { t: 'PROC_EMBED_TRANSCRIPTS', s: 'query-tagged' },
        { t: 'CS_TRANSCRIPTS / CS_VIDEOS', s: 'MCP_MOONSHOTS' },
      ],
      meta: [
        { k: 'DAGs', v: 'moonshots_yt_sync_daily · moonshots_eyl_sync' },
        { k: 'Warehouse', v: 'YT_SYNC_WH' },
        { k: 'EAI stub', v: 'EAI_MOONSHOTS_YOUTUBE (proc stub; PC DAGs call YT directly)' },
        { k: 'Pin check', v: 'Both YT DAGs unpaused on 2026-11-07' },
      ],
    },
    {
      id: 'mychart',
      label: 'Epic MyChart → HOUSEHOLD_HEALTH',
      blurb: 'SMART on FHIR R4 path built; underlying patient data not loaded yet.',
      diagram: 'feed',
      detail:
        'HOUSEHOLD_HEALTH architecture and Snowflake model are production. OAuth patient auth + token vault (Lambda / DynamoDB / SSM), Airflow epic_fhir_load_manual / Monday window step when EPIC_FHIR_LOAD_ENABLED, internal stage → VARIANT RAW. No patient row values shown. Same product-DB pattern as the ledger — not a separate portfolio card.',
      steps: [
        { t: 'Epic MyChart', s: 'SMART on FHIR R4' },
        { t: 'OAuth + token vault', s: 'built' },
        { t: 'Airflow load', s: 'EPIC_FHIR_LOAD_ENABLED' },
        { t: 'Internal stage', s: 'FHIR resources' },
        { t: 'HOUSEHOLD_HEALTH RAW', s: 'model ready · data pending' },
      ],
      meta: [
        { k: 'DB', v: 'HOUSEHOLD_HEALTH (Production)' },
        { k: 'Schemas', v: 'RAW · STAGING · CORE · CTRL' },
        { k: 'DAG', v: 'epic_fhir_load_manual / window Monday step' },
        { k: 'Privacy', v: 'No patient data values on portfolio' },
      ],
    },
    {
      id: 'workforce_capture',
      label: 'AI desk capture → AGENTIC_WORKFORCE',
      blurb: 'Daily MERGE of agents/conversations/graph + nightly chat backup.',
      diagram: 'feed',
      detail:
        'Daily capture MERGEs AGENTS, CONVERSATIONS, GRAPH, CAPTURE. Nightly chat backup MERGEs RAW.CHAT_MESSAGES (SVC_BOX_PIPELINE / R_PIPELINE). agentic_workforce_dbt_daily builds tag:agentic_workforce into ANALYTICS after hr_dbt_daily in the morning chain — no cron of its own. Desk marts (AW-MART-INFRA) remain queued.',
      steps: [
        { t: 'Desk platforms', s: 'capture routine' },
        { t: 'MERGE AGENTS/CONV/GRAPH', s: 'daily' },
        { t: 'RAW.CHAT_MESSAGES', s: 'nightly backup' },
        { t: 'agentic_workforce_dbt_daily', s: '→ ANALYTICS' },
        { t: 'Desks app reads', s: 'marts queued' },
      ],
      meta: [
        { k: 'DB', v: 'AGENTIC_WORKFORCE' },
        { k: 'dbt', v: 'After HR in ledger_dbt_daily chain' },
        { k: 'Non-SF jobs', v: 'nightly_app_backup · pc_usage_check (no Snowflake)' },
      ],
    },
  ],

  databases: [
    {
      id: 'ledger',
      label: 'HOUSEHOLD_LEDGER',
      blurb: 'Finance domain — Monarch, Gmail, Drive; CTRL MCP; 15 SV_* + ledger CS_*.',
      diagram: 'database',
      detail:
        'Schemas: PUBLIC, RAW, CORE, CTRL, ANALYTICS, AI, transient STAGING/INTERMEDIATE. Example path DOCUMENTS_RAW → DOCUMENTS → FACT_DOCUMENT_STANDING → SV_DOCUMENT_STANDING / CS_DOCUMENTS. CTRL hosts MCP_HOUSEHOLD_LEDGER_GROK and MCP_HOUSEHOLD_LEDGER. Layers are schemas inside the product DB, not one database per layer.',
      meta: [
        { k: 'Status', v: 'Production' },
        { k: 'Feeds', v: 'Monarch · Gmail · Drive' },
        { k: 'Catalogs', v: '15 SV_* · CS_DOCUMENTS · CS_EMAIL_MESSAGES · CS_GAPS_IDEA · CS_PARSE_REVIEW' },
      ],
    },
    {
      id: 'moonshots',
      label: 'MOONSHOTS',
      blurb: 'YouTube catalog — RAW/CORE/CTRL/AI; MCP_MOONSHOTS; CS_TRANSCRIPTS / CS_VIDEOS.',
      diagram: 'database',
      detail:
        'Monday ingest trigger + unpaused sync DAGs on the 2026-11-07 pin. Phase 5 ANALYTICS views over CORE for reporting. CTRL hosts MCP_MOONSHOTS for Cynthia.',
      meta: [
        { k: 'Status', v: 'Production' },
        { k: 'Schemas', v: 'RAW · CORE · CTRL · AI (+ ANALYTICS views)' },
        { k: 'Search', v: 'CS_TRANSCRIPTS · CS_VIDEOS' },
      ],
    },
    {
      id: 'workforce',
      label: 'AGENTIC_WORKFORCE',
      blurb: 'AI desk history — agents, conversations, graph, capture, analytics.',
      diagram: 'database',
      detail:
        'Schemas include AGENTS, CONVERSATIONS, GRAPH, CAPTURE, STAGING, INTERMEDIATE, ANALYTICS, AI, TRANSFORM, SECURITY. Capture + nightly chat backup land history; dbt builds ANALYTICS. Dev twin AGENTIC_WORKFORCE_DEV for fixtures.',
      meta: [
        { k: 'Status', v: 'Production' },
        { k: 'dbt', v: 'agentic_workforce_dbt_daily (after HR)' },
      ],
    },
    {
      id: 'control',
      label: 'CONTROL',
      blurb: 'Shared control plane — procs, queues, stages, flags, MCP, tests.',
      diagram: 'database',
      detail:
        'PROC, QUEUE, STAGE, FLAGS, TASKS, MCP, SECURITY, TESTS. Cross-product ops and governance — not a customer domain database. Test-results dashboard reads TESTS.',
      meta: [
        { k: 'Status', v: 'Production' },
        { k: 'Owns', v: 'Pipeline procs · MCP registration · feature flags' },
      ],
    },
    {
      id: 'health',
      label: 'HOUSEHOLD_HEALTH',
      blurb: 'Epic MyChart FHIR — architecture + model built; source data pending.',
      diagram: 'database',
      detail:
        'Production healthcare product DB (RAW, STAGING, CORE, CTRL). SMART on FHIR R4 path wired; underlying patient data not loaded. No patient values on the portfolio.',
      meta: [
        { k: 'Status', v: 'Production (data pending)' },
        { k: 'Source', v: 'Epic MyChart SMART on FHIR R4' },
      ],
    },
  ],

  warehouses: [
    {
      id: 'ledger_wh',
      label: 'LEDGER_WH',
      blurb: 'Primary ledger/dbt warehouse — X-Small, auto_suspend 30s, monitor-capped.',
      diagram: 'warehouse',
      detail:
        'Used by ledger_dbt_daily / COMPUTE_WH path and related ledger work. auto_suspend=30 in warehouses.tf. LEDGER_WH_MONITOR credit quota recorded in change log (temporary 10 then back toward 30; notify 75%, suspend 100%/110%). From 11/2–11/8 the batching standard cites 384 wake-ups with most cost from the per-wake minimum — hence STD-WH-BATCH: batch by default, no row-by-row inserts, QUERY_TAG required, ad-hoc on ADHOC_WH.',
      steps: [
        { t: 'Wake once per run', s: 'STD-WH-BATCH' },
        { t: 'Multi-row INSERT/COPY/MERGE', s: 'no row-by-row' },
        { t: 'QUERY_TAG', s: 'desk or job' },
        { t: 'auto_suspend 30s', s: 'Terraform' },
        { t: 'LEDGER_WH_MONITOR', s: 'cap + notify' },
      ],
      meta: [
        { k: 'Size', v: 'X-Small (as recorded)' },
        { k: 'auto_suspend', v: '30 seconds' },
        { k: 'Monitor', v: 'LEDGER_WH_MONITOR' },
        { k: 'Standard', v: 'STD-WH-BATCH / CD-002' },
      ],
    },
    {
      id: 'ingest_wh',
      label: 'INGEST_WH',
      blurb: 'Morning load window warehouse — auto_suspend 60s; INGEST_MONITOR.',
      diagram: 'warehouse',
      detail:
        'pipeline_load_window at 07:15 ET runs email + Monarch (daily) and YouTube Mondays on INGEST_WH. auto_suspend 60s. INGEST_MONITOR referenced on the warehouse; BUDGET_INGEST_* + COST_CENTER=INGEST tags on ledger Cortex Search services for spend guardrails (config in git — not live spend figures).',
      steps: [
        { t: '07:15 ET window', s: 'pipeline_load_window' },
        { t: 'Monarch + email', s: 'daily' },
        { t: 'YouTube trigger', s: 'Mondays' },
        { t: 'auto_suspend 60s', s: 'Terraform' },
        { t: 'INGEST cost tags', s: 'CS_* COST_CENTER=INGEST' },
      ],
      meta: [
        { k: 'auto_suspend', v: '60 seconds' },
        { k: 'Monitor', v: 'INGEST_MONITOR' },
        { k: 'Jobs', v: 'Fixed-prod ingest path' },
      ],
    },
    {
      id: 'adhoc_wh',
      label: 'ADHOC_WH + agents',
      blurb: 'Ad-hoc and agent reads — RM_ADHOC; SVC_GROK_* default here.',
      diagram: 'warehouse',
      detail:
        'CD-003: ADHOC_WH with RM_ADHOC for ad-hoc and agent work (applied live; Terraform import pending). CD-005: SVC_GROK_* service users default to ADHOC_WH so agents do not wake LEDGER_WH for Q&A. Desk practice: QUERY_TAG then suspend after. Production loads stay on their own warehouses.',
      steps: [
        { t: 'Agent / ad-hoc query', s: 'QUERY_TAG' },
        { t: 'ADHOC_WH', s: 'not LEDGER_WH' },
        { t: 'RM_ADHOC', s: 'monitor' },
        { t: 'Suspend after', s: 'desk practice' },
      ],
      meta: [
        { k: 'Decisions', v: 'CD-003 · CD-005' },
        { k: 'Users', v: 'SVC_GROK_* default warehouse' },
        { k: 'Rule', v: 'Prod loads never select DEV_MIGRATION_WH' },
      ],
    },
    {
      id: 'yt_sync_wh',
      label: 'YT_SYNC_WH',
      blurb: 'Moonshots sync/embed warehouse — query-tagged embed vs sync.',
      diagram: 'warehouse',
      detail:
        'YouTube sync and PROC_EMBED_TRANSCRIPTS. Query tag moonshots_yt_sync:embed splits embed cost from the rest of the sync for attribution. Embeddings only when MOONSHOTS_EMBED_ENABLED.',
      meta: [
        { k: 'Used by', v: 'moonshots_yt_sync_daily' },
        { k: 'Cost split', v: 'QUERY_TAG embed vs sync' },
      ],
    },
    {
      id: 'cost_controls',
      label: 'Caps & savings mechanisms',
      blurb: 'Monitors, Cortex budget guard, Airflow vs tasks, OCR on PC, batching.',
      diagram: 'warehouse',
      detail:
        'Controls in git (not live spend): LEDGER_WH_MONITOR, ACCOUNT_MONITOR_40/50, BUDGET_CORTEX_AI_20 → PROC_CORTEX_REVOKE_AGENTS at 100%, ingest budget object, RM_DEV_MIGRATION (dev only). Savings: Airflow instead of Snowflake tasks; less frequent schedules (drive_poll hourly); Tesseract on PC; query tags; warehouse auto-suspend; prod jobs avoid DEV_MIGRATION_WH; CI JUnit batched one INSERT per run (CD-004); ROUTINE-PRUNE moves recurring LLM routines into existing Airflow windows (CD-008). No dollar amounts on the portfolio.',
      steps: [
        { t: 'Resource monitors', s: 'notify + suspend' },
        { t: 'Cortex spend guard', s: 'BUDGET_CORTEX_AI_20' },
        { t: 'Batch wake-ups', s: 'STD-WH-BATCH' },
        { t: 'PC OCR / Whisper', s: 'off Cortex bill' },
        { t: 'Airflow not tasks', s: 'scheduler cost' },
      ],
      meta: [
        { k: 'Source', v: 'HDA docs/cost-assessment' },
        { k: 'Spend figures', v: 'Blocked — not shown' },
        { k: 'Decisions', v: 'CD-001 … CD-008' },
      ],
    },
  ],

  tools: [
    {
      id: 'read_mcp',
      label: 'MCP_HOUSEHOLD_LEDGER_GROK',
      blurb: 'Read server — Grok desks; analyst_* + search_*; no write tools.',
      diagram: 'tool',
      detail:
        'Bound in HOUSEHOLD_LEDGER.CTRL. Agents: household_finance_agent, ledger_ops_agent. Tools: analyst_* on 15 SV_*, search_* on CS_DOCUMENTS / CS_EMAIL_MESSAGES / CS_GAPS_IDEA / CS_PARSE_REVIEW, document stage list/download. Auth: Snowflake PAT under MCP_LEDGER_CLIENT. Read client never holds write tools.',
      meta: [
        { k: 'Mode', v: 'Read only' },
        { k: 'Agents', v: 'household_finance_agent · ledger_ops_agent' },
        { k: 'Behind', v: 'AGENT_HOUSEHOLD_FINANCE / _RO · AGENT_LEDGER_OPS' },
      ],
    },
    {
      id: 'write_mcp',
      label: 'MCP_HOUSEHOLD_LEDGER',
      blurb: 'Write server — same reads plus mutate / pipeline procs for Cursor/Eta.',
      diagram: 'tool',
      detail:
        'Read surface plus Monarch extract/mutate/holdings, ledger_mutate, Gmail ingest and email pipeline, reconcile_pending, document land/extract/jobs, rebuild_analytics, freshness/QA, metal spot. Separate server so read clients never receive write tools.',
      meta: [
        { k: 'Mode', v: 'Read + approved writes' },
        { k: 'Clients', v: 'Cursor / Eta' },
      ],
    },
    {
      id: 'moonshots_mcp',
      label: 'MCP_MOONSHOTS',
      blurb: 'YouTube catalog MCP for Cynthia — list/search/sync/embed procs.',
      diagram: 'tool',
      detail:
        'On MOONSHOTS.CTRL. Catalog list/get, transcript and video search (ILIKE, semantic, CS_*), sync status, youtube_sync / embed_transcripts / CS refresh procs.',
      meta: [
        { k: 'DB', v: 'MOONSHOTS.CTRL' },
        { k: 'Desk', v: 'Cynthia' },
      ],
    },
  ],

  workflows: [
    {
      id: 'ship',
      label: 'PR → prod ship path',
      blurb: 'CI → merge (code only) → Terraform apply → Airflow tag → PC pull.',
      diagram: 'workflow',
      detail:
        'Merge never applies Terraform and never deploys Airflow. apply=true is manual against environment production. Airflow reaches the office PC only after a tagged release and deploy_airflow.ps1. Last recorded PC apply airflow-20261113-d64a1c8; later tags exist — a tag is not a live pin.',
      steps: [
        { t: 'Branch + PR', s: 'Author' },
        { t: 'PR CI', s: 'ruff · gitleaks · TF plan' },
        { t: 'Merge', s: 'code only' },
        { t: 'Terraform apply', s: 'manual · production' },
        { t: 'Airflow tag + PC pull', s: 'deploy_airflow.ps1' },
      ],
      meta: [{ k: 'Rule', v: 'Merge ≠ apply ≠ deploy' }],
    },
    {
      id: 'ingest_window',
      label: 'pipeline_load_window',
      blurb: '07:15 ET INGEST_WH — email + Monarch daily; YouTube Mondays; optional Epic.',
      diagram: 'workflow',
      detail:
        'Single morning window batches warehouse wake-ups (STD-WH-BATCH). Triggers moonshots_yt_sync_daily after INGEST work on Mondays. Epic FHIR when enabled and source data present.',
      steps: [
        { t: 'Wake INGEST_WH', s: 'once' },
        { t: 'Email + Monarch', s: 'daily' },
        { t: 'YouTube trigger', s: 'Monday' },
        { t: 'Optional Epic', s: 'flag + data' },
        { t: 'Suspend', s: 'auto_suspend 60s' },
      ],
      meta: [
        { k: 'Schedule', v: '15 7 * * *' },
        { k: 'Warehouse', v: 'INGEST_WH' },
      ],
    },
    {
      id: 'dbt_chain',
      label: 'Morning dbt chain',
      blurb: 'ledger_dbt_daily → hr_dbt_daily → agentic_workforce_dbt_daily on LEDGER_WH.',
      diagram: 'workflow',
      detail:
        '06:00 ET Bash dbt for ledger, then triggers HR then agentic and waits — one LEDGER_WH window. dbt_daily (Cosmos, hl_prod) keeps its own 06:00 cron and is created paused — not in that chain. Dev twins force hl_dev / *_DEV with no schedule.',
      steps: [
        { t: 'ledger_dbt_daily', s: '06:00 · LEDGER_WH' },
        { t: 'hr_dbt_daily', s: 'triggered' },
        { t: 'agentic_workforce_dbt_daily', s: 'triggered' },
        { t: 'ANALYTICS + SV_*', s: 'tested models' },
      ],
      meta: [
        { k: 'Fixed prod', v: 'ledger_dbt_daily never DEV_MIGRATION_WH' },
        { k: 'Parallel', v: 'dbt_daily Cosmos paused on create' },
      ],
    },
  ],
}

export function drillItem(areaId, itemId) {
  const list = DRILL_ITEMS[areaId] || []
  return list.find((item) => item.id === itemId) || list[0] || null
}

export function drillArea(areaId) {
  return DRILL_AREAS.find((area) => area.id === areaId) || DRILL_AREAS[0]
}
