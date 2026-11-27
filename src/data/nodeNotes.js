// Short purpose notes for clickable process-flow nodes, one list per diagram. Matched in
// order against the node's visible label text (all <text> lines joined by spaces), so a
// more specific pattern must come before a shorter one it would also match.
const READ_SRV = 'MCP_HOUSEHOLD_LEDGER_GROK. The read server for the Grok desks: analyst totals on the 15 semantic views, search over four Cortex Search services, and document list and download. It holds no write tools.'
const WRITE_SRV = 'MCP_HOUSEHOLD_LEDGER. The write server for Cursor and Eta (Household COO): the same reads plus approved procedures for Monarch and ledger changes, the Gmail and document pipelines, reconciliation and analytics rebuilds.'
const MOON_SRV = 'MCP_MOONSHOTS. A separate YouTube server with catalog lookups, transcript and video search over CS_TRANSCRIPTS and CS_VIDEOS, and sync and embedding jobs. It never touches household money.'

export const NODE_NOTES = {
  lead: [
    [/^Monarch$/, 'Monarch', 'Bank, card and investment accounts. Pulled each morning through EAI_MONARCH into the ledger’s RAW schema.'],
    [/^Gmail$/, 'Gmail', 'Every bank account is set to send a Gmail alert for each transaction. Those emails come in through EAI_GMAIL, get classified, and are matched to transactions by reconcile_pending.'],
    [/^Drive$/, 'Google Drive', 'Statements and documents. drive_poll picks up new files, and doc_extract reads them with pypdf and local Tesseract instead of Cortex OCR.'],
    [/^YouTube$/, 'YouTube', 'Videos and transcripts from Moonshots by Peter Diamandis and Market Mondays by Earn Your Leisure. Every Monday the load window starts the sync into MOONSHOTS.'],
    [/^Airflow/, 'Airflow', 'Runs every production schedule; Snowflake tasks are retired. Starts the 07:15 ET load window on INGEST_WH, then dbt and tests.'],
    [/^Landing schema/, 'RAW', 'Landing schema. One row per landed payload, kept as received.'],
    [/^Conformed schema/, 'CORE', 'Conformed schema. One row per business object, such as an account, transaction or document. Approved writes land here.'],
    [/^Analytics schema/, 'ANALYTICS', 'Reporting grain built by dbt. Feeds the semantic views.'],
    [/^Semantic views/, 'Semantic views', '15 certified semantic views. Agents answer numeric questions from these, not from raw tables.'],
    [/^Cortex Search/, 'Cortex Search services', '6 Cortex Search services over documents, emails and transcripts for text retrieval.'],
    [/Read MCP server/, 'Read MCP server', READ_SRV],
    [/Write MCP server/, 'Write MCP server', WRITE_SRV],
    [/^Read clients$/, 'Read clients', 'Grokbot desks, plus me asking questions. They connect only to the read MCP server, never an open warehouse login.'],
    [/^Write client$/, 'Write client', 'Grokbot and me working through Cursor and Eta (Household COO). Every change goes through an approved procedure on the write MCP server into CORE.'],
  ],
  workflow: [
    [/^(1|Branch \+ PR)\b/, 'Branch and pull request', 'Every change starts on a branch of household-platform and opens a pull request. Nothing is pushed straight to main.'],
    [/^(2|PR CI)\b/, 'Pull request CI', 'GitHub Actions runs ruff and pytest, gitleaks secret scanning, static checks and a Terraform plan. dbt runs in a temporary HOUSEHOLD_LEDGER_CI.DBT_PR_<n> schema when the CI secrets are set.'],
    [/^(3|Merge)\b/, 'Merge', 'Merging lands code on main and nothing else. It does not apply Terraform or reach the running Airflow.'],
    [/^(4|Terraform apply)\b/, 'Terraform apply', 'Databases, schemas, roles, warehouses and monitors change only when I run the apply workflow by hand against production, after an explicit go.'],
    [/^(5|Airflow tag)\b/, 'Airflow release tag', 'An annotated tag pins the Airflow code the PC should run. A newer tag in git is not live until the PC pulls it.'],
    [/^(6|PC pull)\b/, 'Pull on the PC', 'deploy_airflow.ps1 pulls the tagged release onto the Docker worker. Unpausing jobs, grants and drops still each need their own go.'],
  ],
  pipes: [
    [/^▶$/, 'Morning start', 'The 07:15 ET load window on INGEST_WH starts the day’s ingest. Drive polling fires at the same minute without waking a warehouse.'],
    [/^Source feeds/, 'Source feeds', 'Monarch through EAI_MONARCH, Gmail transaction alerts through EAI_GMAIL, Drive documents through drive_poll, and YouTube into MOONSHOTS. Epic MyChart is modeled for HOUSEHOLD_HEALTH and waiting on source data.'],
    [/^◇$/, 'Route by source', 'Monarch and Gmail go to the load step. Drive documents go to OCR first.'],
    [/^Extract and land/, 'Extract and land', 'The load window pulls Monarch data and Gmail transaction alerts into RAW on INGEST_WH, then the warehouse suspends.'],
    [/^doc_extract/, 'doc_extract', 'Reads Drive documents with pypdf, then local Tesseract, and lands the text for CORE. It is paused while the warehouse quota is tight.'],
    [/^RAW and CORE/, 'RAW and CORE', 'RAW keeps payloads as they landed. CORE turns them into clean business objects, often in the same procedure.'],
    [/^dbt/, 'dbt and procedures', 'ledger_dbt_daily builds CORE into ANALYTICS at 06:00 ET, then starts the HR and agentic workforce builds. The separate dbt_daily job stays paused.'],
    [/^Tests$/, 'Tests', 'dbt and Airflow test results land in one append-only test-results table.'],
    [/^●$/, 'Served', 'ANALYTICS feeds the semantic views and Cortex Search services that the MCP servers expose to agents.'],
  ],
  airflow: [
    [/^17 /, '17 production-shaped DAGs', 'DAGs written for production targets. Being production-shaped does not mean a DAG is unpaused.'],
    [/^3 /, '3 dev DAGs', 'ledger_dbt_daily_dev, dbt_daily_dev and monarch_load_dev. Isolated twins on the *_DEV databases that never run on a schedule.'],
    [/^2 /, '2 confirmed unpaused', 'moonshots_yt_sync_daily and moonshots_eyl_sync were confirmed unpaused at the last check. Every other DAG is paused or not yet checked.'],
    [/^pipeline_load_window/, 'pipeline_load_window', 'The 07:15 ET window on INGEST_WH. Loads email and Monarch daily, starts the YouTube sync on Mondays, and runs Epic once health data is enabled.'],
    [/^moonshots_yt/, 'YouTube syncs', 'moonshots_yt_sync_daily refreshes the MOONSHOTS catalog when the Monday window triggers it. moonshots_eyl_sync handles Market Mondays captions.'],
    [/^monarch_load/, 'Monarch and email loads', 'monarch_load pulls Monarch GraphQL through EAI_MONARCH, and the ledger_email jobs land Gmail alerts through EAI_GMAIL. The window calls both.'],
    [/^drive_poll/, 'Drive and OCR', 'drive_poll checks Google Drive hourly for new documents. doc_extract reads them on the PC and lands the text in CORE.'],
    [/^ledger_dbt_daily/, 'Morning dbt chain', 'At 06:00 ET ledger_dbt_daily runs dbt for the ledger, then starts hr_dbt_daily and agentic_workforce_dbt_daily and waits for both.'],
    [/^dbt_daily/, 'dbt_daily', 'An Astronomer Cosmos build targeting hl_prod and ANALYTICS_DBT. It was created paused and is not part of the morning chain.'],
    [/^nightly_app_backup/, 'PC routines', 'A nightly bundle backup and weekday PC usage checks. Neither touches Snowflake.'],
    [/^epic_fhir_load_manual/, 'Epic MyChart load', 'A manual FHIR load into HOUSEHOLD_HEALTH behind EPIC_FHIR_LOAD_ENABLED. The model is ready, but no patient data is loaded yet.'],
    [/^platform_smoke/, 'Smoke and dev twins', 'Manual health checks and the three dev twins. None of them run in production.'],
  ],
  dbt: [
    [/^RAW\b/, 'RAW', 'One landed page, file or API payload, kept exactly as it arrived.'],
    [/^CORE\b/, 'CORE', 'One business object per row. Several procedures write it in the same call as RAW.'],
    [/^dbt STAGING/, 'Staging and marts', 'dbt builds staging models and marts on top of CORE. Production targets hl_prod; dev targets hl_dev and the *_DEV databases.'],
    [/^ANALYTICS/, 'ANALYTICS and semantic views', 'Reporting grain plus the 15 semantic views that Cortex Analyst and the MCP analyst tools read.'],
    [/^CI → merge/, 'From pull request to production', 'dbt checks run in a per-PR schema, merging lands code only, and production changes come from a Terraform apply and an Airflow release pull. The Ship path tile has the steps.'],
  ],
  mcp: [
    [/^MCP_HOUSEHOLD_LEDGER_GROK/, 'Read server', READ_SRV],
    [/^MCP_HOUSEHOLD_LEDGER\b/, 'Write server', WRITE_SRV],
    [/^MCP_MOONSHOTS/, 'Moonshots server', MOON_SRV],
    [/^household_finance_agent/, 'Read agents', 'household_finance_agent covers spending, cash flow, balances, documents and ops search. ledger_ops_agent covers freshness and warehouse health.'],
    [/^same agents/, 'Write-side agents', 'The same MCP agents, plus write procedures. The Cortex agents AGENT_HOUSEHOLD_FINANCE, AGENT_HOUSEHOLD_FINANCE_RO and AGENT_LEDGER_OPS sit behind them on certified data.'],
    [/^Cynthia/, 'Cynthia (Moonshot and architecture advisory)', 'Uses MCP_MOONSHOTS for the YouTube catalog, transcript search, and sync and embedding jobs.'],
    [/^agents · analyst/, 'Read tool families', 'analyst_* totals on semantic views, search_* over Cortex Search services, and document stage list and download.'],
    [/^read families/, 'Write tool families', 'Every read family, plus procedures for Monarch and ledger changes, Gmail and document pipelines, reconciliation, analytics rebuilds, freshness checks and metal prices.'],
    [/^catalog · search/, 'Moonshots tool families', 'Catalog get and list, transcript and video search, sync status, and sync, embedding and search refresh jobs.'],
  ],
  security: [
    [/^R_PIPELINE/, 'Pipeline roles', 'R_PIPELINE and R_PC_PIPELINE run the loads and dbt. People and agents never use them.'],
    [/^Interactive roles/, 'Interactive roles', 'Roles for people working directly in Snowflake. Agents do not get these.'],
    [/^MCP_LEDGER_CLIENT/, 'MCP_LEDGER_CLIENT', 'The role MCP clients sign in under with a Snowflake access token. It reaches the MCP servers, not open tables.'],
    [/^MCP_HOUSEHOLD_LEDGER_GROK/, 'Read server', READ_SRV],
    [/^MCP_HOUSEHOLD_LEDGER\b/, 'Write server', WRITE_SRV],
    [/^R_CORTEX_GATE/, 'R_CORTEX_GATE', 'The only role holding SNOWFLAKE.CORTEX_USER, so Cortex answers come only from certified semantic views and Cortex Search.'],
    [/^certified semantic views/, 'Certified semantic views', 'The 15 semantic views agents may total from, such as SV_DOCUMENT_STANDING.'],
    [/^Cortex Search/, 'Cortex Search services', 'Text retrieval over documents, emails and transcripts, such as CS_DOCUMENTS.'],
    [/^LEDGER_WH/, 'LEDGER_WH', 'X-Small warehouse for dbt and ledger work. Suspends after 30 seconds and has its own LEDGER_WH_MONITOR.'],
    [/^INGEST_WH/, 'INGEST_WH', 'Runs the 07:15 ET load window and suspends after 60 seconds, under INGEST_MONITOR.'],
    [/^ADHOC_WH/, 'ADHOC_WH', 'Agent and desk reads under RM_ADHOC. Grok service users default here, so a question never wakes LEDGER_WH.'],
    [/^YT_SYNC_WH/, 'YT_SYNC_WH', 'Runs the Moonshots sync and transcript embeddings, with query tags that split embed cost from sync cost.'],
    [/^STD-WH-BATCH/, 'Batch standard', 'Wake a warehouse once for a batch of work, skip row-by-row inserts, and tag every query.'],
    [/^BUDGET_CORTEX_AI_20/, 'Cortex AI budget', 'A Cortex AI budget that revokes agent access at 100%.'],
    [/^PC OCR/, 'Local OCR and transcription', 'Tesseract and Whisper run on the PC, so reading documents and audio stays off the Cortex bill.'],
    [/^Airflow not tasks/, 'Airflow instead of tasks', 'Schedules run in Airflow instead of Snowflake tasks, which keeps scheduling off warehouse time.'],
  ],
  features: [
    [/^ACCESS$/, 'Access', 'Two MCP servers sit in front of every catalog: one read-only, one with approved write procedures.'],
    [/^MCP_HOUSEHOLD_LEDGER_GROK/, 'Read server', READ_SRV],
    [/^MCP_HOUSEHOLD_LEDGER\b/, 'Write server', WRITE_SRV],
    [/^CATALOGS$/, 'Catalogs', 'What agents may read: semantic views for numbers and Cortex Search for text.'],
    [/^Semantic views/, 'Semantic views', '15 semantic views such as SV_SPENDING, SV_CASHFLOW and SV_DOCUMENT_STANDING define what agents may total.'],
    [/^Cortex Search/, 'Cortex Search services', 'Six services: CS_DOCUMENTS, CS_EMAIL_MESSAGES, CS_GAPS_IDEA and CS_PARSE_REVIEW on the ledger, CS_TRANSCRIPTS and CS_VIDEOS on Moonshots.'],
    [/^DATA$/, 'Data', 'Layers live as schemas inside each product database instead of separate databases.'],
    [/^RAW \/ CORE/, 'Schemas', 'RAW keeps landed payloads, CORE holds business objects, CTRL hosts the MCP servers, and ANALYTICS holds reporting grain.'],
    [/^WAREHOUSES$/, 'Warehouses', 'Warehouses are split by job so batch loads and agent questions never share compute.'],
    [/^LEDGER_WH \/ INGEST_WH/, 'Batch warehouses', 'LEDGER_WH runs dbt and INGEST_WH runs the one morning load window. Both suspend within a minute.'],
    [/^Interactive \/ agents/, 'Agent warehouse', 'ADHOC_WH handles agent and desk reads on its own monitor.'],
  ],
  grokbot: [
    [/^Grok finance desks/, 'Grok finance desks', 'Stephen (Household CFO) and the finance desks. Stephen is the only bot that answers money and spend questions.'],
    [/^MCP_HOUSEHOLD_LEDGER_GROK/, 'Read server', READ_SRV],
    [/^MCP_HOUSEHOLD_LEDGER\b/, 'Write server', WRITE_SRV],
    [/^MCP_MOONSHOTS/, 'Moonshots server', MOON_SRV],
    [/^spend/, 'Read uses', 'Certified spend and cash flow totals, document standing, and ops freshness checks.'],
    [/^Cursor \/ Eta/, 'Cursor and Eta (Household COO)', 'Eta relays Michael’s go and is the one staff desk that calls the write tools.'],
    [/^mutate/, 'Write uses', 'Monarch and ledger changes, Gmail and document ingest, and analytics rebuilds, each through an approved procedure.'],
    [/^Cynthia/, 'Cynthia (Moonshot and architecture advisory)', 'Researches Moonshots and Market Mondays content through MCP_MOONSHOTS.'],
    [/^YouTube catalog/, 'Moonshots uses', 'Catalog lookups, transcript search, and sync and embedding jobs.'],
    [/^Cortex AGENT/, 'Cortex agents', 'AGENT_HOUSEHOLD_FINANCE, AGENT_HOUSEHOLD_FINANCE_RO and AGENT_LEDGER_OPS are defined in SQL and are separate from the Grok bot staff.'],
  ],
}

const SCHEMA_NOTES = {
  RAW: 'Landing schema. One row per landed payload, kept as received.',
  STAGING: 'Temporary dbt staging models between RAW and CORE.',
  INTERMEDIATE: 'Temporary dbt intermediate models.',
  CORE: 'Conformed schema. One row per business object. Approved writes land here.',
  ANALYTICS: 'Reporting grain built by dbt. Semantic views and dashboards read from here.',
  AI: 'Cortex objects for this database.',
  CTRL: 'Database controls, including the MCP servers and freshness views.',
  AGENTS: 'Captured AI agents from every platform I use.',
  CONVERSATIONS: 'Captured AI chat history.',
  GRAPH: 'Graph tables that link agents and conversations.',
  CAPTURE: 'Bookkeeping for the daily capture routine.',
  TRANSFORM: 'Transform helpers for the workforce models.',
  SECURITY: 'Security objects for this database.',
  PROC: 'Shared pipeline procedures.',
  QUEUE: 'Work queues that feed ingest jobs.',
  STAGE: 'Internal stages for landed files.',
  FLAGS: 'Feature flags that jobs read at run time.',
  TASKS: 'Inventory of retired Snowflake tasks.',
  MCP: 'MCP server registration.',
  TESTS: 'Test results that the test-results dashboard reads.',
}

export function noteFor(label, kind = 'lead') {
  const text = String(label || '').replace(/\s+/g, ' ').trim()
  const hit = (NODE_NOTES[kind] || []).find(([re]) => re.test(text))
  return hit ? { title: hit[1], text: hit[2] } : null
}

export function tableNote(schema, name) {
  const text = SCHEMA_NOTES[schema]
  return text && name ? { title: `${schema}.${name}`, text } : null
}
