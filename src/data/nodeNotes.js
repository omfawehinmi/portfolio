// Short purpose notes for clickable process-flow nodes. Matched in order against
// the node's visible label text (all <text> lines joined by spaces).
export const NODE_NOTES = [
  [/^Monarch$/, 'Monarch', 'Bank, card and investment data. Pulled through EAI_MONARCH (monarch_load / proc_monarch_extract) into HOUSEHOLD_LEDGER RAW each morning.'],
  [/^Gmail$/, 'Gmail', 'Account alert emails. Pulled through EAI_GMAIL, classified, then matched to transactions by reconcile_pending.'],
  [/^Drive$/, 'Google Drive', 'Statements and documents. drive_poll lands new files; doc_extract reads them with pypdf, then local Tesseract (no Cortex OCR).'],
  [/^YouTube$/, 'YouTube', 'Channel catalog and transcripts. The Monday window triggers moonshots_yt_sync_daily into MOONSHOTS.'],
  [/^Source feeds/, 'Source feeds', 'Monarch, Gmail, Drive and YouTube. Each lands in its own product database, not a shared dump.'],
  [/^Airflow/, 'Airflow on the office PC', 'Production scheduler (Snowflake tasks are retired). Runs the 07:15 ET pipeline_load_window on INGEST_WH, then dbt and tests.'],
  [/^Landing schema/, 'RAW', 'Landing schema. One row per landed payload, kept as received.'],
  [/^Conformed schema/, 'CORE', 'Conformed schema. One row per business object (account, transaction, document). The write path lands here.'],
  [/^Analytics schema/, 'ANALYTICS', 'Reporting grain built by dbt. Feeds the semantic views.'],
  [/^Semantic views/, 'Semantic views', '15 certified semantic views. Agents answer numeric questions from these, not from raw tables.'],
  [/^Cortex Search/, 'Cortex Search services', '6 Cortex Search services over documents, emails and transcripts for text retrieval.'],
  [/Read MCP server$/, 'Read MCP server', 'MCP_HOUSEHOLD_LEDGER_GROK. Read-only catalog for Grok desks: analyst_* and search_* tools, no write tools.'],
  [/Write MCP server$/, 'Write MCP server', 'MCP_HOUSEHOLD_LEDGER. Used by Cursor / Eta: the same reads plus approved mutate and pipeline procedures.'],
  [/^Read clients$/, 'Read clients', 'Grok bots and desks. They attach to the read MCP server only, not an open warehouse login.'],
  [/^Write client$/, 'Write client', 'Cursor / Eta. Changes go through the write MCP server’s approved procedures into CORE.'],
  [/^Extract and land/, 'Extract and land', 'pipeline_load_window: Monarch and Gmail extracts land into RAW on INGEST_WH.'],
  [/^doc_extract/, 'doc_extract', 'Reads Drive documents with pypdf, then local Tesseract, and lands the text for CORE.'],
  [/^RAW and CORE/, 'RAW and CORE', 'Landed payloads (RAW) and conformed business objects (CORE).'],
  [/^dbt/, 'dbt', 'Builds CORE into ANALYTICS and the semantic views (ledger_dbt_daily).'],
  [/^Tests$/, 'Tests', 'dbt and Airflow test results land in one append-only test-results table.'],
]

export function noteFor(label) {
  const text = String(label || '').replace(/\s+/g, ' ').trim()
  const hit = NODE_NOTES.find(([re]) => re.test(text))
  return hit ? { title: hit[1], text: hit[2] } : null
}
