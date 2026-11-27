// Short purpose notes for clickable process-flow nodes. Matched in order against
// the node's visible label text (all <text> lines joined by spaces).
export const NODE_NOTES = [
  [/^Monarch$/, 'Monarch', 'Bank, card and investment accounts. Pulled each morning through EAI_MONARCH into the ledger’s RAW schema.'],
  [/^Gmail$/, 'Gmail', 'Every bank account is set to send a Gmail alert for each transaction. Those emails come in through EAI_GMAIL, get classified, and are matched to transactions by reconcile_pending.'],
  [/^Drive$/, 'Google Drive', 'Statements and documents. drive_poll picks up new files, and doc_extract reads them with pypdf and local Tesseract instead of Cortex OCR.'],
  [/^YouTube$/, 'YouTube', 'Channel catalog and transcripts. Every Monday the load window starts moonshots_yt_sync_daily, which syncs into MOONSHOTS.'],
  [/^Source feeds/, 'Source feeds', 'Monarch, Gmail, Drive and YouTube. Each lands in its own product database, not a shared dump.'],
  [/^Airflow/, 'Airflow on the office PC', 'Runs every production schedule; Snowflake tasks are retired. Starts the 07:15 ET load window on INGEST_WH, then dbt and tests.'],
  [/^Landing schema/, 'RAW', 'Landing schema. One row per landed payload, kept as received.'],
  [/^Conformed schema/, 'CORE', 'Conformed schema. One row per business object, such as an account, transaction or document. Approved writes land here.'],
  [/^Analytics schema/, 'ANALYTICS', 'Reporting grain built by dbt. Feeds the semantic views.'],
  [/^Semantic views/, 'Semantic views', '15 certified semantic views. Agents answer numeric questions from these, not from raw tables.'],
  [/^Cortex Search/, 'Cortex Search services', '6 Cortex Search services over documents, emails and transcripts for text retrieval.'],
  [/Read MCP server$/, 'Read MCP server', 'MCP_HOUSEHOLD_LEDGER_GROK. Read-only tools for the Grok desks: analyst totals and search, with no way to write.'],
  [/Write MCP server$/, 'Write MCP server', 'MCP_HOUSEHOLD_LEDGER. Used by Cursor and Eta: the same reads plus approved procedures for changes and pipelines.'],
  [/^Read clients$/, 'Read clients', 'Grok bots and desks. They connect only to the read MCP server, never an open warehouse login.'],
  [/^Write client$/, 'Write client', 'Cursor and Eta. Every change goes through an approved procedure on the write MCP server into CORE.'],
  [/^Extract and land/, 'Extract and land', 'The 07:15 ET load window pulls Monarch data and Gmail transaction alerts into RAW on INGEST_WH.'],
  [/^doc_extract/, 'doc_extract', 'Reads Drive documents with pypdf, then local Tesseract, and lands the text for CORE.'],
  [/^RAW and CORE/, 'RAW and CORE', 'RAW keeps payloads as they landed. CORE turns them into clean business objects.'],
  [/^dbt/, 'dbt', 'ledger_dbt_daily builds CORE into ANALYTICS and the semantic views.'],
  [/^Tests$/, 'Tests', 'dbt and Airflow test results land in one append-only test-results table.'],
]

export function noteFor(label) {
  const text = String(label || '').replace(/\s+/g, ' ').trim()
  const hit = NODE_NOTES.find(([re]) => re.test(text))
  return hit ? { title: hit[1], text: hit[2] } : null
}
