// Pipeline case studies: copy and diagram structure from the design review.
// Node state: 'running' (running), 'paused' (deployed or built, not running), 'planned' (not built).
// Edges with state: 'planned' render dashed and carry the word "planned".

export const pipelinesSection = {
  title: 'Data pipelines',
  subtitle: 'Four pipelines built with orchestration, automated tests and cost guardrails from the start.',
}

export const pipelines = [
  {
    id: 'video-transcripts',
    outcome: 'Turns a video library into searchable text with local speech-to-text, no paid transcription service.',
    flow: {
      ariaLabel: 'Process flow',
      longDescription:
        'Process flow, paused. A daily run starts. If the cooldown has not passed, the run is skipped until the next day. If no new videos are found, it stops. Captions are fetched where available; otherwise audio is transcribed locally on the GPU. Segments load, embedding runs only when its flag is on, and transcripts are ready for search.',
      steps: [
        { n: 1, type: 'trigger', label: 'Daily scheduled run', state: 'paused' },
        { n: 2, type: 'decision', label: 'Cooldown passed?', state: 'paused', branch: { label: 'No', text: 'Skip run, try next day', state: 'paused' } },
        { n: 3, type: 'decision', label: 'New videos found?', state: 'paused', branch: { label: 'No', text: 'Stop, nothing to load', state: 'paused' } },
        { n: 4, type: 'decision', label: 'Captions available?', state: 'paused', branch: { label: 'No', text: 'Transcribe locally on the GPU', state: 'paused' } },
        { n: 5, type: 'action', label: 'Load transcript segments', state: 'paused' },
        { n: 6, type: 'decision', label: 'Embeddings flag on?', state: 'paused', branch: { label: 'No', text: 'Skip embedding', state: 'paused' } },
        { n: 7, type: 'outcome', label: 'Transcripts ready for search', state: 'paused' },
      ],
    },
    number: '01',
    title: 'Video transcripts at scale with local speech-to-text',
    status: 'Deployed, paused pending cooldown',
    statusDetail: 'The DAG is merged and deployed to the office Airflow, and a test run passed. The full backlog is not yet approved.',
    stack: ['Airflow', 'faster-whisper on a local GPU', 'Snowflake'],
    architecture: {
      ariaLabel: 'Video transcript pipeline architecture',
      longDescription:
        'A video platform API feeds one daily Airflow DAG on the office PC. The DAG checks a cooldown, finds new videos, fetches transcripts, backfills gaps and transcribes the rest locally, then loads segments and vectors into Snowflake. A search index built on those tables is refreshed on demand and serves agent search tools, which are suspended today.',
      lanes: [
        { key: 'source', label: 'Source', nodes: [
          { title: 'Video platform API', detail: 'Channel allowlist, published since 2025-01-01', state: 'running' },
        ] },
        { key: 'orchestrate', label: 'Orchestrate', nodes: [
          { title: 'Daily Airflow DAG', detail: 'Office PC; checks a cooldown before any call', state: 'paused' },
        ] },
        { key: 'ingest', label: 'Ingest', nodes: [
          { title: 'Find, fetch, backfill', detail: 'New videos, available transcripts, then gaps', state: 'paused' },
          { title: 'Transcribe locally', detail: 'faster-whisper on the PC GPU, one job at a time', state: 'paused' },
        ] },
        { key: 'store', label: 'Store', nodes: [
          { title: 'Snowflake transcript tables', detail: 'Segments from the load step', state: 'running' },
          { title: 'Embed', detail: 'Vectors for new rows, behind a flag', state: 'paused' },
        ] },
        { key: 'serve', label: 'Serve', nodes: [
          { title: 'Search index', detail: 'Refreshed on demand, not by this DAG', state: 'running' },
          { title: 'Agent search tools', detail: 'Serving suspended', state: 'paused' },
        ] },
      ],
      edges: [
        { label: 'new videos' },
        { label: 'audio + captions' },
        { label: 'transcript segments' },
        { label: 'index, on demand' },
      ],
    },
    problem: [
      'A transcript library for an allowlist of video channels feeds search and research. Caption coverage was poor: a read on 2026-10-25 found 87.3% of the 544 in-scope videos had no real captions (222 placeholder transcripts, 253 missing). Caption scraping was also failing. Scheduled caption jobs hit video platform IP blocks (HTTP 403), and a warehouse-hosted sync task timed out 9 times in 30 days and was suspended.',
    ],
    architectureNotes: [
      'Orchestration: one daily Airflow DAG on the office PC (Docker Compose under WSL2, LocalExecutor, Postgres metadata).',
      'Compute: local faster-whisper on the PC GPU, so transcription needs no paid service.',
      'Storage: Snowflake. The search index is refreshed separately, on demand.',
    ],
    method: [
      ['Scope cut first.', 'Videos published before 2025-01-01 and out-of-scope channels were removed, with backup clones kept. That left 544 videos.'],
      ['Gap classification.', 'A simple SQL rule labels each video full, placeholder or missing, so the backlog is measurable by channel, year and month.'],
      ['Cooldown, then incremental.', 'The DAG checks a cooldown before calling the video platform, finds new videos, fetches available transcripts, backfills gaps, and transcribes only what is still missing.'],
      ['Audio chunking.', 'Audio is transcribed in 10-minute (600-second) windows.'],
      ['Embedding behind a flag.', 'The embed step stays off unless enabled.'],
      ['Environments.', 'Dev runs a 10-video sample; staging runs a 10 + 10 GPU test; prod loads approved batches only.'],
    ],
    reliability: [
      'One GPU job at a time (Airflow pool of 1).',
      'Offline unit tests with stubs run in CI, plus a guard test that checks audio duration is at least 180 seconds.',
      'Rollback: delete loaded rows by batch id, or redeploy the previous DAG commit.',
    ],
    cost: [
      'Transcription runs on owned hardware. No paid transcription API.',
      'The old warehouse sync task stays suspended. Sync runs daily at most, incremental, on capped compute.',
      'Scratch audio files are cleaned per video to protect local disk.',
    ],
    shows: 'How a failing cloud task moved to local GPU compute under one Airflow DAG, with measured scope, gated environments and a clear rollback.',
  },
  {
    id: 'document-intake',
    outcome: 'Turns scanned and digital documents into warehouse rows, reprocessing only files that changed.',
    flow: {
      ariaLabel: 'Process flow',
      longDescription:
        'Process flow, paused. A folder change is detected; if nothing is new it stops without a warehouse connection. Files already seen are skipped. PDFs with a text layer averaging 25 or more characters per page are read with pypdf; others and images go to Tesseract OCR. Text loads idempotently with an audit log, updating the documents table. Search indexing is planned.',
      steps: [
        { n: 1, type: 'trigger', label: 'Folder change detected', state: 'paused' },
        { n: 2, type: 'decision', label: 'Any new or changed files?', state: 'paused', branch: { label: 'No', text: 'Stop, no warehouse connection', state: 'paused' } },
        { n: 3, type: 'action', label: 'Plan: skip files already seen (file ID + checksum)', state: 'paused' },
        { n: 4, type: 'decision', label: 'Text layer, average 25+ chars/page?', state: 'paused', branch: { label: 'No, or image', text: 'Tesseract OCR', state: 'paused' } },
        { n: 5, type: 'action', label: 'Read text with pypdf', state: 'paused' },
        { n: 6, type: 'action', label: 'Idempotent load + audit log', state: 'paused' },
        { n: 7, type: 'outcome', label: 'Documents table updated', state: 'paused', branch: { label: 'Planned', text: 'Indexed for document search', state: 'planned' } },
      ],
    },
    number: '02',
    title: 'Document intake with OCR fallback',
    status: 'Built, paused pending OAuth client setup',
    statusDetail: 'Both DAGs are deployed to the office Airflow and paused until the drive OAuth client is set up.',
    stack: ['Airflow', 'cloud drive changes feed', 'pypdf', 'Tesseract', 'Snowflake'],
    architecture: {
      ariaLabel: 'Document intake pipeline architecture',
      longDescription:
        'Every 15 minutes a polling DAG reads the cloud drive changes feed. Empty polls stop without a Snowflake connection. Changed files go to an extract DAG that skips files already seen, reads PDF text with pypdf or falls back to Tesseract OCR, then loads through an idempotent procedure into a Snowflake documents table with an audit log. The table feeds a search index when search is enabled.',
      lanes: [
        { key: 'source', label: 'Source', nodes: [
          { title: 'Cloud drive folder', detail: 'Changes feed from a saved page token', state: 'paused' },
        ] },
        { key: 'orchestrate', label: 'Orchestrate', nodes: [
          { title: 'drive_poll', detail: 'Every 15 min when enabled, currently paused; max 200 files per trigger', state: 'paused', note: 'No changes: stop, no Snowflake connection' },
          { title: 'doc_extract', detail: 'Triggered with changed file IDs + md5', state: 'paused' },
        ] },
        { key: 'ingest', label: 'Ingest', nodes: [
          { title: 'Plan step', detail: 'Skip any file ID + md5 already seen', state: 'paused' },
          { title: 'pypdf text', detail: 'PDFs with a text layer', state: 'paused', join: 'or' },
          { title: 'Tesseract OCR', detail: 'Average under 25 chars/page, or any image; 300 DPI, max 40 pages', state: 'paused' },
        ] },
        { key: 'store', label: 'Store', nodes: [
          { title: 'Idempotent load', detail: "Owner's-rights procedure, one row per file", state: 'paused' },
          { title: 'Snowflake documents table', detail: 'Plus an append-only audit log', state: 'paused' },
        ] },
        { key: 'serve', label: 'Serve', nodes: [
          { title: 'Search index', detail: 'Planned: indexed when search is enabled', state: 'planned' },
        ] },
      ],
      edges: [
        { label: 'changed files' },
        { label: 'file IDs + md5' },
        { label: 'text' },
        { label: 'index when search enabled, planned', state: 'planned' },
      ],
    },
    problem: [
      'Household documents (statements, bills, scanned pages, phone photos) needed to land as searchable text in Snowflake. The earlier design parsed documents with in-warehouse AI functions on a task that polled every 2 minutes. That meant thousands of warehouse resumes a month and an AI charge on every page.',
    ],
    method: [
      ['Change-based polling.', 'Polls every 15 minutes when enabled (currently paused). drive_poll reads the cloud drive changes feed from a saved page token, capped at 200 files per trigger. Empty polls never open a Snowflake session.'],
      ['Extract DAG.', 'Changed files are handed to the doc_extract DAG, which plans, extracts and loads.'],
      ['Plan before extract.', 'A plan step skips any file whose file ID and md5 were already seen, so only new or changed files are processed.'],
      ['Text first, OCR second.', 'Extraction reads the PDF text layer with pypdf. A PDF averaging under 25 characters per page goes to Tesseract. Images go straight to OCR at 300 DPI, up to 40 pages.'],
      ['Idempotent load with audit.', "Writes go through an owner's-rights procedure, so a rerun does not duplicate rows, and each load is recorded in an audit log."],
      ['AI off by flag.', 'In-warehouse document AI stays off; turning it on is a one-row change that needs approval.'],
      ['Options weighed.', 'A cloud event path was designed and security reviewed, then parked as a future option.'],
    ],
    reliability: [
      'Watermark-based catch-up: if the PC is down, the next poll reads everything new since the last token.',
      'Planned PC hardening: never sleep on AC power, auto-start at boot, heartbeat row with a freshness check.',
      'Old Snowflake tasks stay suspended, not dropped, as rollback.',
    ],
    cost: [
      'Planning volumes: 500 docs a month for the pilot, 5,000 for growth.',
      'OCR runs on local CPU, so there are no per-page AI charges.',
      'No hourly warehouse COPY. Loads run in batches on a dedicated XS warehouse with a resource monitor that suspends at a set credit cap.',
    ],
    security: [
      'Dedicated service user with key-pair auth and a least-privilege role. No admin role.',
      'Airflow UI bound to localhost. Secrets in a gitignored env file, masked in logs. Disk encryption required.',
    ],
    shows: 'Change-driven orchestration and open-source OCR in place of a high-frequency warehouse poll and per-page AI.',
  },
  {
    id: 'analytics-deploys',
    outcome: 'Every model change is built, linted and documented in CI before it can reach production.',
    flow: {
      ariaLabel: 'Process flow',
      longDescription:
        'Process flow, CI running. A pull request runs CI: two dbt builds on DuckDB, a lint and a docs check. Any failure sends it back for a fix and a new push. After review and merge, the daily dbt run (paused today) builds the tables; a failed test keeps the previous tables, otherwise analytics tables update.',
      steps: [
        { n: 1, type: 'trigger', label: 'Pull request opened or updated', state: 'running' },
        { n: 2, type: 'action', label: 'CI: dbt build twice on DuckDB, lint, docs check', state: 'running' },
        { n: 3, type: 'decision', label: 'All checks pass?', state: 'running', branch: { label: 'No', text: 'Fix and push again', state: 'running', loopsTo: 1 } },
        { n: 4, type: 'action', label: 'Owner review + merge', state: 'running' },
        { n: 5, type: 'action', label: 'Daily dbt DAG run', state: 'paused' },
        { n: 6, type: 'decision', label: 'Tests pass?', state: 'paused', branch: { label: 'No', text: 'Previous tables stay', state: 'paused' } },
        { n: 7, type: 'outcome', label: 'Analytics tables updated', state: 'paused' },
      ],
    },
    number: '03',
    title: 'Tested analytics deploys with dbt and Airflow',
    status: 'CI: Running · Daily dbt DAGs: Deployed, paused',
    statusDetail: 'Slim CI, per-PR schemas, prod build on merge, hosted docs and a single Cosmos DAG are planned.',
    stack: ['dbt', 'Airflow', 'GitHub Actions', 'DuckDB', 'sqlfluff', 'Terraform', 'Snowflake'],
    architecture: {
      ariaLabel: 'dbt and Airflow deploy pipeline architecture',
      longDescription:
        'A pull request runs platform-ci on GitHub Actions: dbt deps, parse and compile, two dbt builds on DuckDB, sqlfluff lint and a docs check. Any failure goes back to the pull request. A passing change is reviewed and merged to main. Two daily dbt DAGs are deployed and paused; a single Cosmos DAG is planned. Builds land in Snowflake analytics, where a failed test leaves the previous tables in place. Docs are a CI artifact today; hosted docs are planned.',
      lanes: [
        { key: 'source', label: 'Source', nodes: [
          { title: 'Pull request', detail: 'Platform code and dbt models', state: 'running' },
        ] },
        { key: 'test', label: 'Test', nodes: [
          { title: 'platform-ci', detail: 'GitHub Actions; deps, parse, compile', state: 'running', note: 'Any check fails: fix and push' },
          { title: 'dbt build on DuckDB', detail: 'Two passes: models, then incremental + snapshots', state: 'running' },
          { title: 'Lint + docs check', detail: 'sqlfluff; fail on undocumented model, column or source', state: 'running' },
          { title: 'Slim CI, per-PR schemas', detail: 'Planned: zero-row builds against the published manifest', state: 'planned' },
        ] },
        { key: 'orchestrate', label: 'Orchestrate', nodes: [
          { title: 'Review, merge to main', detail: 'Prod builds only from main', state: 'running' },
          { title: 'Daily dbt DAGs', detail: 'hr_dbt_daily and ledger_dbt_daily', state: 'paused' },
          { title: 'Single Cosmos DAG', detail: 'Planned: one task per model', state: 'planned' },
        ] },
        { key: 'store', label: 'Store', nodes: [
          { title: 'Snowflake analytics', detail: 'Failed test: previous tables stay', state: 'paused' },
        ] },
        { key: 'serve', label: 'Serve', nodes: [
          { title: 'dbt docs + manifest', detail: 'CI artifact today', state: 'running' },
          { title: 'Hosted docs', detail: 'Planned: behind access control', state: 'planned' },
        ] },
      ],
      edges: [
        { label: 'push' },
        { label: 'checks pass' },
        { label: 'build from main' },
        { label: 'docs, hosting planned', state: 'planned' },
      ],
    },
    problem: [
      'A single 8 KB stored procedure rebuilt the analytics layer. It ran incremental MERGEs into 9 dimension and fact tables and, as a side effect, wrote to a core table. Snowflake objects were managed by hand-applied SQL. There were no model tests and no promotion path.',
    ],
    method: [
      ['CI today.', 'On each pull request, platform-ci runs dbt deps, parse and compile, then dbt build --target ci twice on DuckDB. The second pass exercises incremental models and snapshots. It also runs sqlfluff lint, and dbt docs generate plus a check that fails on any undocumented model, column or source. No warehouse is touched.'],
      ['Daily builds.', 'Two Airflow DAGs, hr_dbt_daily and ledger_dbt_daily, run dbt builds per domain on a daily schedule, with a DAG import test. Both are deployed to the office Airflow and paused, with no scheduled runs yet.'],
      ['Planned next.', 'Slim CI with zero-row builds against the published manifest, transient per-PR schemas with cleanup on close, a prod build on merge, hosted docs behind access control, and a single Astronomer Cosmos DAG with one task per model.'],
      ['Migration.', 'The 9 analytics targets become dbt models with enforced contracts, the core-table side effect becomes a seed, and a 7-day parallel run compares row counts and key sums before cutover.'],
      ['Infrastructure as code.', 'Roles, grants, warehouses and monitors move to Terraform. Terraform fmt and validate already run in GitHub Actions.'],
    ],
    reliability: [
      'Prod builds only from main.',
      'A failing test stops downstream models, and existing tables stay in place.',
      'Rollback: pause the DAG and drop new objects; after cutover, rerun the retained procedure and restore from a zero-copy clone.',
      'Separate CI and prod service identities. No secrets in the repo.',
    ],
    cost: [
      'CI runs on DuckDB, so pull requests use no warehouse credits.',
      'XS warehouses only, 60-second auto-suspend plus an explicit suspend after each run.',
      'Planned split resource monitors: 3 credits a month for prod, 2 for CI and dev, suspending at 100%.',
      'GitHub-hosted runners within included minutes. No paid CI.',
    ],
    shows: 'Standard software delivery on a small warehouse: tested models, gated promotion, capped compute and a rollback at every phase.',
  },
  {
    id: 'finance-load',
    outcome: 'Loads personal finance data into the warehouse incrementally, on demand today with a daily schedule planned.',
    flow: {
      ariaLabel: 'Process flow',
      longDescription:
        'Process flow, on demand running. A run starts on demand; a daily schedule is planned. Planned: a weekly full pull, otherwise an incremental pull from the watermark minus 14 days; today every run uses the full lookback. Transactions and holdings land raw, merge into core ledger tables. Planned: a freshness check that warns if data is over 26 hours old.',
      steps: [
        { n: 1, type: 'trigger', label: 'On-demand run', state: 'running', branch: { label: 'Planned', text: 'Daily schedule', state: 'planned' } },
        { n: 2, type: 'decision', label: 'Weekly full due?', state: 'planned', branch: { label: 'No', text: 'Pull from watermark minus 14 days', state: 'planned' } },
        { n: 3, type: 'action', label: 'Pull transactions + holdings (full lookback today)', state: 'running' },
        { n: 4, type: 'action', label: 'Land raw rows', state: 'running' },
        { n: 5, type: 'action', label: 'Merge into core ledger tables', state: 'running' },
        { n: 6, type: 'decision', label: 'Fresh within 26 hours? (planned)', state: 'planned', branch: { label: 'No', text: 'Warn in run log', state: 'planned' } },
        { n: 7, type: 'outcome', label: 'Ledger up to date', state: 'running' },
      ],
    },
    number: '04',
    title: 'Daily personal finance load',
    status: 'On demand: Running · Daily schedule: Planned',
    statusDetail: 'The daily schedule turns on after the reliability fixes are reviewed and deployed.',
    stack: ['Snowflake stored procedure', 'Snowflake task', 'watermarks'],
    architecture: {
      ariaLabel: 'Personal finance load architecture',
      longDescription:
        'An on-demand trigger, running today, and a planned daily Snowflake task both call an extract procedure. It pulls from a personal finance aggregator API, with a holdings extract alongside, into raw landing tables that merge into the core ledger tables. A run log and watermarks are written today. A freshness check, incremental pulls and a downstream dbt build are planned.',
      lanes: [
        { key: 'source', label: 'Source', nodes: [
          { title: 'Finance aggregator API', detail: 'Accounts, transactions, holdings', state: 'running' },
        ] },
        { key: 'orchestrate', label: 'Orchestrate', nodes: [
          { title: 'On-demand trigger', detail: 'Running today', state: 'running' },
          { title: 'Daily Snowflake task', detail: 'Planned, after the reliability fixes', state: 'planned' },
        ] },
        { key: 'ingest', label: 'Ingest', nodes: [
          { title: 'Extract procedure', detail: 'Full lookback now; incremental planned', state: 'running' },
          { title: 'Holdings extract', detail: 'Runs with the load', state: 'running' },
        ] },
        { key: 'store', label: 'Store', nodes: [
          { title: 'Raw landing tables', detail: 'Merged into core ledger tables', state: 'running' },
          { title: 'Run log + watermarks', detail: 'Row counts, request window, caller', state: 'running' },
        ] },
        { key: 'serve', label: 'Serve', nodes: [
          { title: 'Freshness check', detail: 'Planned: warn after 26 h', state: 'planned' },
          { title: 'Ledger dbt models', detail: 'Planned: reads core tables', state: 'planned' },
        ] },
      ],
      edges: [
        { label: 'pull' },
        { label: 'landed rows' },
        { label: 'merge' },
        { label: 'warn after 26 h, planned', state: 'planned' },
      ],
    },
    problem: [
      'A personal finance aggregator feeds account and transaction data into the ledger. The loader worked, but an assessment found gaps:',
    ],
    problemPoints: [
      '4 runs stuck in a started state with no error recorded.',
      'One follow-up request pending for a month.',
      'A 400-day lookback re-merged on every run, about 3,700 rows each time.',
      'TLS verification turned off.',
      'No alert when the API token expired.',
    ],
    method: [
      ['Trigger.', 'On demand today. Planned: daily on a Snowflake task, with on demand kept for ad hoc refreshes.'],
      ['Incremental pulls (planned).', 'Each run starts at the last watermark minus 14 days, with a weekly full pull of about 400 days. Today each run re-merges a 400-day lookback.'],
      ['Holdings.', 'A separate holdings extract procedure runs with the load.'],
      ['Auditability.', 'The run log records per-entity row counts, the request window and the caller (schedule or user).'],
    ],
    reliabilityIntro: 'These fixes come before the schedule is turned on:',
    reliabilityOrdered: true,
    reliability: [
      'Failures are caught and written to the run log with the error message, so no run is left orphaned.',
      'Retry with backoff on rate-limit and server errors, plus an alert when the token is rejected.',
      'TLS verification turned back on.',
      'Stale follow-up requests swept or expired.',
      'Planned: a freshness check in the daily ledger QA that warns when the last success is more than 26 hours old.',
    ],
    cost: [
      'Measured baseline: average run 37 seconds, max 90 seconds.',
      'Stays on the existing XS warehouse with a short auto-suspend. No new warehouse.',
      'Incremental pulls cut the rows merged per run.',
      'No AI functions in the path.',
    ],
    shows: 'Assessment before automation: read the run history, fix silent failures, then schedule. Small, measured and reversible.',
  },
]
