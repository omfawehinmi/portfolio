// Pipeline case studies: copy and diagram structure from the design review.
// Node status: 'live' (running), 'paused' (deployed or built, not running), 'planned' (not built).
// Edges with planned: true render dashed and carry the word "planned".

export const pipelinesCopySha = 'd00e0e9'

export const pipelines = [
  {
    id: 'video-transcripts',
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
          { title: 'Video platform API', detail: 'Channel allowlist, published since 2025-01-01', status: 'live' },
        ] },
        { key: 'orchestrate', label: 'Orchestrate', nodes: [
          { title: 'Daily Airflow DAG', detail: 'Office PC; checks a cooldown before any call', status: 'paused' },
        ] },
        { key: 'ingest', label: 'Ingest', nodes: [
          { title: 'Find, fetch, backfill', detail: 'New videos, available transcripts, then gaps', status: 'paused' },
          { title: 'Transcribe locally', detail: 'faster-whisper on the PC GPU, one job at a time', status: 'paused' },
        ] },
        { key: 'store', label: 'Store', nodes: [
          { title: 'Snowflake transcript tables', detail: 'Segments from the load step', status: 'paused' },
          { title: 'Embed', detail: 'Vectors for new rows, behind a flag', status: 'paused' },
        ] },
        { key: 'serve', label: 'Serve', nodes: [
          { title: 'Search index', detail: 'Refreshed on demand, not by this DAG', status: 'paused' },
          { title: 'Agent search tools', detail: 'Serving suspended', status: 'paused' },
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
    number: '02',
    title: 'Document intake with OCR fallback',
    status: 'Deployed, paused pending OAuth client setup',
    statusDetail: 'Both DAGs are on the office Airflow and paused until the drive OAuth client is set up.',
    stack: ['Airflow', 'cloud drive changes feed', 'pypdf', 'Tesseract', 'Snowflake'],
    architecture: {
      ariaLabel: 'Document intake pipeline architecture',
      longDescription:
        'Every 15 minutes a polling DAG reads the cloud drive changes feed. Empty polls stop without a Snowflake connection. Changed files go to an extract DAG that skips files already seen, reads PDF text with pypdf or falls back to Tesseract OCR, then loads through an idempotent procedure into a Snowflake documents table with an audit log. The table feeds a search index when search is enabled.',
      lanes: [
        { key: 'source', label: 'Source', nodes: [
          { title: 'Cloud drive folder', detail: 'Changes feed from a saved page token', status: 'paused' },
        ] },
        { key: 'orchestrate', label: 'Orchestrate', nodes: [
          { title: 'drive_poll', detail: 'Every 15 min, max 200 files per trigger', status: 'paused', note: 'No changes: stop, no Snowflake connection' },
          { title: 'doc_extract', detail: 'Triggered with changed file IDs + md5', status: 'paused' },
        ] },
        { key: 'ingest', label: 'Ingest', nodes: [
          { title: 'Plan step', detail: 'Skip any file ID + md5 already seen', status: 'paused' },
          { title: 'pypdf text', detail: 'PDFs with a text layer', status: 'paused', join: 'or' },
          { title: 'Tesseract OCR', detail: 'Average under 25 chars/page, or any image; 300 DPI, max 40 pages', status: 'paused' },
        ] },
        { key: 'store', label: 'Store', nodes: [
          { title: 'Idempotent load', detail: "Owner's-rights procedure, one row per file", status: 'paused' },
          { title: 'Snowflake documents table', detail: 'Plus an append-only audit log', status: 'paused' },
        ] },
        { key: 'serve', label: 'Serve', nodes: [
          { title: 'Search index', detail: 'Indexed when search is enabled', status: 'paused' },
        ] },
      ],
      edges: [
        { label: 'changed files' },
        { label: 'file IDs + md5' },
        { label: 'text' },
        { label: 'index when search enabled' },
      ],
    },
    problem: [
      'Household documents (statements, bills, scanned pages, phone photos) needed to land as searchable text in Snowflake. The earlier design parsed documents with in-warehouse AI functions on a task that polled every 2 minutes. That meant thousands of warehouse resumes a month and an AI charge on every page.',
    ],
    method: [
      ['Change-based polling.', 'Every 15 minutes, drive_poll reads the cloud drive changes feed from a saved page token, capped at 200 files per trigger. Empty polls never open a Snowflake session.'],
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
    number: '03',
    title: 'Tested analytics deploys with dbt and Airflow',
    status: 'CI running · daily dbt DAGs deployed, paused · the rest planned',
    statusDetail: 'Slim CI, per-PR schemas, prod build on merge, hosted docs and a single Cosmos DAG are planned.',
    stack: ['dbt', 'Airflow', 'GitHub Actions', 'DuckDB', 'sqlfluff', 'Terraform', 'Snowflake'],
    architecture: {
      ariaLabel: 'dbt and Airflow deploy pipeline architecture',
      longDescription:
        'A pull request runs platform-ci on GitHub Actions: dbt deps, parse and compile, two dbt builds on DuckDB, sqlfluff lint and a docs check. Any failure goes back to the pull request. A passing change is reviewed and merged to main. Two daily dbt DAGs are deployed and paused; a single Cosmos DAG is planned. Builds land in Snowflake analytics, where a failed test leaves the previous tables in place. Docs are a CI artifact today; hosted docs are planned.',
      lanes: [
        { key: 'source', label: 'Source', nodes: [
          { title: 'Pull request', detail: 'Platform code and dbt models', status: 'live' },
        ] },
        { key: 'test', label: 'Test', nodes: [
          { title: 'platform-ci', detail: 'GitHub Actions; deps, parse, compile', status: 'live', note: 'Any check fails: fix and push' },
          { title: 'dbt build on DuckDB', detail: 'Two passes: models, then incremental + snapshots', status: 'live' },
          { title: 'Lint + docs check', detail: 'sqlfluff; fail on undocumented model, column or source', status: 'live' },
          { title: 'Slim CI, per-PR schemas', detail: 'Planned: zero-row builds against the published manifest', status: 'planned' },
        ] },
        { key: 'orchestrate', label: 'Orchestrate', nodes: [
          { title: 'Review, merge to main', detail: 'Prod builds only from main', status: 'live' },
          { title: 'Daily dbt DAGs', detail: 'hr_dbt_daily and ledger_dbt_daily', status: 'paused' },
          { title: 'Single Cosmos DAG', detail: 'Planned: one task per model', status: 'planned' },
        ] },
        { key: 'store', label: 'Store', nodes: [
          { title: 'Snowflake analytics', detail: 'Failed test: previous tables stay', status: 'paused' },
        ] },
        { key: 'serve', label: 'Serve', nodes: [
          { title: 'dbt docs + manifest', detail: 'CI artifact today', status: 'live' },
          { title: 'Hosted docs', detail: 'Planned: behind access control', status: 'planned' },
        ] },
      ],
      edges: [
        { label: 'push' },
        { label: 'checks pass' },
        { label: 'build from main' },
        { label: 'docs, hosting planned', planned: true },
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
    number: '04',
    title: 'Daily personal finance load',
    status: 'Running on demand · daily schedule planned',
    statusDetail: 'The daily schedule turns on after the reliability fixes are reviewed and deployed.',
    stack: ['Snowflake stored procedure', 'Snowflake task', 'watermarks'],
    architecture: {
      ariaLabel: 'Personal finance load architecture',
      longDescription:
        'An on-demand trigger, running today, and a planned daily Snowflake task both call an extract procedure. It pulls from a personal finance aggregator API, with a holdings extract alongside, into raw landing tables that merge into the core ledger tables. A run log and watermarks feed a freshness check. Incremental pulls and a downstream dbt build are planned.',
      lanes: [
        { key: 'source', label: 'Source', nodes: [
          { title: 'Finance aggregator API', detail: 'Accounts, transactions, holdings', status: 'live' },
        ] },
        { key: 'orchestrate', label: 'Orchestrate', nodes: [
          { title: 'On-demand trigger', detail: 'Running today', status: 'live' },
          { title: 'Daily Snowflake task', detail: 'Planned, after the reliability fixes', status: 'planned' },
        ] },
        { key: 'ingest', label: 'Ingest', nodes: [
          { title: 'Extract procedure', detail: 'Full lookback now; incremental planned', status: 'live' },
          { title: 'Holdings extract', detail: 'Runs with the load', status: 'live' },
        ] },
        { key: 'store', label: 'Store', nodes: [
          { title: 'Raw landing tables', detail: 'Merged into core ledger tables', status: 'live' },
          { title: 'Run log + watermarks', detail: 'Row counts, request window, caller', status: 'live' },
        ] },
        { key: 'serve', label: 'Serve', nodes: [
          { title: 'Freshness check', detail: 'Planned: warn after 26 h', status: 'planned' },
          { title: 'Ledger dbt models', detail: 'Planned: reads core tables', status: 'planned' },
        ] },
      ],
      edges: [
        { label: 'pull' },
        { label: 'landed rows' },
        { label: 'merge' },
        { label: 'freshness + dbt, planned', planned: true },
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
      'A freshness check in the daily ledger QA warns when the last success is more than 26 hours old.',
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
