import { publicUrl } from '../utils.js'

const CAMERA_EMBED =
  'https://player.vimeo.com/video/938258584?background=1&autoplay=1&loop=1&muted=1&autopause=0'
const CAMERA_VIDEO =
  'https://player.vimeo.com/video/938258584?badge=0&autopause=0&player_id=0&app_id=58479'

export const projects = [
  {
    id: 'myorglab',
    group: 'Production',
    icon: 'uil-snowflake',
    title: 'MyOrgLab',
    outcome:
      'Customer practice built like HOUSEHOLD_LEDGER: in-account Cortex agents, semantic views, and MCP bridges — not a multi-tenant copy of their data.',
    modalTitle: 'MyOrgLab: governed agents in the customer account',
    thumb: publicUrl('assets/img/myorglab.png'),
    summary: [
      'The same control plane I run on HOUSEHOLD_LEDGER, packaged for customers: domain analyst agents on semantic views, an orchestrator for multi-step work, and MCP bridges so other agents reuse that expertise under the customer’s identity and policy.',
    ],
    points: [
      'Runs in the customer’s Snowflake account: not a multi-tenant SaaS copy of their data.',
      'Migrations, Streamlit, and Power BI on the same certified grain the agents read.',
    ],
    links: [{ href: 'https://www.myorglab.com/', label: 'myorglab.com' }],
  },
  {
    id: 'semantic-hr',
    group: 'Production',
    icon: 'uil-users-alt',
    title: 'HR Intelligence',
    outcome: 'Workday and SuccessFactors as governed Snowflake semantics: the memory agents actually use.',
    modalTitle: 'HR Intelligence: workforce intelligence on Snowflake',
    thumb: publicUrl('assets/img/semantic-hr.png'),
    summary: [
      'Pulled out of the MyOrgLab stack because it is the piece hiring managers probe first: HRIS grain, certified metrics, and talk-to-data on the same warehouse.',
      'Workday, SuccessFactors, Fieldglass, Eightfold, Visier, and One Model land into Snowflake with row access and masking, then Power BI / Streamlit / Cortex sit on that layer.',
    ],
    points: [
      'Same pattern I ran as sole engineer on the Workday-to-Snowflake integration at Novanta and as the HRIS landing engineer at NetApp.',
      'Agents do not invent headcount or attrition; they read the semantic layer.',
    ],
    links: [{ href: 'https://www.myorglab.com/semantic-hr', label: 'HR Intelligence' }],
  },
  {
    id: 'cortex-self-serve',
    group: 'Production',
    icon: 'uil-processor',
    title: 'Cortex self-service',
    outcome: 'Workday or SuccessFactors into certified Snowflake grain, so Tier 0 and Tier 1 reporting queues go away.',
    modalTitle: 'Cortex self-service: HRIS in, reporting queues out',
    thumb: publicUrl('assets/img/cortex-self-serve.png'),
    thumbPos: 'center',
    summary: [
      'The point is not a chatbot. Land Workday or SuccessFactors into Snowflake with certified metrics, then put Cortex on that grain so the business asks instead of opening a Tier 0 or Tier 1 reporting ticket.',
      'Stood up that pattern at Novanta (Workday) and at NetApp (SuccessFactors). At Novanta the reporting queues went to zero once domain logic lived in the warehouse Cortex already reads.',
    ],
    points: [
      'Workday or SuccessFactors: API to raw/stage/prod, then row access and masking on the same grain Cortex uses.',
      'Tier 0/1 tickets die because the question already has a certified answer in Snowflake.',
      'Novanta: sole engineer on the Workday-to-Snowflake integration, then Cortex, queues to zero. NetApp: SuccessFactors (plus Fieldglass, Eightfold, Visier) into Snowflake on the same idea.',
    ],
  },
  {
    id: 'sof',
    group: 'Financial services',
    featured: true,
    icon: 'uil-briefcase-alt',
    title: 'SOF Financial Services',
    outcome: 'Operating financial-services firm plus the client portal clients actually log into.',
    modalTitle: 'SOF Financial Services: tax, books, and the client portal',
    thumb: publicUrl('assets/img/sof-overview.png'),
    thumbPos: 'center top',
    parts: ['sof-portal', 'sof-tax', 'sof-finance'],
    summary: [
      'SOF Financial Services is a Lynn / Boston firm that has offered bookkeeping, tax planning, payroll, and business financing since 2010. The public site is the practice; the product I built and still operate is the client portal behind it.',
      'React on the front, Flask on the back, AWS underneath. Clients use it to send W-2s and 1099s, watch tax and bookkeeping work move, and message the finance team: not to read a brochure.',
    ],
    points: [
      'Firm services on the live site: monthly books, statements, tax prep, IRS audit representation, invoicing, controllership, reconciliation, bill pay, payroll, and financing.',
      'Portal is the operating system: document vault, messaging center, and real-time status, with QuickBooks and TaxAct on the books-and-returns side.',
      'Company positioning on soffinancialservices.com: 20+ years in business, 5,000+ clients served, 50+ years combined experience. Those are the firm’s numbers, not a personal KPI.',
    ],
    links: [{ href: 'https://www.soffinancialservices.com/', label: 'soffinancialservices.com' }],
  },
  {
    id: 'sof-portal',
    group: 'Financial services',
    parentId: 'sof',
    icon: 'uil-lock-alt',
    title: 'Client portal',
    outcome: 'AWS + MFA document vault, messaging center, and live status on the work.',
    modalTitle: 'SOF client portal: documents, messages, MFA',
    thumb: publicUrl('assets/img/sof-portal.png'),
    thumbPos: 'center top',
    summary: [
      'This is the product. Clients land in a portal with Upload, Documents, and Messages: the same pattern the live site walks through under “Our Advanced Client Portal.”',
      'Sensitive files (returns, W-2s, 1099s, expense reports) sit in cloud storage on AWS with multi-factor authentication. The messaging center is how the finance team and the client stay on one thread instead of emailing attachments around.',
    ],
    points: [
      'Secure cloud file sharing and record management: not a shared inbox.',
      'Real-time updates and notifications as tax and bookkeeping work moves.',
      'File transfer into the same vault the team uses for monthly books, tax prep, and statements.',
    ],
    links: [{ href: 'https://www.soffinancialservices.com/', label: 'soffinancialservices.com' }],
  },
  {
    id: 'sof-tax',
    group: 'Financial services',
    parentId: 'sof',
    icon: 'uil-file-alt',
    title: 'Tax & bookkeeping',
    outcome: 'QuickBooks + TaxAct workflows: books, statements, and returns through the portal.',
    modalTitle: 'SOF tax and bookkeeping',
    thumb: publicUrl('assets/img/sof-tax.png'),
    thumbPos: 'center top',
    summary: [
      'The firm’s core practice is year-round books and tax, not a once-a-year drop box. Monthly bookkeeping, financial statements, account reconciliation, invoicing and collections, bill payment, and controllership all sit on the same client relationship.',
      'Tax planning and preparation, plus IRS audit representation, use that same document trail. QuickBooks and TaxAct are the production tools; the portal is how clients feed the work and see progress.',
    ],
    points: [
      'Books and statements stay current so tax is not a scramble.',
      'Clients send source docs through the vault instead of chasing email.',
      'Payroll is in the same service mix as books and tax on the public site.',
    ],
    links: [{ href: 'https://www.soffinancialservices.com/', label: 'soffinancialservices.com' }],
  },
  {
    id: 'sof-finance',
    group: 'Financial services',
    parentId: 'sof',
    icon: 'uil-chart-bar',
    title: 'Business financing',
    outcome: 'Needs assessments and loan packages built from the same books.',
    modalTitle: 'SOF business financing',
    thumb: publicUrl('assets/img/sof-finance.png'),
    thumbPos: 'center',
    summary: [
      'Alongside tax, the firm runs financial needs assessments and prepares loan packages with in-depth financials: the “Finance” lane on the live site. Business plan preparation sits in the same stack when a client is raising or borrowing.',
      'That only works if the books are real. The portal’s statements and document vault are the packet; financing is not a separate product bolted on later.',
    ],
    points: [
      'Loan packages pull from the same statements the portal already holds.',
      'Payroll and financing are listed with books and tax as the firm’s full offer.',
    ],
    links: [{ href: 'https://www.soffinancialservices.com/', label: 'soffinancialservices.com' }],
  },
  {
    id: 'private-platform',
    group: 'Private platform',
    featured: true,
    icon: 'uil-usd-circle',
    title: 'HOUSEHOLD_LEDGER agent system',
    outcome:
      'Production Snowflake finance platform: Monarch, Gmail, and Drive feed CORE; Cortex agents ask through MCP_HOUSEHOLD_LEDGER_GROK and write through MCP_HOUSEHOLD_LEDGER.',
    modalTitle: 'HOUSEHOLD_LEDGER: production agent system',
    parts: [
      'household-gmail',
      'household-stage',
      'source-extract',
      'household-grok-mcp',
      'household-semantics',
      'household-tools',
    ],
    summary: [
      'HOUSEHOLD_LEDGER is the warehouse I operate. Monarch is the system of record for posted accounts and transactions. Gmail card alerts land early through EAI_GMAIL. Drive documents land through drive_poll / doc_extract. Agents answer from semantic views and Cortex Search services catalogs instead of inventing missing figures.',
      'AGENT_HOUSEHOLD_FINANCE and AGENT_LEDGER_OPS sit on top. Cursor uses MCP_HOUSEHOLD_LEDGER (write). Grok finance bots use MCP_HOUSEHOLD_LEDGER_GROK (read-only). Same account, split catalogs.',
    ],
    points: [
      'Schemas: RAW land, CORE truth, ANALYTICS marts + semantic views, CTRL jobs / agents / MCP.',
      'Sibling prod DBs: MOONSHOTS, AGENTIC_WORKFORCE, CONTROL. Airflow on the office PC is the scheduler.',
    ],
  },
  {
    id: 'household-gmail',
    group: 'Private platform',
    parentId: 'private-platform',
    icon: 'uil-envelope',
    title: 'Gmail early spend',
    outcome:
      'Card alerts land in HOUSEHOLD_LEDGER hours before Monarch posts: EAI_GMAIL → proc_gmail_ingest → CS_EMAIL_MESSAGES.',
    modalTitle: 'Gmail path: EAI_GMAIL and CS_EMAIL_MESSAGES',
    summary: [
      'EAI_GMAIL pulls MIME into RAW.EMAIL_MESSAGES_RAW. proc_email_pipeline classifies unclear bank mail and parses pending spend. proc_reconcile_pending matches pending email rows to posted Monarch rows.',
      'Cortex Search CS_EMAIL_MESSAGES embeds the mailbox so household_finance_agent can retrieve what the card alert said without guessing.',
    ],
    points: [
      'EAI_GMAIL → proc_gmail_ingest → EMAIL_MESSAGES_RAW → pending bridge.',
      'SV_EARLY_ACTIVITY is email pending plus Monarch pending — not posted spend.',
      'Search is RAG on CORE email text, not the agent inventing a charge.',
    ],
  },
  {
    id: 'household-stage',
    group: 'Private platform',
    parentId: 'private-platform',
    icon: 'uil-file-alt',
    title: 'Document stage embedding',
    outcome:
      'PDFs and Drive files land on DOCUMENTS_STAGE; proc_document_extract pulls dues and rates; CS_DOCUMENTS makes them searchable.',
    modalTitle: 'Documents: DOCUMENTS_STAGE → CORE → CS_DOCUMENTS',
    summary: [
      'Loan contracts, statements, and rate letters do not live cleanly in Monarch. drive_poll stages bytes; doc_extract runs pypdf then local Tesseract (no Cortex OCR); proc_document_extract writes dues and rates into CORE.',
      'CS_DOCUMENTS and SV_DOCUMENT_STANDING let the agent cite a standing document instead of inventing a rate. doc_extract has been paused for warehouse quota on some pins — status is called out in the as-built.',
    ],
    points: [
      'DOCUMENTS_STAGE / DOCUMENTS_DOWNLOAD_STAGE are the file system of record inside the account.',
      'Extracted fields feed SV_DOCUMENT_STANDING and SV_LIABILITIES_DUES.',
      'Intake is Airflow drive_poll → doc_extract → Snowflake stages — not object storage pipes.',
    ],
  },
  {
    id: 'source-extract',
    group: 'Private platform',
    parentId: 'private-platform',
    icon: 'uil-sync',
    title: 'Monarch extract',
    outcome:
      'Posted Monarch activity via EAI_MONARCH and proc_monarch_extract into RAW/CORE. Community MCP is a client fallback, not the system of record.',
    modalTitle: 'Monarch GraphQL extract (EAI_MONARCH)',
    summary: [
      'Monarch stays the place activity is posted. Snowflake copies accounts and transactions through EAI_MONARCH → proc_monarch_extract (and proc_monarch_holdings_extract for brokerage positions into CORE.HOLDINGS), MERGE into RAW and CORE.',
      'proc_monarch_mutate and proc_ledger_mutate are write-MCP tools for approved changes. If the warehouse path is down, Cursor can still talk to Monarch through a community MCP client — that does not replace the Snowflake extract as truth.',
    ],
    points: [
      'Account dedupe on mask + institution + type; proc_rebuild_analytics after MERGE.',
      'proc_enqueue_monarch_extract_followup queues a delayed extract after a refresh.',
      'Email pending never overwrites figures already posted in Monarch.',
    ],
  },
  {
    id: 'household-grok-mcp',
    group: 'Private platform',
    parentId: 'private-platform',
    icon: 'uil-comments',
    title: 'MCP_HOUSEHOLD_LEDGER_GROK',
    outcome:
      'Read-only Snowflake MCP so Grok bots can ask HOUSEHOLD_LEDGER without write tools.',
    modalTitle: 'Read MCP: MCP_HOUSEHOLD_LEDGER_GROK',
    summary: [
      'Cursor and Cowork attach to MCP_HOUSEHOLD_LEDGER (agents, analyst, search, mutate/pipeline procs). Grok finance bots attach to MCP_HOUSEHOLD_LEDGER_GROK with no write tools.',
      'MCP_MOONSHOTS on MOONSHOTS.CTRL is a separate YouTube catalog server — not household money. Auth is a Snowflake PAT under role MCP_LEDGER_CLIENT; the PAT is not in git.',
    ],
    points: [
      'MCP is the API catalog the bots are allowed to see, not the agent itself.',
      'Read/write split is the governance: Grok can analyze; writes stay on MCP_HOUSEHOLD_LEDGER.',
    ],
  },
  {
    id: 'household-semantics',
    group: 'Private platform',
    parentId: 'private-platform',
    icon: 'uil-chart-line',
    title: 'semantic views',
    outcome:
      'ANALYTICS star schema plus Cortex Analyst contracts: agents query semantic views, not ad-hoc vibes.',
    modalTitle: 'ANALYTICS semantic views',
    summary: [
      'CORE holds operational truth. ANALYTICS holds dims, facts, and semantic views so Cortex Analyst answers without ad-hoc SQL every turn: SV_SPENDING, SV_CASHFLOW, SV_BALANCES_NET_WORTH, SV_LIABILITIES_DUES, SV_EARLY_ACTIVITY, SV_BUDGET_VARIANCE, SV_INVESTMENTS, SV_OBLIGATIONS, SV_DOCUMENT_STANDING, SV_OPS_FRESHNESS, SV_TRANSACTIONS, SV_RECURRING_FIXED, SV_DEBT_PAYOFF, SV_ACCOUNTS_RECEIVABLE, SV_SAVINGS_GOALS.',
      'Each analyst_* MCP tool is bound to a view FQN. Agents prefer Analyst for totals and refuse rates or due dates the views do not have.',
    ],
    points: [
      'Certified semantic views are the grain agents are allowed to total from.',
      'SV_EARLY_ACTIVITY is pending email plus Monarch pending — not posted spend.',
      'CTRL.SV_OPS_FRESHNESS is what ledger_ops_agent reads before anyone trusts a figure.',
    ],
  },
  {
    id: 'household-tools',
    group: 'Private platform',
    parentId: 'private-platform',
    icon: 'uil-wrench',
    title: 'Cortex tools and EAIs',
    outcome:
      'Agents only get tools I bound. EAI_MONARCH and EAI_GMAIL reach out; procs mutate; Analyst and Search read.',
    modalTitle: 'Tool bindings and External Access Integrations',
    summary: [
      'External Access Integrations: EAI_MONARCH (GraphQL), EAI_GMAIL, plus EAI_MARKET_DATA and EAI_MOONSHOTS_YOUTUBE where needed. AGENT_HOUSEHOLD_FINANCE gets Analyst on each semantic view, four ledger Cortex Search services, proc_metal_spot, and approved write procedures, plus skills on @CTRL.AGENT_SKILLS.',
      'Tool priority is the product: Analyst first, Search second, metal spot, then writes. No general Python REPL around the warehouse.',
    ],
    points: [
      'MCP catalog: agents, analyst_* on semantic views, search_* on Cortex Search services, pipeline and mutate procs on the write server.',
      'R_CORTEX_GATE is the sole SNOWFLAKE.CORTEX_USER holder.',
      'ledger_ops_agent stays read-heavy so a freshness check cannot mutate Monarch.',
    ],
  },
  {
    id: 'camera-tracking',
    group: 'Lab',
    icon: 'uil-webcam',
    title: 'Camera tracking',
    outcome: 'YOLOv8 body/head detection on a Pico W that actually drives the steppers. The clip is the control loop PALMS I1 is built to take.',
    modalTitle: 'Camera tracking: the PALMS vision loop',
    embed: CAMERA_EMBED,
    video: CAMERA_VIDEO,
    summary: [
      'On-prem person tracking for the punching bag: pretrained YOLOv8 body and head detection, a standard webcam, Raspberry Pi Pico W, BLE, and two 12V steppers on one clock. Detection is supposed to become the azimuth command on PALMS I1’s rotating carrier.',
    ],
    github: 'https://github.com/omfawehinmi/portfolio/tree/main/machine_learning/camera_tracking',
  },
  {
    id: 'palms-bag',
    group: 'Lab',
    icon: 'uil-box',
    title: 'Robotic punching bag',
    outcome: 'PALMS I1: three crown cameras to find the opponent, then rotate the arms to strike.',
    modalTitle: 'PALMS I1: vision-guided punching bag',
    videoFile: publicUrl('assets/video/hybrid-punches.mp4'),
    parts: ['camera-tracking'],
    summary: [
      'The computer vision job is to locate the person and command azimuth. Three cameras sit 120 degrees apart on a fixed crown. Detection is supposed to estimate opponent position, then a bounded rotate command turns the shoulder carrier so the arms face the target.',
      'The live vision loop is the YOLOv8 camera-tracking rig: body and head detection, Raspberry Pi Pico W, BLE, two 12V steppers on one clock. The punch clip here is the mechanism in Blender (corrected palms, full reach, camera crown in the same assembly). It is not live firmware or closed-loop tracking.',
    ],
    points: [
      'Intended chain: camera detection → opponent position → bounded azimuth → carrier drive.',
      'Crown cameras stay fixed; only the padded shoulder band and arms orbit, so the vision mount does not spin with the punch.',
      'Safety and stop behavior still need engineering. Vision is how it aims, not how it is declared safe.',
    ],
  },
  {
    id: 'inpainting',
    group: 'Lab',
    icon: 'uil-paint-tool',
    title: 'Inpainting detection',
    outcome: 'WPI coursework: snippets of CNN / ensemble work on CIFAKE and DeepFake Art.',
    modalTitle: 'Inpainting detection: WPI snippet',
    thumb: publicUrl('assets/img/inpainting.png'),
    summary: [
      'A college project from my WPI data-science work, not a product. Snippets of transfer learning, augmentation, frozen CNN heads, and a Hyperopt-tuned ensemble on CIFAKE and the DeepFake Art inpainting split.',
    ],
    github: 'https://github.com/omfawehinmi/portfolio/tree/main/machine_learning/inpainting',
  },
]

export const projectGroups = ['Production', 'Private platform', 'Financial services', 'Lab']

export const highlights = [
  {
    projectId: 'myorglab',
    icon: 'uil-snowflake',
    label: 'Sole-engineer Workday → Snowflake',
    detail: 'Built the Novanta Workday-to-Snowflake integration end to end: planning through production.',
  },
  {
    projectId: 'cortex-self-serve',
    icon: 'uil-processor',
    label: '0 Tier 0/1 queues left',
    detail: 'Workday in Snowflake at Novanta. Cortex replaced all 50+ recurring Tier 0/1 requests.',
  },
  {
    projectId: 'private-platform',
    icon: 'uil-chart-line',
    label: 'HOUSEHOLD_LEDGER + MCP',
    detail: 'Monarch, Gmail, Drive → Snowflake; MCP_HOUSEHOLD_LEDGER write / _GROK read on semantic views and Cortex Search services.',
  },
  {
    projectId: 'semantic-hr',
    icon: 'uil-database',
    label: 'HRIS → Snowflake',
    detail: 'Workday, SuccessFactors, Fieldglass, Eightfold, Visier, One Model.',
  },
]
