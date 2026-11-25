// Richard's signed-off copy, word for word, from household-tracker
// files/brand/portfolio-2026-11-22/COPY.md via files/design/review/portfolio/src/copy.js.
// Diagram labels and mono sublabels are the checked names from that pack. Do not invent names or claims.

// COPY DECK: Richard (Brand), 2026-11-22, household-tracker files/brand/portfolio-2026-11-22/COPY.md. Word for word.
// Swap copy here only. Diagram labels (DIAG) are design labels drawn from the cited household-data-architecture files.
export const COPY = {
  section:{kicker:"Projects",title:"Work that shows the platform",intro:"The lead project is the Snowflake data platform and the MCP servers that sit on it. The tiles under it are how that platform is shaped. Other shipped work follows in a shorter list."},
  lead:{title:"Snowflake and MCP architecture",subhead:"Snowflake data architecture and MCP",
    oneLine:"A production Snowflake platform I run: layers as schemas, Airflow designed as the scheduler, and separate read and write MCP servers so agents use governed catalogs instead of an open warehouse login.",
    tags:["Snowflake","Airflow","dbt","Terraform","Cortex","MCP","semantic views"],
    intro:"This is the data platform I operate. Snowflake is the system of record. Airflow is designed to schedule production work. dbt and procedures are designed to transform landed data. MCP servers are the interface agents are allowed to use. Semantic views and Cortex Search are designed to be the catalogs those agents read. The work is private and production-shaped, and it is the same pattern behind MyOrgLab customer deployments: built in the account, on governed data, not as a multi-tenant copy."},
  tiles:[
    {id:"mcp",n:"2",title:"Read and write MCP servers",svg:"mcp",
     body:"Agents reach the platform only through MCP servers, never through a general warehouse login. The read server is designed to answer from semantic views and Cortex Search, and it carries no write tools. The write server is a separate server with the same Analyst, Search and agent tools as the read server, plus the approved procedures that write to the conformed schema. Splitting them means a client that only needs answers gets the read server alone and never holds a way to change data.",
     caption:"Two request paths side by side: a read path through the catalogs, and a write path that adds approved procedures."},
    {id:"domains",n:"3",title:"Data domains",svg:"erd",
     body:"The platform is split by product area, not by one database per layer. Each product area holds its own schemas for landing, conformed data, control, and analytics. A row in the landing schema is one landed page, file, or API payload. A row in the conformed schema is one business thing. A row in analytics is one reporting grain. Some landing procedures write the landing and conformed schemas in the same call, so the conformed layer is not only a later rebuild of landing data. Dev is a separate environment with synthetic fixtures, not a copy of production data.",
     caption:"Product-area databases hold layers as schemas. Each layer has a defined row meaning."},
    {id:"security",n:"5",title:"Security",svg:"security",
     body:"Access is separated by role and by what a client is allowed to call. Pipeline work, interactive use, and agents do not share one open path into the warehouse. MCP is split into a write server and a read server, so a read client can use catalogs without write tools. A dedicated role holds the Cortex user privilege, and answers are supposed to come from certified views and search services, not from ad hoc access. Public copy stops at that architecture. It does not describe account identifiers, hostnames, secrets, or how to reach a system.",
     caption:"Role separation and a read and write MCP split, so agents use only the catalogs they are allowed to see."},
    {id:"pipes",n:"4",title:"Pipelines and feeds",svg:"pipes",
     body:"Production scheduling is designed to run on self-hosted Airflow, not on Snowflake tasks. Sources are designed to land, then transform in dbt or procedures, then test. Document text on this path is designed to run OCR locally for cost: a PDF text layer first, then local OCR when that is not enough, so document bytes are not a Cortex parse call. A planned S3 and Snowpipe intake was dropped and is not described as running. Intake is designed to stay on the Airflow path.",
     caption:"Sources are designed to move through Airflow into Snowflake landing and conformed layers, then into analytics, then to MCP. Snowflake tasks are not the designed scheduler."},
    {id:"stack",n:"6",title:"Stack",svg:"stack",
     body:"Snowflake holds the data and the catalogs. Airflow is designed to schedule production jobs. dbt is designed to build tested models. Terraform manages infrastructure in code. GitHub Actions is the CI path. Cortex is designed to be used for agents, Analyst, and Search where it is gated on. Document text is designed to run OCR locally for cost. The same family of tools (Terraform, dbt, Airflow, CI/CD) is the pattern I also use in customer deployments through MyOrgLab.",
     caption:"Airflow, Snowflake, dbt, Terraform, and CI, with MCP as the agent interface."},
    {id:"features",n:"7",title:"Snowflake features",svg:"features",
     body:"Layers are schemas inside product-area databases, with a defined meaning for each row. Semantic views are the Analyst catalogs. Cortex Search services are the search catalogs. MCP servers in the account are designed to expose those catalogs as a read path or a write path, not as a general login. Cortex use is designed to be gated by role. Warehouse work is designed to be batched so a warehouse wakes once per run, and interactive and agent work can sit on a warehouse separate from production loads.",
     caption:"Semantic views and Cortex Search are the catalogs. Separate read and write MCP servers sit in front of them."}],
  more:[
    {prev:"hr-intelligence",title:"HR Intelligence",line:"Workday and SuccessFactors landed as governed Snowflake semantics, the same grain agents and dashboards read. At Novanta I was the sole engineer on the Workday-to-Snowflake integration and put Cortex on that grain so recurring Tier 0 and Tier 1 reporting queues went away. At NetApp I land SuccessFactors, Fieldglass, Eightfold, and Visier into Snowflake and One Model.",tags:["Workday","SuccessFactors","Fieldglass","Eightfold","Visier","One Model","Snowflake","Cortex"]},
    {prev:"myorglab",title:"MyOrgLab",line:"Snowflake-native practice: in-account agents, semantic views, and MCP bridges used in customer deployments.",tags:["Snowflake","Cortex","semantic views","MCP","Streamlit","Power BI"]},
    {prev:"rearch",title:"Snowflake platform re-architecture",line:"Lift and shift of a Snowflake data platform into a Terraform, dbt, and Airflow managed architecture.",tags:["Terraform","dbt","Airflow","Snowflake"]},
    {prev:"sof-portal",title:"SOF client portal",line:"The portal clients still log into: document vault, messaging, and live status, with MFA, on AWS.",tags:["React","Flask","AWS","MFA"]},
    {prev:"palms",title:"Robotic punching bag",line:"PALMS I1: three crown cameras find the opponent, then the arms rotate to strike. The punch clip is the mechanism in Blender, not live closed-loop firmware.",tags:["YOLOv8","Raspberry Pi Pico W","BLE","steppers","Blender"]},
    {prev:"camera",title:"Camera tracking",line:"The live vision loop the bag is built to take: YOLOv8 body and head detection on a Pico W that drives the steppers.",tags:["YOLOv8","Raspberry Pi Pico W","BLE"]},
    {prev:"inpainting",title:"Inpainting detection",line:"WPI coursework, not a product: CNN and ensemble snippets on CIFAKE and DeepFake Art.",tags:["CNN","Hyperopt","CIFAKE","DeepFake Art"]}]
};
// DIAGRAM LABELS: design labels. Mono sublabels are real structural names (database, schema, table, DAG, server) from the
// platform repos; clients, roles and warehouses stay generic. Sources per diagram in README SOURCES.
export const DIAG = {
  lead:{title:"Lead diagram: request path from agents through separate read and write MCP servers to governed catalogs and schemas",
   layers:[
    {label:"Clients",nodes:[{t:"Read clients",s:"agents and approved clients"},{t:"Write client",s:"approved write path"}]},
    {label:"MCP servers",nodes:[{t:"Read MCP server",s:"catalogs, no write tools",e:1},{t:"Write MCP server",s:"read tools plus approved procedures",e:1}]},
    {label:"Cortex",nodes:[{t:"Cortex agents"},{t:"Analyst",s:"text to SQL"},{t:"Search"}]},
    {label:"Catalogs",nodes:[{t:"Semantic views",e:1},{t:"Cortex Search services",e:1}]},
    {label:"Snowflake",nodes:[{t:"Analytics schema",s:"one reporting grain"},{t:"Conformed schema",s:"one business thing"}]}],
   edges:[[0,0,1,0],[0,1,1,1],[1,0,2,0],[1,0,2,1],[1,0,2,2],[2,0,3,0],[2,0,3,1],[2,1,3,0],[2,2,3,1],[3,0,4,0],[3,1,4,1],[1,1,4,1]]},
  pipes:{title:"Platform flow and pipeline shape",layers:[
    {label:"Sources",nodes:[{t:"Files"},{t:"APIs"},{t:"Documents"}]},
    {label:"Extract and land",nodes:[{t:"Self-hosted Airflow",s:"designed as the production scheduler",e:1},{t:"Local text extraction",s:"designed to use the PDF text layer, then local OCR"},{t:"S3 and Snowpipe",s:"dropped, not running",d:1}]},
    {label:"Snowflake",nodes:[{t:"Landing schema"},{t:"Conformed schema"}]},
    {label:"Transform",nodes:[{t:"dbt or procedures",e:1}]},
    {label:"Test",nodes:[{t:"Tests"}]},
    {label:"Serve",nodes:[{t:"Analytics"},{t:"MCP servers",e:1}]}],
   edges:[[0,0,1,0],[0,1,1,0],[0,2,1,1],[1,0,2,0],[1,0,2,1],[1,1,2,0],[2,0,3,0],[2,1,3,0],[3,0,4,0],[4,0,5,0],[4,0,5,1]]},
  security:{title:"Access split by role and by server",layers:[
    {label:"Roles",nodes:[{t:"Pipeline role"},{t:"Interactive role"},{t:"Agent client role"}]},
    {label:"MCP split",nodes:[{t:"Write MCP server",e:1},{t:"Read MCP server",s:"no write tools",e:1}]},
    {label:"Cortex gate",nodes:[{t:"Dedicated role",s:"holds Cortex user privilege"}]},
    {label:"Catalogs",nodes:[{t:"Certified views"},{t:"Search services"}]}]},
  stack:{title:"Stack diagram",layers:[
    {label:"Agent interface",nodes:[{t:"MCP servers",e:1}]},
    {label:"Transform",nodes:[{t:"dbt",s:"models and tests"}]},
    {label:"Warehouse and catalogs",nodes:[{t:"Snowflake",s:"data, semantic views, Cortex Search",e:1}]},
    {label:"Scheduler",nodes:[{t:"Airflow",s:"self-hosted"}]},
    {label:"Infrastructure and CI",nodes:[{t:"Terraform"},{t:"GitHub Actions"}]}]},
  features:{title:"Catalogs and MCP split",layers:[
    {label:"MCP servers in the account",nodes:[{t:"Read server",e:1},{t:"Write server",e:1}]},
    {label:"Catalogs",nodes:[{t:"Semantic views",s:"Analyst catalogs"},{t:"Cortex Search",s:"search catalogs"}]},
    {label:"Schemas",nodes:[{t:"Layers as schemas",s:"defined row meaning"}]},
    {label:"Warehouses",nodes:[{t:"Batched loads",s:"designed to wake once per run"},{t:"Interactive and agents",s:"separate warehouse"}]}]},
  erd:{title:"Schema ERD for the documents path: landing, conformed, analytics, plus catalogs",
   areas:["Area A: operational data","Area B: media catalog","Area C: workforce","Control"],
   schemas:["landing","conformed","control","analytics"],
   entities:[
    {layer:"LANDING: RAW",name:"DOCUMENTS_RAW",rows:[["LOAD_ID","PK"],["DOCUMENT_ID","PK"],["STAGE_PATH"],["SOURCE_CHANNEL"]],meaning:"One landed page, file, or API payload"},
    {layer:"CONFORMED: CORE",name:"DOCUMENTS",rows:[["DOCUMENT_ID","PK"],["RAW_LOAD_ID","","ref"],["DOCUMENT_KIND","FK"],["TEXT_EXTRACT"]],meaning:"One business thing"},
    {layer:"ANALYTICS",name:"FACT_DOCUMENT_STANDING",rows:[["AS_OF_DATE","PK"],["DOCUMENT_ID","PK"],["STATUS"],["DAYS_TO_EXPIRY"]],meaning:"One reporting grain"},
    {layer:"SEMANTIC VIEW",name:"SV_DOCUMENT_STANDING",rows:[["facts"],["dimensions"]],meaning:"Analyst catalog"},
    {layer:"CORTEX SEARCH: PAUSED",name:"CS_DOCUMENTS",rows:[["SEARCH_TEXT"],["attributes"]],meaning:"Search catalog"}]}
};

// REDO additions: labels reused from the DIAG layers above, arranged for the new diagram types.
DIAG.lead.airflow={t:"Self-hosted Airflow",s:"designed as the production scheduler"};
DIAG.erd.searchNote="Search service is designed to index conformed document text. Object names are real. No row data is shown.";
// Mono sublabels: neutral real names (schema, table, catalog, DAG) or neutral labels where the real name carries a
// personal or product identifier. Server names stay as the node labels. README SOURCES lists the file each one comes from.
DIAG.code={raw:"RAW",core:"CORE",analytics:"ANALYTICS",readSrv:"",writeSrv:"",
  svOps:"CTRL.SV_OPS_FRESHNESS",svDoc:"SV_DOCUMENT_STANDING",cs:"CS_DOCUMENTS (paused)",airflow:"drive_poll, doc_extract (paused)",
  land:"video sync DAG, email poll DAG",ocr:"doc_extract (paused)",dbt:"dbt DAGs: dbt_daily, daily dbt DAG"};
Object.assign(DIAG.pipes,{airflow:DIAG.pipes.layers[1].nodes[0],ocr:DIAG.pipes.layers[1].nodes[1],dropped:DIAG.pipes.layers[1].nodes[2],dbt:DIAG.pipes.layers[3].nodes[0]});
Object.assign(DIAG.security,{roles:DIAG.security.layers[0].nodes.map(n=>n.t),write:"Write MCP server",read:{t:"Read MCP server",s:"no write tools"},
  gate:{t:"Dedicated role",s:"holds Cortex user privilege"},cat:["Certified views","Search services"]});
DIAG.features.title="Snowflake capability matrix";
DIAG.features.rows=[
  {label:"Access",z:"agent",cells:[{i:"hex",t:"Read server",s:"no write tools"},{i:"hex",t:"Write server",s:"read tools plus procedures"}]},
  {label:"Catalogs",z:"data",cells:[{i:"grid",t:"Semantic views",s:"Analyst catalogs"},{i:"search",t:"Cortex Search",s:"search catalogs"}]},
  {label:"Data",z:"data",cells:[{i:"layers",t:"Layers as schemas",s:"defined row meaning"}]},
  {label:"Warehouses",cells:[{i:"clock",t:"Batched loads",s:"designed to wake once per run"},{i:"split",t:"Interactive and agents",s:"separate warehouse"}]}];

COPY.mychart = {
  id: "mychart",
  kicker: "Healthcare data ingestion",
  title: "MyChart to Snowflake ingestion",
  summary: "A SMART on FHIR ingestion path from a MyChart patient portal into Snowflake raw tables. Built and deployed, awaiting first patient authorization.",
  open: "Designed and deployed a SMART on FHIR R4 ingestion path from an Epic MyChart patient portal into Snowflake. The patient authorizes access through OAuth, and tokens are kept in a token vault on AWS Lambda, DynamoDB and SSM Parameter Store. A scheduled Apache Airflow load is designed to pull FHIR R4 resources onto a Snowflake internal stage, and stored procedures load them into VARIANT raw tables. Infrastructure is defined in Terraform. The path is pending its first patient authorization, so no health data has been loaded yet.",
  caption: "Ingestion path from patient authorization to Snowflake raw tables. Built, awaiting first load. No patient data is shown.",
  status: "Built, awaiting first load",
  chip: "Infrastructure in Terraform",
  nodes: [
    "MyChart (FHIR R4)",
    "Patient authorization (OAuth)",
    "Token vault (Lambda, DynamoDB, SSM)",
    "Scheduled load (Airflow)",
    "Internal stage",
    "Load procedures",
    "Raw tables (VARIANT)"
  ]
};
COPY.industries = {
  line: "The same pattern also covers Healthcare, Logistics, Legal, Finance and Operations, and Public Sector.",
  link: "See all industries",
  href: "https://www.myorglab.com/intelligence"
};
DIAG.mychart = COPY.mychart;
