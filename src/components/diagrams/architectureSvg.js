// Flat architecture diagrams. Object names are real structural names from the
// platform repos (see README SOURCES). Do not invent names.
// Diagram language: Stripe Docs → How Issuing works → Issuing architecture →
// Issuing balance funding source. Soft white rounded nodes, circular actor
// discs, thin gray connectors, blended transparent canvas. Light-mode only.
/* Diagram-only palette (ERDs + process flows). Page chrome / cards / nav stay on site tokens. */
const C={
  ink:'#0b1d36',          /* node titles — matches site navy */
  sub:'#5b6b7c',          /* secondary labels */
  line:'#c5d0dc',         /* node borders */
  conn:'#7b8fa3',         /* default edges */
  acc:'#0f8ecf',          /* read-path accent (vivid cyan-blue) */
  tint:'#dff3fb',         /* soft accent wash */
  accInk:'#0a6fa3',       /* accent ink / markers */
  dark:'#0b1d36',
  dot:'transparent',
  panel:'transparent',
  soft:'#e8eef4',         /* lane rules */
  node:'#ffffff',
  disc:'#eef6fb'          /* actor discs */
};
const Z={
  data:{f:'#ffffff',s:'#7eb8d4',k:'#0a6fa3',n:'Snowflake data and catalogs',tint:'#d7f0fb'},
  agent:{f:'#ffffff',s:'#a78bda',k:'#5b3fa6',n:'MCP and agents',tint:'#ebe4fb'},
  sec:{f:'#ffffff',s:'#d4a15c',k:'#8a4b12',n:'Roles and access',tint:'#fcebd8'},
  src:{f:'#ffffff',s:'#5fbf9a',k:'#0f6b4c',n:'Sources and scheduling',tint:'#d9f5ea'}
};
let COMPACT=false;
/** Tile density: shrink fonts/icons/spacing so diagrams fill the block without empty gaps. Lead tiles stay slightly larger. */
let DENSITY=1;
const setCompact=(on)=>{COMPACT=!!on};
const setDensity=(d)=>{DENSITY=d==null?1:d};
const F="Poppins, sans-serif",MONO="ui-monospace, SFMono-Regular, Menlo, monospace";
const esc=s=>String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;');
let mt;const meas=(t,sz,w,it,mono)=>{if(!mt){const m=document.createElementNS('http://www.w3.org/2000/svg','svg');m.setAttribute('style','position:absolute;left:-9999px;top:0;width:10px;height:10px');m.setAttribute('text-rendering','geometricPrecision');mt=document.createElementNS('http://www.w3.org/2000/svg','text');m.appendChild(mt);document.body.appendChild(m)}
 mt.setAttribute('font-family',mono?MONO:F);mt.setAttribute('font-size',sz);mt.setAttribute('font-weight',w);mt.setAttribute('font-style',it?'italic':'normal');mt.textContent=t;return mt.getBBox().width};
const wrapW=(t,maxW0,sz,w,it,mono)=>{const maxW=maxW0*0.9;const ws=String(t).split(' '),o=[];let l='';for(const x of ws){const c=(l?l+' ':'')+x;if(meas(c,sz,w,it,mono)>maxW&&l){o.push(l);l=x}else l=c}if(l)o.push(l);return o};
const fs=(n)=>Math.max(6.8,(COMPACT?Math.max(7.2,n-1.15):n)*DENSITY);
const T=(x,y,t,o={})=>`<text x="${x}" y="${y}" text-anchor="${o.a||'middle'}" dominant-baseline="middle" font-family="${o.mono?MONO:F}" font-size="${o.sz||fs(10)}" font-weight="${o.w||400}"${o.it?' font-style="italic"':''}${o.ls?` letter-spacing="${o.ls}"`:''} fill="${o.c||C.ink}">${esc(t)}</text>`;
const lab=(x,y,t,c=C.accInk,a='start')=>T(x,y,t.toUpperCase(),{sz:fs(8.2),w:500,ls:.55,c,a,mono:1});
let ID='a';
const defs=()=>`<defs><filter id="${ID}sh" x="-40%" y="-40%" width="180%" height="200%"><feDropShadow dx="0" dy="1" stdDeviation="1.5" flood-color="#101828" flood-opacity="0.06"/></filter>
 <filter id="${ID}shH" x="-40%" y="-40%" width="180%" height="200%"><feDropShadow dx="0" dy="2" stdDeviation="2.2" flood-color="#101828" flood-opacity="0.10"/></filter>
 <marker id="${ID}a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M1.5,1.5 L8.5,5 L1.5,8.5 Z" fill="${C.conn}"/></marker>
 <marker id="${ID}c" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M1.5,1.5 L8.5,5 L1.5,8.5 Z" fill="${C.accInk}"/></marker>
 <marker id="${ID}w" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M1.5,1.5 L8.5,5 L1.5,8.5 Z" fill="${C.ink}"/></marker></defs>`;
// kinds: default grey, 'c' accent (read and data path), 'w' write path (ink, dashed), 'd' dashed grey
const arrow=(d,k)=>{const col=k==='c'?C.accInk:k==='w'?C.ink:C.conn;return `<path class="edge" d="${d}" fill="none" stroke="${col}" stroke-width="${COMPACT?1.1:1.35}"${k==='w'||k==='d'?' stroke-dasharray="4 3.5"':''} stroke-linecap="round" stroke-linejoin="round" marker-end="url(#${ID}${k==='c'?'c':k==='w'?'w':'a'})"/>`};
// Transparent canvas — card color shows through; no bordered diagram panel.
const panel=(W,H)=>`<rect x="0" y="0" width="${W}" height="${H}" fill="${C.panel}" stroke="none"/>`;
const group=(x,y,w,h,t,o={})=>{const z=o.z?Z[o.z]:null;return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${COMPACT?10:14}" fill="${o.fill||'none'}" stroke="${z?z.s:o.grey?C.line:C.line}" stroke-width="1.15" stroke-dasharray="${o.solid?'0':'3.5 3.5'}"/>`+(t?lab(x+10,y+12,t,z?z.k:o.grey?C.sub:C.accInk):'')};
const legend=(x,y,keys,maxX=1e9)=>{let g='',xx=x;keys.forEach(k=>{const z=Z[k];const iw=15+meas(z.n,fs(8.2),400)/0.9;if(xx>x&&xx+iw>maxX){xx=x;y+=13}g+=`<rect x="${xx}" y="${y-4}" width="9" height="9" rx="2.5" fill="${z.tint||C.tint}" stroke="${z.s}"/>`+T(xx+13,y,z.n,{sz:fs(8.2),c:C.sub,a:'start'});xx+=13+meas(z.n,fs(8.2),400)+14});return g};
const rxBox=()=>COMPACT?10:14;
const outline={
 box:(x,y,w,h)=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rxBox()}"/>`,
 pill:(x,y,w,h)=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${COMPACT?Math.min(10,h/2):h/2}"/>`,
 doc:(x,y,w,h)=>`<path d="M${x+6},${y} H${x+w-11} L${x+w},${y+11} V${y+h-6} Q${x+w},${y+h} ${x+w-6},${y+h} H${x+6} Q${x},${y+h} ${x},${y+h-6} V${y+6} Q${x},${y} ${x+6},${y} Z"/><path d="M${x+w-11},${y} V${y+11} H${x+w}" fill="none"/>`,
 dia:(x,y,w,h)=>`<path d="M${x+w/2},${y} L${x+w},${y+h/2} L${x+w/2},${y+h} L${x},${y+h/2} Z"/>`,
 cyl:(x,y,w,h)=>{const ry=COMPACT?5:6;return `<path d="M${x},${y+ry} a${w/2},${ry} 0 0 1 ${w},0 v${h-2*ry} a${w/2},${ry} 0 0 1 -${w},0 z"/><path d="M${x},${y+ry} a${w/2},${ry} 0 0 0 ${w},0" fill="none"/>`},
 disc:(x,y,w,h)=>{const r=Math.min(w,h)/2;return `<circle cx="${x+w/2}" cy="${y+h/2}" r="${r}"/>`}};
const insetFor=(shape)=>{const base={box:[12,8],pill:[12,5],doc:[10,6],dia:[0,0],cyl:[10,8],disc:[6,6]};const c={box:[8,4],pill:[8,3],doc:[7,3.5],dia:[0,0],cyl:[7,5],disc:[4,4]};const pick=COMPACT||DENSITY<0.95?c:base;return pick[shape]};
const block=(shape,x,y,w,h,o={})=>{const z=o.z?Z[o.z]:o.e?Z.data:null;const fill=shape==='disc'?(C.disc):(z?(z.tint||C.node):C.node);const stroke=z?z.s:C.line;const hero=o.hero||o.lift||shape==='disc'||!z;return `<g class="nd-face" fill="${fill}" stroke="${stroke}" stroke-width="${DENSITY<0.95?1:1.15}"${o.dash?' stroke-dasharray="3.5 3.5"':''}${hero?` filter="url(#${ID}sh)"`:''}>${outline[shape](x,y,w,h)}</g>`};
// node: soft Issuing-style block; interactive .nd group for hover/click enlarge.
const node=(shape,x,y,w,lines,o={})=>{const [px,py]=insetFor(shape);const maxW=w-2*px;const L=[];
 const lhMul=DENSITY<0.95?1.05:(COMPACT?1.1:1.25);
 lines.forEach(l=>{const sz=fs(l.sz||10),ww=l.w||500;wrapW(l.t,maxW,sz,ww,l.it,l.mono).forEach(s=>L.push({...l,t:s,sz,lh:sz*lhMul}))});
 const th=L.reduce((a,l)=>a+l.lh,0);const minH=o.h?Math.round(o.h*Math.min(1,DENSITY+0.08)):0;const h=Math.max(minH,th+2*py+(shape==='cyl'?(COMPACT?2:4):0));let ty=y+(h-th)/2+(shape==='cyl'?2:0),g='';
 L.forEach(l=>{g+=T(x+w/2,ty+l.lh/2,l.t,{sz:l.sz,w:l.w||500,c:l.c||(l.sub?C.sub:C.ink),it:l.it,mono:l.mono});ty+=l.lh});
 const nid=o.id||`${shape}-${Math.round(x)}-${Math.round(y)}`;
 return {g:`<g class="nd" tabindex="0" role="button" data-node="${esc(nid)}" data-b="${x},${y},${w},${h},${px},${py}" data-shape="${shape}">${block(shape,x,y,w,h,o)}${g}</g>`,h,x,y,w}};
const n2=(t,s,sz=10,code,z)=>[{t,w:600,sz:fs(sz)}].concat(s?[{t:s,sz:fs(8.2),sub:1}]:[]).concat(code?[{t:code,sz:fs(8),mono:1,c:z?Z[z].k:C.accInk}]:[]);
/** Circular actor/system disc with monoline glyph (Issuing-style). */
const discNode=(cx,cy,r,label,glyph='●',o={})=>{
  const rr=Math.round(r*Math.min(1,DENSITY+0.05));
  const x=cx-rr,y=cy-rr,w=rr*2,h=rr*2;
  const face=`<g class="nd-face" fill="${C.disc}" stroke="${C.line}" stroke-width="1.1" filter="url(#${ID}sh)"><circle cx="${cx}" cy="${cy}" r="${rr}"/></g>`;
  const icon=T(cx,cy+0.5,glyph,{sz:fs(o.glyphSz||12),w:600,c:C.accInk});
  const labY=cy+rr+Math.round(11*DENSITY);
  const lbl=label?T(cx,labY,label,{sz:fs(8.4),w:500,c:C.ink}):'';
  const nid=o.id||`disc-${Math.round(cx)}-${Math.round(cy)}`;
  return {g:`<g class="nd nd-disc" tabindex="0" role="button" data-node="${esc(nid)}" data-b="${x},${y},${w},${h+ (label?16:0)},0,0" data-shape="disc">${face}${icon}${lbl}</g>`,h:h+(label?16:0),x,y,w}};
const badge=(cx,cy,n)=>{const r=(COMPACT?6.5:8)*Math.min(1,DENSITY+0.05);return `<g class="nd" tabindex="0" role="button" data-node="badge-${n}" data-b="${cx-r},${cy-r},${r*2},${r*2},0,0" data-shape="badge"><circle cx="${cx}" cy="${cy}" r="${r}" fill="${C.ink}" filter="url(#${ID}sh)"/>`+T(cx,cy+0.4,String(n),{sz:fs(8),w:600,c:'#fff'})+`</g>`};
const tag=(cx,cy,t,o={})=>{const tw=meas(t,fs(8),500,0,1)/0.9+14;const h=(COMPACT?14:16)*Math.min(1,DENSITY+0.05);return `<g class="nd nd-tag" data-b="${cx-tw/2},${cy-h/2},${tw},${h},4,0" data-shape="pill"><rect x="${cx-tw/2}" y="${cy-h/2}" width="${tw}" height="${h}" rx="${h/2}" fill="${C.node}" stroke="${C.line}" filter="url(#${ID}sh)"/>`+T(cx,cy+0.4,t,{sz:fs(8),w:500,mono:1,c:C.sub})+`</g>`};
const svg=(W,H,title,body)=>`<svg class="bx-arch" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMin meet" text-rendering="geometricPrecision" role="img" aria-label="${esc(title)}" xmlns="http://www.w3.org/2000/svg">${defs()}<title>${esc(title)}</title>${panel(W,H)}${body}</svg>`;
const ns=(n,L)=>L.find(x=>x.label===n);
/** compact: denser lanes; density from inst — tile denser than open; lead tile milder. */
const begin=(compact,inst="",opts={})=>{
  setCompact(compact);
  const s=String(inst||'');
  const isTile=s==='tile'||s==='t'||s.includes('tile');
  if(!isTile){setDensity(1);return}
  setDensity(opts.lead?0.9:0.78);
};

// 1. Lead: zone overview, left to right. Data and read path in accent; the write path is ink and dashed.
export function dLead(D,tall, inst=""){
  begin(false,inst,{lead:1});ID=(tall?'lT':'lW')+inst;const N=D.lead,K=D.code;let o='',a='';
 if(!tall){const W=740,H=316;
  [[12,'Sources'],[104,'Scheduler'],[244,'Snowflake'],[412,'Catalogs'],[550,'MCP servers'],[666,'Clients']].forEach(([x,t])=>o+=lab(x,20,t,C.sub));
  ['Files','APIs','Documents'].forEach((t,i)=>o+=node('doc',12,52+i*70,74,n2(t),{h:40,z:'src'}).g);
  const af=n2(N.airflow.t,N.airflow.s,10,K.airflow,'src');const ah=node('box',104,0,112,af,{h:66}).h;o+=node('box',104,Math.round(141-ah/2),112,af,{h:66,z:'src'}).g;
  o+=group(236,32,152,250,'',{z:'data'});
  [['Landing schema',46,K.raw],['Conformed schema',120,K.core],['Analytics schema',194,K.analytics]].forEach(([t,y,k])=>o+=node('cyl',252,y,120,n2(t,'',10,k,'data'),{h:52,z:'data',hero:k===K.core}).g);
  o+=node('box',412,90,118,n2('Semantic views','',10,K.svOps,'data'),{h:44,z:'data'}).g+node('box',412,166,118,n2('Cortex Search services','',10,K.cs,'data'),{h:48,z:'data'}).g;
  o+=node('box',550,88,104,[{t:'MCP',sz:8.6,mono:1,c:Z.agent.k}].concat(n2('Read MCP server')),{h:50,z:'agent',hero:1}).g+node('box',550,166,104,[{t:'MCP',sz:8.6,mono:1,c:Z.agent.k}].concat(n2('Write MCP server')),{h:50,z:'agent',hero:1}).g;
  o+=node('box',666,90,64,n2('Read clients'),{h:44}).g+node('box',666,168,64,n2('Write client'),{h:44}).g;
  [72,142,212].forEach(y=>a+=arrow(`M86,${y} H94 Q98,${y} 98,${y<141?y+4:y-4} V${y<141?137:145} Q98,141 102,141`,'c'));
  a+=arrow('M216,141 H226 Q230,141 230,137 V76 Q230,72 234,72 H250','c')+arrow('M312,98 V118','c')+arrow('M312,172 V192','c');
  a+=arrow('M372,220 H390 Q394,220 394,216 V116 Q394,112 398,112 H410','c')+arrow('M372,146 H382 Q386,146 386,150 V186 Q386,190 390,190 H410','c');
  a+=arrow('M530,112 H548','c')+arrow('M530,192 H534 Q538,192 538,188 V117 Q538,113 542,113 H548','c')+arrow('M654,112 H664','c');
  a+=arrow('M666,190 H656','w');
  a+=arrow('M602,216 V262 Q602,266 598,266 H406 Q402,266 402,262 V164 Q402,160 398,160 H374','w');
  o+=tag(497,266,'write path',{light:1})+legend(12,300,['src','data','agent']);
  return svg(W,H,N.title,a+o)}
 const W=330;let y=10;const row=(label,h,f)=>{o+=lab(12,y+8,label,C.sub);f(y+18);y+=h+18+18};
 row('Sources',40,yy=>['Files','APIs','Documents'].forEach((t,i)=>o+=node('doc',12+i*104,yy,96,n2(t),{h:40,z:'src'}).g));
 a+=arrow(`M165,${y-16} V${y-2}`,'c');
 row('Scheduler',50,yy=>o+=node('box',50,yy,230,n2(N.airflow.t,N.airflow.s,10,K.airflow,'src'),{h:50,z:'src'}).g);
 a+=arrow(`M165,${y-16} V${y-2}`,'c');const sy=y;
 o+=group(6,y+12,318,82,'',{z:'data'});
 row('Snowflake',58,yy=>[['Landing schema',K.raw],['Conformed schema',K.core],['Analytics schema',K.analytics]].forEach(([t,k],i)=>o+=node('cyl',14+i*102,yy+4,96,n2(t,'',9.6,k,'data'),{h:52,z:'data',hero:i==1}).g));
 a+=arrow(`M250,${y-10} V${y-2}`,'c');
 row('Catalogs',42,yy=>{o+=node('box',12,yy,148,n2('Semantic views','',10,K.svOps,'data'),{h:42,z:'data'}).g+node('box',170,yy,148,n2('Cortex Search services','',10,K.cs,'data'),{h:42,z:'data'}).g});
 a+=arrow(`M86,${y-16} V${y-2}`,'c');const my=y;
 row('MCP servers',46,yy=>{o+=node('box',12,yy,148,n2('Read MCP server','',10,K.readSrv,'agent'),{h:46,z:'agent',hero:1}).g+node('box',170,yy,148,n2('Write MCP server','',10,K.writeSrv,'agent'),{h:46,z:'agent',hero:1}).g});
 a+=arrow(`M86,${y-16} V${y-2}`,'c')+arrow(`M244,${y-2} V${y-16}`,'w');
 row('Clients',40,yy=>{o+=node('box',12,yy,148,n2('Read clients'),{h:40}).g+node('box',170,yy,148,n2('Write client'),{h:40}).g});
 a+=arrow(`M318,${my+41} H322 Q326,${my+41} 326,${my+37} V${sy+104} Q326,${sy+100} 322,${sy+100} H200 Q196,${sy+100} 196,${sy+96} V${sy+82}`,'w');
 o+=legend(12,y+2,['src','data','agent'],W-12);
 return svg(W,y+26,N.title,a+o)};

// 2. MCP: three servers with agents + summarized tool families (not only the R/W split).
export function dMcp(D, inst=""){
  begin(false,inst);ID='m'+inst;const K=D.code;const M=D.mcpTools||{};const W=560;let o='',a='';
  const servers=M.servers||[
    {t:K.readSrv||'MCP_HOUSEHOLD_LEDGER_GROK',s:'Read · Grok desks',families:'agents · analyst_* · search_* · docs',z:'agent'},
    {t:K.writeSrv||'MCP_HOUSEHOLD_LEDGER',s:'Write · Cursor / Eta',families:'read families + mutate / pipeline procs',z:'agent'},
    {t:'MCP_MOONSHOTS',s:'YouTube · Cynthia',families:'catalog · search · sync / embed',z:'src'},
  ];
  o+=lab(14,16,'MCP servers')+lab(200,16,'Agents')+lab(380,16,'Tool families');
  servers.forEach((s,i)=>{
    const y=36+i*118;
    o+=node('box',14,y,170,n2(s.t,s.s,9.2),{h:52,z:s.z||'agent',hero:i<2,lift:1}).g;
    const agents=(M.agents&&M.agents[i])||(i===0?['household_finance_agent','ledger_ops_agent']:i===1?['same agents + write procs','Cortex AGENT_* behind']:['Cynthia desk','catalog tools']);
    o+=node('box',200,y,160,[{t:agents[0],sz:9,w:600},{t:agents[1]||'',sz:8.4,sub:1}],{h:52,z:'agent'}).g;
    o+=node('box',380,y,166,[{t:s.families,sz:8.8,c:C.sub}],{h:52,dash:i===0,lift:1}).g;
    a+=arrow(`M184,${y+26} H198`,'c')+arrow(`M360,${y+26} H378`,'c');
  });
  o+=node('box',14,392,532,[{t:M.note||'Read server never holds write tools. Families summarized — not a private catalog dump.',sz:9,c:C.sub}],{h:36,dash:1}).g;
  o+=legend(14,444,['agent','src','data']);
  return svg(W,468,M.title||'MCP servers, agents, and tool families',a+o)};

// 2b. Grokbot: which agent attaches to which MCP, and what that access is for.
export function dGrokbot(D, inst=""){
  begin(false,inst);ID='gb'+inst;const G=D.grokbot||{};const W=560;let o='',a='';
  const rows=G.rows||[
    {agent:'Grok finance desks',mcp:'MCP_HOUSEHOLD_LEDGER_GROK',use:'spend · cashflow · docs · ops freshness',z:'agent'},
    {agent:'Cursor / Eta',mcp:'MCP_HOUSEHOLD_LEDGER',use:'mutate · ingest · rebuild analytics',z:'agent'},
    {agent:'Cynthia',mcp:'MCP_MOONSHOTS',use:'YouTube catalog · transcripts · sync',z:'src'},
  ];
  o+=lab(14,16,'Agent / desk')+lab(200,16,'MCP server')+lab(390,16,'Uses that access');
  rows.forEach((r,i)=>{
    const y=36+i*78;
    o+=node('box',14,y,170,n2(r.agent,'',9.4),{h:48,z:r.z||'agent',lift:1}).g;
    o+=node('box',200,y,172,n2(r.mcp,'',9),{h:48,z:'agent',hero:1}).g;
    o+=node('box',390,y,156,[{t:r.use,sz:8.8,c:C.sub}],{h:48}).g;
    a+=arrow(`M184,${y+24} H198`,'c')+arrow(`M372,${y+24} H388`,'c');
  });
  o+=node('box',14,286,532,[{t:G.note||'Cortex AGENT_HOUSEHOLD_FINANCE / _RO / AGENT_LEDGER_OPS sit on SV_* and CS_* behind these MCP catalogs.',sz:9,c:C.sub}],{h:40,dash:1}).g;
  o+=legend(14,342,['agent','src']);
  return svg(W,366,G.title||'Grokbot architecture — agents per MCP',a+o)};

// 3. Data domains: Issuing-style system map (no crow's-foot). Soft cards + thin arrows.
export function dErd(D, inst=""){
  begin(false,inst);ID='e'+inst;const E=D.erd.entities;const W=620;let o='',a='';
 D.erd.areas.forEach((t,i)=>{const cols=3;const x=16+(i%cols)*198,y=16+Math.floor(i/cols)*40;
  o+=node('pill',x,y,186,[{t,sz:9.4,w:500,c:i===4?Z.sec.k:(i===0?Z.data.k:C.sub)}],{h:32,z:i===4?'sec':(i===0?'data':null),lift:1,id:`area-${i}`}).g});
 const ent=(x,y,w,e,cat)=>{
  const rows=cat?[]:e.rows;
  const nameSz=e.name.length>20?9.6:10.6;
  const nameLines=wrapW(e.name,w-28,nameSz,600,0,1);
  const ml=cat?[]:wrapW(e.meaning,w-28,9.4,400,1);
  const headH=20+nameLines.length*14+10,rowH=18,noteGap=12,noteLh=14,padX=14;
  const h=headH+(rows.length?rows.length*rowH+10:0)+(cat?0:noteGap+Math.max(1,ml.length)*noteLh+8);
  let g=`<g class="nd" tabindex="0" role="button" data-node="${esc(e.name)}" data-b="${x},${y},${w},${h},10,6" data-shape="box">`;
  g+=`<g class="nd-face" filter="url(#${ID}sh)"><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="14" fill="${cat?(Z.data.tint||C.tint):C.node}" stroke="${C.line}" stroke-width="1.15"/></g>`;
  g+=`<path d="M${x+10},${y+headH} H${x+w-10}" stroke="${C.line}"/>`;
  g+=T(x+padX,y+14,e.layer,{sz:8.6,w:500,ls:.4,c:cat?Z.data.k:C.sub,a:'start',mono:1});
  nameLines.forEach((l,i)=>g+=T(x+padX,y+28+i*14,l,{sz:nameSz,w:600,a:'start',mono:1}));
  rows.forEach((r,i)=>{const yy=y+headH+14+i*rowH;if(i)g+=`<path d="M${x+padX},${yy-rowH/2} H${x+w-padX}" stroke="${C.soft}"/>`;
   g+=T(x+padX,yy,r[0],{sz:9.4,a:'start',mono:1})+((r[1]||r[2])?T(x+w-padX,yy,r[1]||r[2],{sz:8.6,w:500,c:r[1]?C.accInk:C.sub,a:'end',mono:1}):'')});
  ml.forEach((l,i)=>g+=T(x+padX,y+h-14-(ml.length-1-i)*noteLh,l,{sz:9.4,it:1,c:C.sub,a:'start'}));
  return {g:g+'</g>',h}};
 const w=186,xs=[16,218,420],y=116;const hs=[];[0,1,2].forEach(i=>{const r=ent(xs[i],y,w,E[i],0);o+=r.g;hs.push(r.h)});
 // Simple directional links — no crow's-foot UML ends.
 a+=arrow(`M${xs[0]+w},${y+56} H${xs[1]}`,'c')+arrow(`M${xs[1]+w},${y+72} H${xs[2]}`,'c');
 const y2=y+Math.max(...hs)+56;const s1=ent(xs[1],y2,w,E[4],1),s2=ent(xs[2],y2,w,E[3],1);o+=s1.g+s2.g;
 a+=arrow(`M${xs[1]+w/2},${y+hs[1]} V${y2-2}`,'d')+arrow(`M${xs[2]+w/2},${y+hs[2]} V${y2-2}`,'d');
 o+=tag(xs[1]+w/2,y+hs[1]+24,'indexes')+tag(xs[2]+w/2,y+hs[2]+24,'reads');
 const ky=y2;
 o+=T(16,ky+10,'Relationships',{sz:10,w:600,a:'start'});
 o+=`<line class="edge" x1="16" y1="${ky+28}" x2="48" y2="${ky+28}" stroke="${C.conn}" stroke-width="1.35" stroke-linecap="round"/>`+T(56,ky+28,'primary path',{sz:9.2,c:C.sub,a:'start'});
 o+=`<line class="edge" x1="16" y1="${ky+48}" x2="48" y2="${ky+48}" stroke="${C.conn}" stroke-width="1.35" stroke-dasharray="4 3.5" stroke-linecap="round"/>`+T(56,ky+48,'derived or read',{sz:9.2,c:C.sub,a:'start'});
 o+=T(16,ky+70,'Real object names, no row data',{sz:9.2,c:C.sub,a:'start'});
 const notePad=14,noteW=W-32;const nl=wrapW(D.erd.searchNote,noteW-2*notePad,9.6,400,1);
 const noteH=notePad*2+Math.max(1,nl.length)*14;const noteY=Math.max(y2+s1.h,ky+92)+20;
 o+=`<g class="nd" tabindex="0" role="button" data-node="search-note" data-b="16,${noteY},${noteW},${noteH},10,6" data-shape="box"><g class="nd-face" filter="url(#${ID}sh)"><rect x="16" y="${noteY}" width="${noteW}" height="${noteH}" rx="14" fill="${C.node}" stroke="${C.line}" stroke-width="1.15" stroke-dasharray="4 3.5"/></g>`;
 nl.forEach((l,i)=>o+=T(16+notePad,noteY+notePad+8+i*14,l,{sz:9.6,it:1,c:C.sub,a:'start'}));
 o+='</g>';
 return svg(W,noteY+noteH+16,D.erd.title,a+o)};

// 4. Pipelines: BPMN-style lanes with hairline separators and mono lane labels, events, gateway, tasks.
export function dPipes(D, inst=""){
  begin(true,inst);ID='p'+inst;const Pp=D.pipes,K=D.code;const W=400,H=300;let o='',a='';
 const lanes=[['Sources',0,60],['Airflow on office PC',60,120],['Snowflake',180,96]];
 lanes.forEach(([t,y],i)=>{if(i)o+=`<line x1="8" y1="${y}" x2="${W-8}" y2="${y}" stroke="${C.soft}"/>`;o+=lab(12,y+10,t,i<2?Z.src.k:Z.data.k)});
  o+=discNode(22,32,11,'', '▶',{id:'start',glyphSz:9}).g;
  const t1=node('box',44,18,120,n2('Monarch · Gmail · Drive · YT','',8.8),{h:26,z:'src'});o+=t1.g;
  o+=T(176,24,'EAI_MONARCH · EAI_GMAIL',{sz:fs(7.8),mono:1,c:Z.src.k,a:'start'})+T(176,35,'pipeline_load_window · drive_poll',{sz:fs(7.8),c:C.sub,a:'start'})+T(176,46,'MyChart FHIR → HOUSEHOLD_HEALTH',{sz:fs(7.8),c:Z.data.k,a:'start'});
 o+=T(10,78,Pp.airflow.s,{sz:fs(7.8),it:1,c:C.sub,a:'start'});
 o+=discNode(100,122,14,'','◇',{id:'gateway',glyphSz:11}).g;
 const ex=node('box',210,92,168,n2('Extract and land','',8.8,K.land,'src'),{h:28,z:'src'});o+=ex.g;
 const oc=node('box',210,132,168,n2(Pp.ocr.t,Pp.ocr.s,8.6,K.ocr,'src'),{h:28,z:'src'});o+=oc.g;
 o+=tag(154,108,'Monarch / Gmail',{light:1})+tag(154,158,'Drive docs',{light:1});
 const ld=node('box',232,210,144,n2('RAW and CORE','',8.8,K.raw+', '+K.core,'data'),{h:34,z:'data'});o+=ld.g;
 const db=node('box',140,216,78,n2(Pp.dbt.t,'',8.6),{h:26,z:'data'});o+=db.g;
 const ts=node('box',78,216,48,n2('Tests','',8.6),{h:26});o+=ts.g;
 o+=discNode(34,229,10,'', '●',{id:'end',glyphSz:8}).g+T(12,250,'ANALYTICS,',{sz:fs(7.8),c:C.sub,a:'start'})+T(12,261,'MCP servers',{sz:fs(7.8),c:C.sub,a:'start'})+T(140,268,K.dbt,{sz:fs(7.8),mono:1,c:Z.data.k,a:'start'})+legend(14,286,['src','data']);
 a+=arrow('M29,32 H42','c')+arrow(`M104,${18+t1.h} V106`,'c')+arrow('M114,122 H120 Q124,122 124,118 V108 Q124,104 128,104 H208','c')+arrow('M100,136 V150 Q100,154 104,154 H208','c');
 a+=arrow('M378,106 H384 Q388,106 388,110 V225 Q388,229 384,229 H378','c')+arrow(`M294,${132+oc.h} V208`,'c')+arrow('M232,227 H220','c')+arrow('M140,229 H128','c')+arrow('M78,229 H44','c');
 return svg(W,H,Pp.title,a+o)};

// 5. Security: nested access zones as dashed outlines, accent only on the MCP split.
export function dSec(D, inst=""){
  begin(true,inst);ID='s'+inst;const S=D.security,K=D.code;const W=380,H=300;let o='';
 o+=group(8,6,364,274,'Roles',{z:'sec'})+group(18,58,344,214,'MCP split',{z:'agent'})+group(28,128,324,136,'Cortex gate',{z:'sec'})+group(38,182,304,72,'Catalogs',{z:'data'});
 S.roles.forEach((t,i)=>o+=node('pill',16+i*116,24,108,[{t,sz:8.6,w:500}],{h:22,z:'sec'}).g);
 o+=node('box',34,74,148,n2(S.write,'',8.6,K.writeSrv,'agent'),{h:30,z:'agent'}).g+node('box',194,74,148,n2(S.read.t,S.read.s,8.6,K.readSrv,'agent'),{h:30,z:'agent'}).g;
 o+=`<path d="M62,148 l12,4 v8 q0,8 -12,12 q-12,-4 -12,-12 v-8 z" fill="${C.node}" stroke="${Z.sec.k}" stroke-width="1.1"/>`;
 o+=node('box',96,142,220,n2(S.gate.t,S.gate.s,8.6),{h:26,z:'sec'}).g;
 o+=node('cyl',54,198,120,n2(S.cat[0],'',8.4,K.svDoc,'data'),{h:30,z:'data'}).g+node('cyl',196,198,120,n2(S.cat[1],'',8.4,K.cs,'data'),{h:30,z:'data'}).g;
 o+=legend(14,288,['sec','agent','data']);
 return svg(W,H,S.title,o)};

// 6. Stack: flat layer bands, the agent interface in accent.
export function dStack(D, inst=""){
  begin(true,inst);ID='k'+inst;const L=D.stack.layers;const W=360;let o='';const x=10,w=340,h=24;
 const zs=['agent','src','data','src',null];L.forEach((l,i)=>{const y=8+i*(h+4);const z=zs[i]?Z[zs[i]]:null;const one=l.nodes.length==1&&l.nodes[0].s;
  const labW=meas(l.label.toUpperCase(),fs(7.4),500,0,1)+16;
  const leftMax=w-labW-20;
  o+=`<g class="nd" data-b="${x},${y},${w},${h},7,2" data-shape="box"><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rxBox()}" fill="${z?z.f:C.node}" stroke="${z?z.s:C.line}"/>`;
  if(one){
    const tLines=wrapW(l.nodes[0].t,leftMax,fs(8.4),600);
    const sLines=l.nodes[0].s?wrapW(l.nodes[0].s,leftMax,fs(7.4),400):[];
    o+=T(x+10,y+(sLines.length?9:h/2),tLines[0],{sz:fs(8.4),w:600,a:'start'});
    if(sLines.length)o+=T(x+10,y+18,sLines[0]+(sLines.length>1?'…':''),{sz:fs(7.4),c:C.sub,a:'start'});
  }else{
    const joined=wrapW(l.nodes.map(n=>n.t).join(' and '),leftMax,fs(8.4),600);
    o+=T(x+10,y+h/2,joined[0]+(joined.length>1?'…':''),{sz:fs(8.4),w:600,a:'start'});
  }
  o+=T(x+w-10,y+h/2,l.label.toUpperCase(),{sz:fs(7.4),w:500,ls:.35,mono:1,c:z?z.k:C.sub,a:'end'})+`</g>`});
 return svg(W,8+L.length*(h+4)+4,D.stack.title,o)};

// 7. Snowflake features: capability matrix, hairline cells, accent icons.
export function dFeat(D, inst=""){
  begin(true,inst);ID='f'+inst;const R=D.features.rows;const W=448;let o='';const rh=28,hx=78,cw=178,pad=6;const FH=pad*2+R.length*(rh+4)-4;
 const icon=(k,x,y,s)=>{switch(k){
  case 'hex':return `<rect x="${x}" y="${y+1}" width="11" height="10" rx="2" fill="none" stroke="${s}" stroke-width="1.1"/><path d="M${x+2.5},${y+6} h6" stroke="${s}" stroke-width="1.1"/>`;
  case 'grid':return `<rect x="${x-1}" y="${y}" width="13" height="12" rx="2" fill="none" stroke="${s}" stroke-width="1.1"/><path d="M${x-1},${y+4} h13 M${x+4},${y} v12" stroke="${s}" stroke-width="1"/>`;
  case 'search':return `<circle cx="${x+5}" cy="${y+5}" r="4.2" fill="none" stroke="${s}" stroke-width="1.1"/><path d="M${x+8.5},${y+8.5} l3.5,3.5" stroke="${s}" stroke-width="1.3"/>`;
  case 'layers':return `<path d="M${x+6},${y} l7,3.5 l-7,3.5 l-7,-3.5 z M${x-1},${y+6} l7,3.5 l7,-3.5 M${x-1},${y+9} l7,3.5 l7,-3.5" fill="none" stroke="${s}" stroke-width="1"/>`;
  case 'clock':return `<circle cx="${x+6}" cy="${y+6}" r="5.5" fill="none" stroke="${s}" stroke-width="1.1"/><path d="M${x+6},${y+2.5} v3.5 l2.5,1.5" stroke="${s}" stroke-width="1.1" fill="none"/>`;
  default:return `<path d="M${x},${y+6} h5 l3.5,-5 h3.5 M${x+5},${y+6} l3.5,5 h3.5" fill="none" stroke="${s}" stroke-width="1.1"/>`}};
 R.forEach((r,i)=>{const y=pad+i*(rh+4);o+=`<g class="nd" data-b="${pad},${y},${hx-12},${rh},3,2" data-shape="box">`+T(pad+5,y+rh/2,r.label.toUpperCase(),{sz:fs(7.6),w:500,ls:.35,mono:1,c:C.sub,a:'start'})+'</g>';
  const span=r.cells.length==1;r.cells.forEach((c,j)=>{const x=hx+j*(cw+5),w=span?cw*2+5:cw;
   const z=r.z?Z[r.z]:null;o+=`<g class="nd" data-b="${x+20},${y},${w-20},${rh},3,2" data-shape="box"><rect x="${x}" y="${y}" width="${w}" height="${rh}" rx="${rxBox()}" fill="${z?z.f:C.node}" stroke="${z?z.s:C.line}"/>`+icon(c.i,x+8,y+8,z?z.k:C.sub);
   o+=T(x+26,c.s?y+9:y+rh/2,c.t,{sz:fs(8.2),w:600,a:'start'})+(c.s?T(x+26,y+19,c.s,{sz:fs(7.4),c:c.m&&z?z.k:C.sub,a:'start',mono:c.m}):'')+'</g>'})});
 o+=legend(pad+6,FH+8,['agent','data']);
 return svg(W,FH+18,D.features.title,o)};

export function dWorkflow(D, inst=""){
  begin(true,inst);ID='wf'+inst;const Wf=D.workflow;const W=420;let o='',a='';
  const step=28,nh=22;
  o+=T(12,14,'Merge ≠ apply ≠ deploy',{sz:fs(10),w:600,a:'start'})+T(12,27,"Michael's go gates apply, unpause, grants",{sz:fs(7.8),c:C.sub,a:'start'});
  Wf.steps.forEach((s,i)=>{const y=38+i*step;const z=i<2?'src':i===2?null:i===3?'data':'src';
   o+=badge(20,y+nh/2,i+1)+node('box',36,y,368,n2(s.t,s.s,8.6),{h:nh,z:z||undefined}).g;
   if(i<Wf.steps.length-1)a+=arrow(`M20,${y+nh} V${y+step-2}`,'c')});
  o+=legend(12,38+Wf.steps.length*step+2,['src','data']);
  return svg(W,38+Wf.steps.length*step+18,Wf.title,a+o)};

export function dDatabase(D, dbId, inst=""){
  // Stripe-light object cards: taller rows so layer/name + note never clip.
  begin(false,inst);ID='dbx'+inst+(dbId||'');const db=D.databases?.[dbId]||D.databases?.ledger;if(!db)return '';
  const W=460;let o='';
  o+=T(14,16,db.name,{sz:12,w:600,a:'start'})+T(14,32,db.status+' · '+db.oneLine,{sz:8.6,c:C.sub,a:'start'});
  const cols=Math.min(4,Math.max(2,db.schemas.length));
  db.schemas.forEach((s,i)=>{const x=14+(i%cols)*110,y=48+Math.floor(i/cols)*32;
   o+=node('pill',x,y,104,[{t:s,sz:8.6,w:500,mono:1}],{h:26,z:'data',lift:1}).g});
  const schemaRows=Math.ceil(db.schemas.length/cols);let y=48+schemaRows*32+16;
  o+=lab(14,y,'Objects / path',C.sub);y+=12;
  const rowH=44;
  (db.objects||[]).forEach((obj,i)=>{const yy=y+i*rowH;
   o+=node('box',14,yy,432,[{t:obj.layer+' · '+obj.name,sz:9.4,w:600,mono:1},{t:obj.note,sz:8.4,sub:1}],{h:38,z:i===0?'data':null,lift:1}).g});
  o+=legend(14,y+(db.objects||[]).length*rowH+10,db.id==='health'?['src','data']:['data']);
  return svg(W,y+(db.objects||[]).length*rowH+28,db.name+' schemas',o)};

export function dAirflow(D, inst=""){
  begin(true,inst);ID='af'+inst;const A=D.airflow;const W=420;let o='';
  o+=T(12,14,A.headline,{sz:fs(11),w:600,a:'start'})+T(12,27,'Office-PC Airflow pin · git ≠ unpaused',{sz:fs(7.8),c:C.sub,a:'start'});
  A.counts.forEach((c,i)=>{const x=12+i*134;o+=node('box',x,38,126,[{t:c.n,sz:fs(12),w:600,c:Z.src.k},{t:c.label,sz:fs(7.6),sub:1}],{h:30,z:'src'}).g});
  const rows=[
    ['pipeline_load_window','07:15 ET INGEST_WH window'],
    ['moonshots_yt / eyl','YT · unpaused on 11-07 pin'],
    ['monarch_load · ledger_email_*','called by window'],
    ['drive_poll / doc_extract','Drive → OCR → CORE'],
    ['ledger_dbt_daily → hr → aw','06:00 ET dbt chain'],
    ['dbt_daily (Cosmos)','created paused'],
    ['nightly_app_backup · pc_usage','PC routines · no SF'],
    ['epic_fhir_load_manual','manual · awaits source data'],
    ['platform_smoke · *_dev (3)','manual / not production'],
  ];
  rows.forEach((r,i)=>{const y=84+i*17;o+=T(16,y,r[0],{sz:fs(8),mono:1,c:Z.src.k,a:'start'})+T(244,y,r[1],{sz:fs(7.8),c:C.sub,a:'start'})});
  o+=legend(12,84+rows.length*17+6,['src'],W-20);
  return svg(W,84+rows.length*17+20,A.title,o)};

export function dDbt(D, inst=""){
  begin(true,inst);
  ID='db'+inst;const L=D.dbt.layers;const W=380;let o='',a='';
  const step=34,nh=24;
  L.forEach((l,i)=>{const y=12+i*step;const z=i<4?'data':'src';
   o+=node(i<2?'cyl':'box',36,y,308,n2(l.t,l.s,8.6),{h:nh,z}).g;
   if(i<L.length-1)a+=arrow(`M190,${y+nh} V${y+step-2}`,'c')});
  o+=legend(12,12+L.length*step+2,['data','src']);
  return svg(W,12+L.length*step+18,D.dbt.title,a+o)};

export function dRearch(inst=""){
  begin(true,inst);ID='r'+inst;let o='',a='';['Terraform','dbt','Airflow','Snowflake'].forEach((t,i)=>{const xx=12+i*88;o+=node(i==3?'cyl':'box',xx,16,72,n2(t,'',9),{h:32,e:i==3}).g;if(i<3)a+=arrow(`M${xx+72},32 H${xx+86}`,'c')});
 return svg(370,64,'Re-architecture preview: Terraform, dbt and Airflow managing a Snowflake platform',a+o)};


export function dMychart(D,tall, inst=""){
  begin(true,inst);
  ID=(tall?'hT':'hW')+inst;
  const M=D.mychart;
  const compact=inst==='tile';
  const steps=[
    {t:M.nodes[0],z:'src',shape:'doc'},
    {t:M.nodes[1],z:'sec',shape:'box'},
    {t:M.nodes[2],z:'sec',shape:'box',hero:1},
    {t:M.nodes[3],z:'src',shape:'box'},
    {t:M.nodes[4],z:'data',shape:'box'},
    {t:M.nodes[5],z:'data',shape:'box'},
    {t:M.nodes[6],z:'data',shape:'cyl',hero:1}
  ];
  const place=(s,x,y,w,hMin)=>node(s.shape,x,y,w,n2(s.t,'',9.6),{h:hMin||56,z:s.z,hero:s.hero});
  let o='',a='';

  if(!tall){
    const W=920,pad=18,gap=16,nw=Math.floor((W-pad*2-gap*6)/7);
    const nh=Math.max(...steps.map(s=>place(s,0,0,nw,60).h));
    const nodeY=54;
    const xs=steps.map((_,i)=>pad+i*(nw+gap));
    const clusters=[{z:'sec',i:[1,2]},{z:'data',i:[4,5,6]}];
    clusters.forEach(c=>{
      const x0=xs[c.i[0]]-8, x1=xs[c.i[c.i.length-1]]+nw+8;
      o+=group(x0,30,x1-x0,nh+36,'',{z:c.z});
      o+=lab(x0+8,42,Z[c.z].n,Z[c.z].k);
    });
    steps.forEach((s,i)=>{o+=place(s,xs[i],nodeY,nw,nh).g+badge(xs[i],nodeY,i+1)});
    for(let i=0;i<6;i++){const mid=nodeY+nh/2;a+=arrow(`M${xs[i]+nw},${mid} H${xs[i+1]}`,'c')}
    const stW=meas(M.status,8.6,500,0,1)/0.88+14;
    o+=tag(W-16-stW/2,18,M.status);
    const footY=30+nh+36+20;
    const chW=meas(M.chip,8.6,500,0,1)/0.88+14;
    o+=tag(16+chW/2,footY,M.chip,{light:1});
    o+=legend(16+chW+18,footY,['src','sec','data'],W-16);
    return svg(W,footY+22,M.caption,a+o);
  }

  if(compact){
    const W=330,pad=12,gap=8,r1=4,r2=3;
    const w1=Math.floor((W-pad*2-gap*(r1-1))/r1);
    const w2=Math.floor((W-pad*2-gap*(r2-1))/r2);
    const h1=Math.max(...steps.slice(0,r1).map(s=>place(s,0,0,w1,52).h));
    const h2=Math.max(...steps.slice(r1).map(s=>place(s,0,0,w2,52).h));
    const y1=46,y2=y1+h1+32;
    const stW=meas(M.status,8.6,500,0,1)/0.88+14;
    o+=tag(Math.min(W-10-stW/2,W/2),16,M.status);
    const x1=steps.slice(0,r1).map((_,i)=>pad+i*(w1+gap));
    const row2Inner=r2*w2+(r2-1)*gap;
    const x2start=Math.round((W-row2Inner)/2);
    const x2=steps.slice(r1).map((_,i)=>x2start+i*(w2+gap));
    o+=group(x2[0]-8,y2-18,row2Inner+16,h2+26,'',{z:'data'});
    o+=lab(x2[0]+8,y2-8,Z.data.n,Z.data.k);
    steps.slice(0,r1).forEach((s,i)=>{o+=place(s,x1[i],y1,w1,h1).g+badge(x1[i],y1,i+1)});
    steps.slice(r1).forEach((s,i)=>{o+=place(s,x2[i],y2,w2,h2).g+badge(x2[i],y2,r1+i+1)});
    for(let i=0;i<r1-1;i++){const mid=y1+h1/2;a+=arrow(`M${x1[i]+w1},${mid} H${x1[i+1]}`,'c')}
    const turnX=x1[r1-1]+w1/2, midY=y1+h1+10, destX=x2[0]+w2/2;
    a+=`<path d="M${turnX},${y1+h1} V${midY} H${destX}" fill="none" stroke="${C.accInk}" stroke-width="1.1" stroke-linejoin="round"/>`;
    a+=arrow(`M${destX},${midY} V${y2}`,'c');
    for(let i=0;i<r2-1;i++){const mid=y2+h2/2;a+=arrow(`M${x2[i]+w2},${mid} H${x2[i+1]}`,'c')}
    const footY=y2+h2+24;
    const chW=meas(M.chip,8.6,500,0,1)/0.88+14;
    o+=tag(pad+chW/2,footY,M.chip,{light:1});
    o+=legend(pad,footY+18,['src','sec','data'],W-pad);
    return svg(W,footY+36,M.caption,a+o);
  }

  const W=330,pad=16,nw=W-pad*2;
  const stW=meas(M.status,8.6,500,0,1)/0.88+14;
  o+=tag(Math.min(W/2, W-10-stW/2),18,M.status);
  let y=44;
  const placed=[];
  const clusters=[{z:'sec',i:[1,2]},{z:'data',i:[4,5,6]}];
  const heights=steps.map(s=>place(s,0,0,nw,48).h);
  clusters.forEach(c=>{
    const y0=(()=>{let yy=44;for(let i=0;i<c.i[0];i++)yy+=heights[i]+18;return yy})();
    let gh=0;c.i.forEach((i,k)=>{gh+=heights[i]+(k?18:0)});
    o+=group(pad-8,y0-16,nw+16,gh+24,'',{z:c.z});
    o+=lab(pad,y0-6,Z[c.z].n,Z[c.z].k);
  });
  steps.forEach((s,i)=>{
    const r=place(s,pad,y,nw,heights[i]);
    placed.push({y,h:r.h});
    o+=r.g+badge(pad,y,i+1);
    y+=r.h+18;
  });
  for(let i=0;i<6;i++){a+=arrow(`M${pad+nw/2},${placed[i].y+placed[i].h} V${placed[i+1].y}`,'c')}
  const chW=meas(M.chip,8.6,500,0,1)/0.88+14;
  o+=tag(pad+chW/2,y+6,M.chip,{light:1});
  o+=legend(pad,y+26,['src','sec','data'],W-pad);
  return svg(W,y+48,M.caption,a+o);
}

/** High-level map of drill areas — clickable labels are rendered in React; this is the visual map. */
export function dDrillMap(inst=""){
  begin(true,inst);ID='map'+inst;const W=520;let o='',a='';
  const areas=[
    {t:'Source feeds',s:'Monarch · Gmail · Drive · YT · MyChart',x:14,y:36,z:'src'},
    {t:'Databases',s:'5 prod DBs + *_DEV',x:270,y:36,z:'data'},
    {t:'Warehouses & cost',s:'LEDGER · INGEST · ADHOC · caps',x:14,y:130,z:'sec'},
    {t:'MCP & tools',s:'Read · write · Moonshots',x:270,y:130,z:'agent'},
    {t:'Workflows',s:'Ship · ingest window · dbt chain',x:14,y:224,z:'src'},
    {t:'E2E processing',s:'Land → transform → test → serve',x:270,y:224,z:'data'},
  ];
  o+=T(14,16,'Platform map — pick an area to drill in',{sz:fs(11),w:600,a:'start'});
  areas.forEach((ar,i)=>{
    o+=node('box',ar.x,ar.y,236,[{t:ar.t,sz:10,w:600},{t:ar.s,sz:8.2,sub:1}],{h:72,z:ar.z,hero:1,id:`map-${i}`}).g;
  });
  a+=arrow('M250,72 H268','c')+arrow('M132,108 V128','c')+arrow('M388,108 V128','c');
  o+=legend(14,310,['src','data','sec','agent'],W-20);
  return svg(W,336,'Household platform map',a+o)};

/** Per-feed / E2E process flow from a DrillItem.steps list. */
export function dFeedFlow(item, inst=""){
  begin(false,inst);ID='ff'+inst+(item?.id||'');
  const steps=item?.steps||[{t:item?.label||'Feed',s:item?.blurb}];
  const W=Math.min(640,48+steps.length*110);let o='',a='';
  o+=T(14,16,item?.label||'Feed path',{sz:fs(11),w:600,a:'start'});
  o+=T(14,30,item?.blurb||'',{sz:fs(8.2),c:C.sub,a:'start'});
  const nw=96,gap=14,y=52;
  steps.forEach((s,i)=>{
    const x=14+i*(nw+gap);
    const z=i===0?'src':i>=steps.length-2?'data':(i===1?'sec':null);
    o+=badge(x,y,i+1)+node('box',x,y+14,nw,n2(s.t,s.s,8.4),{h:56,z:z||'data',lift:1}).g;
    if(i<steps.length-1)a+=arrow(`M${x+nw},${y+42} H${x+nw+gap}`,'c');
  });
  o+=legend(14,y+86,['src','sec','data'],W-20);
  return svg(W,y+110,item?.label||'Feed',a+o)};

/** Warehouse / cost diagram for a DrillItem. */
export function dWarehouseFlow(item, inst=""){
  begin(false,inst);ID='wh'+inst+(item?.id||'');
  const W=520;let o='',a='';
  o+=T(14,16,item?.label||'Warehouse',{sz:fs(12),w:600,a:'start'});
  o+=T(14,32,item?.blurb||'',{sz:fs(8.4),c:C.sub,a:'start'});
  const steps=item?.steps||[
    {t:'Batch wake-ups',s:'STD-WH-BATCH'},
    {t:'Dedicated WH',s:'prod vs ADHOC'},
    {t:'auto_suspend',s:'30–60s'},
    {t:'Monitors / budgets',s:'notify · suspend'},
  ];
  steps.forEach((s,i)=>{
    const y=52+i*48;
    o+=badge(20,y+16,i+1)+node('box',40,y,440,n2(s.t,s.s,9),{h:40,z:i%2?'sec':'data',lift:1}).g;
    if(i<steps.length-1)a+=arrow(`M20,${y+40} V${y+48}`,'c');
  });
  const foot=52+steps.length*48+8;
  o+=legend(14,foot,['data','sec'],W-20);
  return svg(W,foot+22,item?.label||'Warehouse',a+o)};

/** Single MCP / tool family diagram. */
export function dToolFlow(item, inst=""){
  begin(false,inst);ID='tf'+inst+(item?.id||'');
  const W=480;let o='',a='';
  o+=T(14,16,item?.label||'MCP',{sz:fs(11),w:600,a:'start'});
  o+=T(14,30,item?.blurb||'',{sz:fs(8.2),c:C.sub,a:'start'});
  o+=node('box',14,48,200,n2('MCP server',item?.label,9),{h:56,z:'agent',hero:1}).g;
  o+=node('box',250,48,200,n2('Agents / clients', (item?.meta||[]).find(m=>m.k==='Agents'||m.k==='Clients'||m.k==='Desk')?.v||'Attached desks',8.6),{h:56,z:'agent'}).g;
  o+=node('box',14,128,436,n2('Tool families', item?.detail?.slice(0,120)||'analyst_* · search_* · procs',8.4),{h:52,z:'data',lift:1}).g;
  a+=arrow('M214,76 H248','c')+arrow('M114,104 V126','c');
  o+=legend(14,198,['agent','data'],W-20);
  return svg(W,220,item?.label||'Tool',a+o)};
