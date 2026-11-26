// Flat architecture diagrams. Object names are real structural names from the
// platform repos (see README SOURCES). Do not invent names.
// Diagram language: Stripe docs ERD — simple, modern. Diagram canvas blends
// into the bento card (no contrasted panel). Nodes, edges, and labels keep
// enough contrast on that shared surface so nothing disappears into the card.
const lightC={ink:'#0a2540',sub:'#4d5d6d',line:'#9aa8b6',conn:'#5c6b7a',acc:'#29b5e8',tint:'#eaf7fd',accInk:'#0a6688',dark:'#0a2540',dot:'transparent',panel:'transparent',soft:'#d7dee6',node:'#ffffff'};
const darkC={ink:'#f2f8fc',sub:'#b4c7d4',line:'#5a7388',conn:'#9db6c6',acc:'#5cc8ef',tint:'#14344a',accInk:'#8ad8f4',dark:'#071422',dot:'transparent',panel:'transparent',soft:'#2a3d50',node:'#243447'};
const lightZ={data:{f:'#e0f3fb',s:'#1689b5',k:'#0a6688',n:'Snowflake data and catalogs'},agent:{f:'#e8e9f8',s:'#6b68b8',k:'#3f3d8f',n:'MCP and agents'},
 sec:{f:'#f3e8d4',s:'#b07f2e',k:'#7a4a00',n:'Roles and access'},src:{f:'#dcefe4',s:'#3f8a68',k:'#1a5f45',n:'Sources and scheduling'}};
const darkZ={data:{f:'#1a455c',s:'#5cc8ef',k:'#c5ebf8',n:'Snowflake data and catalogs'},agent:{f:'#2e3258',s:'#b4a6e8',k:'#ddd4f8',n:'MCP and agents'},
 sec:{f:'#4a3c22',s:'#d6a24e',k:'#f0d39a',n:'Roles and access'},src:{f:'#1d4536',s:'#6fbf9a',k:'#b6e4cf',n:'Sources and scheduling'}};
const isDark=()=>typeof document!=='undefined'&&document.body.classList.contains('dark-theme');
let C=lightC,Z=lightZ,COMPACT=false;
const syncTheme=()=>{if(isDark()){C=darkC;Z=darkZ}else{C=lightC;Z=lightZ}};
const setCompact=(on)=>{COMPACT=!!on};
syncTheme();
const F="Poppins, sans-serif",MONO="ui-monospace, SFMono-Regular, Menlo, monospace";
const esc=s=>String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;');
let mt;const meas=(t,sz,w,it,mono)=>{if(!mt){const m=document.createElementNS('http://www.w3.org/2000/svg','svg');m.setAttribute('style','position:absolute;left:-9999px;top:0;width:10px;height:10px');m.setAttribute('text-rendering','geometricPrecision');mt=document.createElementNS('http://www.w3.org/2000/svg','text');m.appendChild(mt);document.body.appendChild(m)}
 mt.setAttribute('font-family',mono?MONO:F);mt.setAttribute('font-size',sz);mt.setAttribute('font-weight',w);mt.setAttribute('font-style',it?'italic':'normal');mt.textContent=t;return mt.getBBox().width};
const wrapW=(t,maxW0,sz,w,it,mono)=>{const maxW=maxW0*0.9;const ws=String(t).split(' '),o=[];let l='';for(const x of ws){const c=(l?l+' ':'')+x;if(meas(c,sz,w,it,mono)>maxW&&l){o.push(l);l=x}else l=c}if(l)o.push(l);return o};
const fs=(n)=>COMPACT?Math.max(7.8,n-1):n;
const T=(x,y,t,o={})=>`<text x="${x}" y="${y}" text-anchor="${o.a||'middle'}" dominant-baseline="middle" font-family="${o.mono?MONO:F}" font-size="${o.sz||fs(10)}" font-weight="${o.w||400}"${o.it?' font-style="italic"':''}${o.ls?` letter-spacing="${o.ls}"`:''} fill="${o.c||C.ink}">${esc(t)}</text>`;
const lab=(x,y,t,c=C.accInk,a='start')=>T(x,y,t.toUpperCase(),{sz:fs(8.2),w:500,ls:.55,c,a,mono:1});
let ID='a';
const defs=()=>`<defs><filter id="${ID}sh" x="-25%" y="-25%" width="150%" height="170%"><feDropShadow dx="0" dy="1" stdDeviation="1.2" flood-color="${isDark()?'#000':'#0a2540'}" flood-opacity="${COMPACT?0.1:0.14}"/></filter>
 <marker id="${ID}a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M1,1 L9,5 L1,9" fill="none" stroke="${C.conn}" stroke-width="1.5"/></marker>
 <marker id="${ID}c" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M1,1 L9,5 L1,9" fill="none" stroke="${C.accInk}" stroke-width="1.5"/></marker>
 <marker id="${ID}w" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M1,1 L9,5 L1,9" fill="none" stroke="${C.ink}" stroke-width="1.5"/></marker></defs>`;
// kinds: default grey, 'c' accent (read and data path), 'w' write path (ink, dashed), 'd' dashed grey
const arrow=(d,k)=>{const col=k==='c'?C.accInk:k==='w'?C.ink:C.conn;return `<path d="${d}" fill="none" stroke="${col}" stroke-width="${COMPACT?1.05:1.2}"${k==='w'||k==='d'?' stroke-dasharray="4 3"':''} stroke-linejoin="round" marker-end="url(#${ID}${k==='c'?'c':k==='w'?'w':'a'})"/>`};
// Transparent canvas — card color shows through; no bordered/dotted diagram panel.
const panel=(W,H)=>`<rect x="0" y="0" width="${W}" height="${H}" fill="${C.panel}" stroke="none"/>`;
const group=(x,y,w,h,t,o={})=>{const z=o.z?Z[o.z]:null;return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${COMPACT?6:8}" fill="${o.fill||'none'}" stroke="${z?z.s:o.grey?C.line:C.acc}" stroke-width="1.15" stroke-dasharray="${o.solid?'0':'3 3'}"/>`+(t?lab(x+8,y+10,t,z?z.k:o.grey?C.sub:C.accInk):'')};
// legend: small swatches plus free text, one row; keys are zone ids.
const legend=(x,y,keys,maxX=1e9)=>{let g='',xx=x;keys.forEach(k=>{const z=Z[k];const iw=15+meas(z.n,fs(8.2),400)/0.9;if(xx>x&&xx+iw>maxX){xx=x;y+=13}g+=`<rect x="${xx}" y="${y-4}" width="9" height="9" rx="2" fill="${z.f}" stroke="${z.s}"/>`+T(xx+13,y,z.n,{sz:fs(8.2),c:C.sub,a:'start'});xx+=13+meas(z.n,fs(8.2),400)+14});return g};
const rxBox=()=>COMPACT?3.5:5;
const outline={
 box:(x,y,w,h)=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rxBox()}"/>`,
 pill:(x,y,w,h)=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${COMPACT?Math.min(6,h/2):h/2}"/>`,
 doc:(x,y,w,h)=>`<path d="M${x+5},${y} H${x+w-9} L${x+w},${y+9} V${y+h-5} Q${x+w},${y+h} ${x+w-5},${y+h} H${x+5} Q${x},${y+h} ${x},${y+h-5} V${y+5} Q${x},${y} ${x+5},${y} Z"/><path d="M${x+w-9},${y} V${y+9} H${x+w}" fill="none"/>`,
 dia:(x,y,w,h)=>`<path d="M${x+w/2},${y} L${x+w},${y+h/2} L${x+w/2},${y+h} L${x},${y+h/2} Z"/>`,
 cyl:(x,y,w,h)=>{const ry=COMPACT?4:5;return `<path d="M${x},${y+ry} a${w/2},${ry} 0 0 1 ${w},0 v${h-2*ry} a${w/2},${ry} 0 0 1 -${w},0 z"/><path d="M${x},${y+ry} a${w/2},${ry} 0 0 0 ${w},0" fill="none"/>`}};
const insetFor=(shape)=>{const base={box:[10,6],pill:[10,4],doc:[8,5],dia:[0,0],cyl:[8,7]};const c={box:[8,3.5],pill:[9,2.5],doc:[7,3],dia:[0,0],cyl:[7,5]};return COMPACT?c[shape]:base[shape]};
const block=(shape,x,y,w,h,o={})=>{const z=o.z?Z[o.z]:o.e?Z.data:null;const hero=o.hero||o.lift||!z;return `<g fill="${z?z.f:C.node}" stroke="${z?z.s:C.line}" stroke-width="1.15"${o.dash?' stroke-dasharray="3 3"':''}${hero?` filter="url(#${ID}sh)"`:''}>${outline[shape](x,y,w,h)}</g>`};
// node: flat block plus wrapped lines; height grows to fit. lines: [{t, sz, w, c, it, mono, sub}]
const node=(shape,x,y,w,lines,o={})=>{const [px,py]=insetFor(shape);const maxW=w-2*px;const L=[];
 lines.forEach(l=>{const sz=fs(l.sz||10),ww=l.w||400;wrapW(l.t,maxW,sz,ww,l.it,l.mono).forEach(s=>L.push({...l,t:s,sz,lh:sz*(COMPACT?1.08:1.22)}))});
 const th=L.reduce((a,l)=>a+l.lh,0);const h=Math.max(o.h||0,th+2*py+(shape==='cyl'?(COMPACT?2:4):0));let ty=y+(h-th)/2+(shape==='cyl'?2:0),g='';
 L.forEach(l=>{g+=T(x+w/2,ty+l.lh/2,l.t,{sz:l.sz,w:l.w,c:l.c||(l.sub?C.sub:C.ink),it:l.it,mono:l.mono});ty+=l.lh});
 return {g:`<g class="nd" data-b="${x},${y},${w},${h},${px},${py}" data-shape="${shape}">${block(shape,x,y,w,h,o)}${g}</g>`,h}};
const n2=(t,s,sz=10,code,z)=>[{t,w:600,sz:fs(sz)}].concat(s?[{t:s,sz:fs(8.2),sub:1}]:[]).concat(code?[{t:code,sz:fs(8),mono:1,c:z?Z[z].k:C.accInk}]:[]);
const badge=(cx,cy,n)=>{const r=COMPACT?6.5:8;return `<g class="nd badge" data-b="${cx-r},${cy-r},${r*2},${r*2},0,0" data-shape="badge"><circle cx="${cx}" cy="${cy}" r="${r}" fill="${C.ink}"/>`+T(cx,cy+0.4,String(n),{sz:fs(8),w:600,c:isDark()?'#071422':'#fff'})+`</g>`};
const tag=(cx,cy,t,o={})=>{const tw=meas(t,fs(8),500,0,1)/0.9+12;const h=COMPACT?14:16;return `<g class="nd" data-b="${cx-tw/2},${cy-h/2},${tw},${h},4,0" data-shape="pill"><rect x="${cx-tw/2}" y="${cy-h/2}" width="${tw}" height="${h}" rx="${h/2}" fill="${o.light?C.node:C.dark}" stroke="${o.light?C.line:C.dark}"/>`+T(cx,cy+0.4,t,{sz:fs(8),w:500,mono:1,c:o.light?C.sub:(isDark()?'#071422':'#fff')})+`</g>`};
const svg=(W,H,title,body)=>{syncTheme();return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" text-rendering="geometricPrecision" role="img" aria-label="${esc(title)}" xmlns="http://www.w3.org/2000/svg">${defs()}<title>${esc(title)}</title>${panel(W,H)}${body}</svg>`};
const ns=(n,L)=>L.find(x=>x.label===n);
const begin=(compact)=>{syncTheme();setCompact(compact)};

// 1. Lead: zone overview, left to right. Data and read path in accent; the write path is ink and dashed.
export function dLead(D,tall, inst=""){
  begin(false);ID=(tall?'lT':'lW')+inst;const N=D.lead,K=D.code;let o='',a='';
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
  begin(false);ID='m'+inst;const K=D.code;const M=D.mcpTools||{};const W=560;let o='',a='';
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
  begin(false);ID='gb'+inst;const G=D.grokbot||{};const W=560;let o='',a='';
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

// 3. Data domains: Stripe-docs crow's-foot ERD — larger boxes/notes so text fits.
export function dErd(D, inst=""){
  begin(false);ID='e'+inst;const E=D.erd.entities;const W=600;let o='',a='';
 D.erd.areas.forEach((t,i)=>{const cols=3;const x=14+(i%cols)*192,y=14+Math.floor(i/cols)*36;
  o+=node('pill',x,y,180,[{t,sz:9.2,w:500,c:i===4?Z.sec.k:(i===0?Z.data.k:C.sub)}],{h:30,z:i===4?'sec':(i===0?'data':null),lift:1}).g});
 const ent=(x,y,w,e,cat)=>{
  const rows=cat?[]:e.rows;
  const nameSz=e.name.length>20?9.4:10.4;
  const nameLines=wrapW(e.name,w-24,nameSz,600,0,1);
  const ml=cat?[]:wrapW(e.meaning,w-24,9.2,400,1);
  const headH=18+nameLines.length*13+10,rowH=17,noteGap=10,noteLh=13,padX=12;
  const h=headH+(rows.length?rows.length*rowH+8:0)+(cat?0:noteGap+Math.max(1,ml.length)*noteLh+6);
  let g=`<g class="nd" data-b="${x},${y},${w},${h},8,4" data-shape="box">`;
  g+=`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" fill="${cat?Z.data.f:C.node}" stroke="${cat?Z.data.s:C.line}" filter="url(#${ID}sh)"/>`;
  g+=`<path d="M${x+0.5},${y+headH} H${x+w-0.5}" stroke="${cat?Z.data.s:C.line}"/>`;
  g+=T(x+padX,y+12,e.layer,{sz:8.6,w:500,ls:.4,c:cat?Z.data.k:C.sub,a:'start',mono:1});
  nameLines.forEach((l,i)=>g+=T(x+padX,y+26+i*13,l,{sz:nameSz,w:600,a:'start',mono:1}));
  rows.forEach((r,i)=>{const yy=y+headH+12+i*rowH;if(i)g+=`<path d="M${x+padX},${yy-rowH/2} H${x+w-padX}" stroke="${C.soft}"/>`;
   g+=T(x+padX,yy,r[0],{sz:9.2,a:'start',mono:1})+((r[1]||r[2])?T(x+w-padX,yy,r[1]||r[2],{sz:8.6,w:500,c:r[1]?C.accInk:C.sub,a:'end',mono:1}):'')});
  ml.forEach((l,i)=>g+=T(x+padX,y+h-12-(ml.length-1-i)*noteLh,l,{sz:9.2,it:1,c:C.sub,a:'start'}));
  return {g:g+'</g>',h}};
 const w=180,xs=[14,210,406],y=108;const hs=[];[0,1,2].forEach(i=>{const r=ent(xs[i],y,w,E[i],0);o+=r.g;hs.push(r.h)});
 const cf=(x1,x2,yy)=>{const s=C.conn;let g=`<line x1="${x1}" y1="${yy}" x2="${x2}" y2="${yy}" stroke="${s}" stroke-width="1.15"/>`;
  g+=`<line x1="${x1+5}" y1="${yy-5}" x2="${x1+5}" y2="${yy+5}" stroke="${s}"/><line x1="${x1+9}" y1="${yy-5}" x2="${x1+9}" y2="${yy+5}" stroke="${s}"/>`;
  g+=`<path d="M${x2-8},${yy} L${x2},${yy-5} M${x2-8},${yy} L${x2},${yy+5}" stroke="${s}" fill="none"/><circle cx="${x2-12}" cy="${yy}" r="3.2" fill="${C.node}" stroke="${s}"/>`;return g};
 const mo=(x1,x2,yy)=>{const s=C.conn;return `<line x1="${x1}" y1="${yy}" x2="${x2}" y2="${yy}" stroke="${s}" stroke-width="1.15"/><path d="M${x1+8},${yy} L${x1},${yy-5} M${x1+8},${yy} L${x1},${yy+5}" stroke="${s}" fill="none"/><circle cx="${x1+11}" cy="${yy}" r="3.2" fill="${C.node}" stroke="${s}"/><line x1="${x2-5}" y1="${yy-5}" x2="${x2-5}" y2="${yy+5}" stroke="${s}"/><line x1="${x2-9}" y1="${yy-5}" x2="${x2-9}" y2="${yy+5}" stroke="${s}"/>`};
 a+=mo(xs[0]+w,xs[1],y+52)+cf(xs[1]+w,xs[2],y+68);
 const y2=y+Math.max(...hs)+48;const s1=ent(xs[1],y2,w,E[4],1),s2=ent(xs[2],y2,w,E[3],1);o+=s1.g+s2.g;
 a+=arrow(`M${xs[1]+w/2},${y+hs[1]} V${y2-2}`,'d')+arrow(`M${xs[2]+w/2},${y+hs[2]} V${y2-2}`,'d');
 o+=tag(xs[1]+w/2,y+hs[1]+22,'indexes',{light:1})+tag(xs[2]+w/2,y+hs[2]+22,'reads',{light:1});
 const ky=y2;o+=T(12,ky+6,'Key',{sz:10,w:600,a:'start'});
 o+=`<line x1="12" y1="${ky+24}" x2="40" y2="${ky+24}" stroke="${C.conn}"/><line x1="16" y1="${ky+19}" x2="16" y2="${ky+29}" stroke="${C.conn}"/><line x1="20" y1="${ky+19}" x2="20" y2="${ky+29}" stroke="${C.conn}"/>`+T(48,ky+24,'exactly one',{sz:9.2,c:C.sub,a:'start'});
 o+=`<line x1="12" y1="${ky+44}" x2="40" y2="${ky+44}" stroke="${C.conn}"/><path d="M32,${ky+44} L40,${ky+39} M32,${ky+44} L40,${ky+49}" stroke="${C.conn}" fill="none"/><circle cx="28" cy="${ky+44}" r="3.2" fill="${C.node}" stroke="${C.conn}"/>`+T(48,ky+44,'zero or many',{sz:9.2,c:C.sub,a:'start'});
 o+=`<line x1="12" y1="${ky+64}" x2="40" y2="${ky+64}" stroke="${C.conn}" stroke-dasharray="4 3"/>`+T(48,ky+64,'derived or read',{sz:9.2,c:C.sub,a:'start'});
 o+=T(12,ky+84,'Real object names, no row data',{sz:9.2,c:C.sub,a:'start'});
 const notePad=12,noteW=W-24;const nl=wrapW(D.erd.searchNote,noteW-2*notePad,9.4,400,1);
 const noteH=notePad*2+Math.max(1,nl.length)*13;const noteY=Math.max(y2+s1.h,ky+100)+16;
 o+=`<rect x="12" y="${noteY}" width="${noteW}" height="${noteH}" rx="8" fill="${C.node}" stroke="${C.line}" stroke-dasharray="4 3" filter="url(#${ID}sh)"/>`;
 nl.forEach((l,i)=>o+=T(12+notePad,noteY+notePad+7+i*13,l,{sz:9.4,it:1,c:C.sub,a:'start'}));
 return svg(W,noteY+noteH+14,D.erd.title,a+o)};

// 4. Pipelines: BPMN-style lanes with hairline separators and mono lane labels, events, gateway, tasks.
export function dPipes(D, inst=""){
  begin(true);ID='p'+inst;const Pp=D.pipes,K=D.code;const W=400,H=300;let o='',a='';
 const lanes=[['Sources',0,60],['Airflow on office PC',60,120],['Snowflake',180,96]];
 lanes.forEach(([t,y],i)=>{if(i)o+=`<line x1="1" y1="${y}" x2="${W-1}" y2="${y}" stroke="${C.line}"/>`;o+=lab(10,y+9,t,i<2?Z.src.k:Z.data.k)});
 const ev=(cx,cy,r,sw)=>`<circle cx="${cx}" cy="${cy}" r="${r}" fill="${C.node}" stroke="${C.ink}" stroke-width="${sw}"/>`;
  o+=ev(22,32,7,1.1)+T(22,46,'Start',{sz:fs(7.8),c:C.sub});
  const t1=node('box',44,18,120,n2('Monarch · Gmail · Drive · YT','',8.8),{h:26,z:'src'});o+=t1.g;
  o+=T(176,24,'EAI_MONARCH · EAI_GMAIL',{sz:fs(7.8),mono:1,c:Z.src.k,a:'start'})+T(176,35,'drive_poll · moonshots_*',{sz:fs(7.8),c:C.sub,a:'start'})+T(176,46,'MyChart FHIR planned',{sz:fs(7.8),c:Z.sec.k,a:'start'});
 o+=T(10,78,Pp.airflow.s,{sz:fs(7.8),it:1,c:C.sub,a:'start'});
 o+=`<g class="nd" data-b="86,108,28,28,0,0" data-shape="dia">${block('dia',86,108,28,28)}<path d="M95,117 l10,10 M105,117 l-10,10" stroke="${C.ink}" stroke-width="1.1"/></g>`;
 const ex=node('box',210,92,168,n2('Extract and land','',8.8,K.land,'src'),{h:28,z:'src'});o+=ex.g;
 const oc=node('box',210,132,168,n2(Pp.ocr.t,Pp.ocr.s,8.6,K.ocr,'src'),{h:28,z:'src'});o+=oc.g;
 o+=tag(154,108,'Monarch / Gmail',{light:1})+tag(154,158,'Drive docs',{light:1});
 const ld=node('box',232,210,144,n2('RAW and CORE','',8.8,K.raw+', '+K.core,'data'),{h:34,z:'data'});o+=ld.g;
 const db=node('box',140,216,78,n2(Pp.dbt.t,'',8.6),{h:26,z:'data'});o+=db.g;
 const ts=node('box',78,216,48,n2('Tests','',8.6),{h:26});o+=ts.g;
 o+=ev(34,229,8,2)+T(12,250,'ANALYTICS,',{sz:fs(7.8),c:C.sub,a:'start'})+T(12,261,'MCP servers',{sz:fs(7.8),c:C.sub,a:'start'})+T(140,268,K.dbt,{sz:fs(7.8),mono:1,c:Z.data.k,a:'start'})+legend(14,286,['src','data']);
 a+=arrow('M29,32 H42','c')+arrow(`M104,${18+t1.h} V106`,'c')+arrow('M114,122 H120 Q124,122 124,118 V108 Q124,104 128,104 H208','c')+arrow('M100,136 V150 Q100,154 104,154 H208','c');
 a+=arrow('M378,106 H384 Q388,106 388,110 V225 Q388,229 384,229 H378','c')+arrow(`M294,${132+oc.h} V208`,'c')+arrow('M232,227 H220','c')+arrow('M140,229 H128','c')+arrow('M78,229 H44','c');
 return svg(W,H,Pp.title,a+o)};

// 5. Security: nested access zones as dashed outlines, accent only on the MCP split.
export function dSec(D, inst=""){
  begin(true);ID='s'+inst;const S=D.security,K=D.code;const W=380,H=300;let o='';
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
  begin(true);ID='k'+inst;const L=D.stack.layers;const W=360;let o='';const x=10,w=340,h=24;
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
  begin(true);ID='f'+inst;const R=D.features.rows;const W=448;let o='';const rh=28,hx=78,cw=178,pad=6;const FH=pad*2+R.length*(rh+4)-4;
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
  begin(true);ID='wf'+inst;const Wf=D.workflow;const W=420;let o='',a='';
  const step=28,nh=22;
  o+=T(12,14,'Merge ≠ apply ≠ deploy',{sz:fs(10),w:600,a:'start'})+T(12,27,"Michael's go gates apply, unpause, grants",{sz:fs(7.8),c:C.sub,a:'start'});
  Wf.steps.forEach((s,i)=>{const y=38+i*step;const z=i<2?'src':i===2?null:i===3?'data':'src';
   o+=badge(20,y+nh/2,i+1)+node('box',36,y,368,n2(s.t,s.s,8.6),{h:nh,z:z||undefined}).g;
   if(i<Wf.steps.length-1)a+=arrow(`M20,${y+nh} V${y+step-2}`,'c')});
  o+=legend(12,38+Wf.steps.length*step+2,['src','data']);
  return svg(W,38+Wf.steps.length*step+18,Wf.title,a+o)};

export function dDatabase(D, dbId, inst=""){
  // Stripe-light object cards: taller rows so layer/name + note never clip.
  begin(false);ID='dbx'+inst+(dbId||'');const db=D.databases?.[dbId]||D.databases?.ledger;if(!db)return '';
  const W=460;let o='';
  o+=T(14,16,db.name,{sz:12,w:600,a:'start'})+T(14,32,db.status+' · '+db.oneLine,{sz:8.6,c:C.sub,a:'start'});
  const cols=Math.min(4,Math.max(2,db.schemas.length));
  db.schemas.forEach((s,i)=>{const x=14+(i%cols)*110,y=48+Math.floor(i/cols)*32;
   o+=node('pill',x,y,104,[{t:s,sz:8.6,w:500,mono:1}],{h:26,z:db.status.startsWith('Planned')?'sec':'data',lift:1}).g});
  const schemaRows=Math.ceil(db.schemas.length/cols);let y=48+schemaRows*32+16;
  o+=lab(14,y,'Objects / path',C.sub);y+=12;
  const rowH=44;
  (db.objects||[]).forEach((obj,i)=>{const yy=y+i*rowH;
   o+=node('box',14,yy,432,[{t:obj.layer+' · '+obj.name,sz:9.4,w:600,mono:1},{t:obj.note,sz:8.4,sub:1}],{h:38,z:i===0?'data':null,lift:1}).g});
  o+=legend(14,y+(db.objects||[]).length*rowH+10,db.status.startsWith('Planned')?['sec','data']:['data']);
  return svg(W,y+(db.objects||[]).length*rowH+28,db.name+' schemas',o)};

export function dAirflow(D, inst=""){
  begin(true);ID='af'+inst;const A=D.airflow;const W=420;let o='';
  o+=T(12,14,A.headline,{sz:fs(11),w:600,a:'start'})+T(12,27,'Office-PC Airflow pin · git ≠ unpaused',{sz:fs(7.8),c:C.sub,a:'start'});
  A.counts.forEach((c,i)=>{const x=12+i*134;o+=node('box',x,38,126,[{t:c.n,sz:fs(12),w:600,c:Z.src.k},{t:c.label,sz:fs(7.6),sub:1}],{h:30,z:'src'}).g});
  const rows=[
    ['moonshots_yt_sync_daily','YT sync · unpaused'],
    ['moonshots_eyl_sync','EYL sync · unpaused'],
    ['monarch_load','Monarch → RAW/CORE'],
    ['ledger_email_*','Gmail alerts + poll'],
    ['drive_poll / doc_extract','Drive → OCR → CORE'],
    ['ledger_dbt_daily / dbt_daily','dbt → ANALYTICS'],
    ['agentic_workforce_dbt_daily','workforce dbt'],
    ['hr_dbt_daily · platform_smoke','blueprint / manual'],
    ['*_dev twins (3)','not production'],
  ];
  rows.forEach((r,i)=>{const y=84+i*17;o+=T(16,y,r[0],{sz:fs(8),mono:1,c:Z.src.k,a:'start'})+T(244,y,r[1],{sz:fs(7.8),c:C.sub,a:'start'})});
  o+=legend(12,84+rows.length*17+6,['src'],W-20);
  return svg(W,84+rows.length*17+20,A.title,o)};

export function dDbt(D, inst=""){
  begin(true);
  ID='db'+inst;const L=D.dbt.layers;const W=380;let o='',a='';
  const step=34,nh=24;
  L.forEach((l,i)=>{const y=12+i*step;const z=i<4?'data':'src';
   o+=node(i<2?'cyl':'box',36,y,308,n2(l.t,l.s,8.6),{h:nh,z}).g;
   if(i<L.length-1)a+=arrow(`M190,${y+nh} V${y+step-2}`,'c')});
  o+=legend(12,12+L.length*step+2,['data','src']);
  return svg(W,12+L.length*step+18,D.dbt.title,a+o)};

export function dRearch(inst=""){
  begin(true);ID='r'+inst;let o='',a='';['Terraform','dbt','Airflow','Snowflake'].forEach((t,i)=>{const xx=12+i*88;o+=node(i==3?'cyl':'box',xx,16,72,n2(t,'',9),{h:32,e:i==3}).g;if(i<3)a+=arrow(`M${xx+72},32 H${xx+86}`,'c')});
 return svg(370,64,'Re-architecture preview: Terraform, dbt and Airflow managing a Snowflake platform',a+o)};


export function dMychart(D,tall, inst=""){
  begin(true);
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
