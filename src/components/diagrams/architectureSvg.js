// Flat architecture diagrams after the approved mockup in household-tracker
// files/design/review/portfolio/src/svg.js. Object names are the real structural
// names from the platform repos listed in that README. Do not invent names.
// Inline SVG builders. Flat style after Cloudflare's reference architecture diagrams: hairline 1px strokes, a dot-grid panel,
// numbered step badges, dashed groups, small uppercase labels. Restrained color: one soft tint per zone (data, agents and MCP,
// access, sources and scheduling) plus neutral. Object names are real structural names from the platform repos (see README SOURCES).
// Original light palette; dark mirrors the site dark-theme tokens so diagrams
// follow the header toggle (default dark).
const lightC={ink:'#0b1d36',sub:'#57606a',line:'#d0d7de',conn:'#6e7781',acc:'#29b5e8',tint:'#eaf7fd',accInk:'#0a6688',dark:'#24292f',dot:'#dfe4e9',panel:'#fbfcfd',soft:'#f3f5f7',node:'#ffffff'};
const darkC={ink:'#f4fbfe',sub:'#9bb4c4',line:'#2a4558',conn:'#7a96a8',acc:'#5cc8ef',tint:'#10263f',accInk:'#8ad8f4',dark:'#071422',dot:'#1a3a55',panel:'#0d1b2a',soft:'#10263f',node:'#10263f'};
const lightZ={data:{f:'#eaf7fd',s:'#29b5e8',k:'#0a6688',n:'Snowflake data and catalogs'},agent:{f:'#f2f0fb',s:'#9b8fd6',k:'#5b4bb5',n:'MCP and agents'},
 sec:{f:'#fdf4e7',s:'#d6a24e',k:'#8a5300',n:'Roles and access'},src:{f:'#eaf6f0',s:'#5aae8a',k:'#1d6e4f',n:'Sources and scheduling'}};
const darkZ={data:{f:'#123149',s:'#5cc8ef',k:'#8ad8f4',n:'Snowflake data and catalogs'},agent:{f:'#1c2438',s:'#9b8fd6',k:'#c4b8f0',n:'MCP and agents'},
 sec:{f:'#2a2418',s:'#d6a24e',k:'#e8c57a',n:'Roles and access'},src:{f:'#143028',s:'#5aae8a',k:'#8fd0b0',n:'Sources and scheduling'}};
const isDark=()=>typeof document!=='undefined'&&document.body.classList.contains('dark-theme');
let C=lightC,Z=lightZ;
const syncTheme=()=>{if(isDark()){C=darkC;Z=darkZ}else{C=lightC;Z=lightZ}};
syncTheme();
const F="Poppins, sans-serif",MONO="ui-monospace, SFMono-Regular, Menlo, monospace";
const esc=s=>String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;');
let mt;const meas=(t,sz,w,it,mono)=>{if(!mt){const m=document.createElementNS('http://www.w3.org/2000/svg','svg');m.setAttribute('style','position:absolute;left:-9999px;top:0;width:10px;height:10px');m.setAttribute('text-rendering','geometricPrecision');mt=document.createElementNS('http://www.w3.org/2000/svg','text');m.appendChild(mt);document.body.appendChild(m)}
 mt.setAttribute('font-family',mono?MONO:F);mt.setAttribute('font-size',sz);mt.setAttribute('font-weight',w);mt.setAttribute('font-style',it?'italic':'normal');mt.textContent=t;return mt.getBBox().width};
const wrapW=(t,maxW0,sz,w,it,mono)=>{const maxW=maxW0*0.88;const ws=String(t).split(' '),o=[];let l='';for(const x of ws){const c=(l?l+' ':'')+x;if(meas(c,sz,w,it,mono)>maxW&&l){o.push(l);l=x}else l=c}if(l)o.push(l);return o};
const T=(x,y,t,o={})=>`<text x="${x}" y="${y}" text-anchor="${o.a||'middle'}" dominant-baseline="middle" font-family="${o.mono?MONO:F}" font-size="${o.sz||10}" font-weight="${o.w||400}"${o.it?' font-style="italic"':''}${o.ls?` letter-spacing="${o.ls}"`:''} fill="${o.c||C.ink}">${esc(t)}</text>`;
const lab=(x,y,t,c=C.accInk,a='start')=>T(x,y,t.toUpperCase(),{sz:8.6,w:500,ls:.6,c,a,mono:1});
let ID='a';
const defs=()=>`<defs><filter id="${ID}sh" x="-20%" y="-20%" width="140%" height="160%"><feDropShadow dx="0" dy="1" stdDeviation="0.8" flood-color="#0b1d36" flood-opacity="0.10"/><feDropShadow dx="0" dy="6" stdDeviation="6" flood-color="#32325d" flood-opacity="0.16"/></filter><pattern id="${ID}dg" width="12" height="12" patternUnits="userSpaceOnUse"><circle cx="1.5" cy="1.5" r="0.9" fill="${C.dot}"/></pattern>
 <marker id="${ID}a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M1,1 L9,5 L1,9" fill="none" stroke="${C.conn}" stroke-width="1.6"/></marker>
 <marker id="${ID}c" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M1,1 L9,5 L1,9" fill="none" stroke="${C.accInk}" stroke-width="1.6"/></marker>
 <marker id="${ID}w" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M1,1 L9,5 L1,9" fill="none" stroke="${C.ink}" stroke-width="1.6"/></marker></defs>`;
// kinds: default grey, 'c' accent (read and data path), 'w' write path (ink, dashed), 'd' dashed grey
const arrow=(d,k)=>{const col=k==='c'?C.accInk:k==='w'?C.ink:C.conn;return `<path d="${d}" fill="none" stroke="${col}" stroke-width="1.1"${k==='w'||k==='d'?' stroke-dasharray="4 3"':''} stroke-linejoin="round" marker-end="url(#${ID}${k==='c'?'c':k==='w'?'w':'a'})"/>`};
const panel=(W,H)=>`<rect x="0.5" y="0.5" width="${W-1}" height="${H-1}" rx="10" fill="${C.panel}" stroke="${C.line}"/><rect x="0.5" y="0.5" width="${W-1}" height="${H-1}" rx="10" fill="url(#${ID}dg)"/>`;
const group=(x,y,w,h,t,o={})=>{const z=o.z?Z[o.z]:null;return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" fill="${o.fill||'none'}" stroke="${z?z.s:o.grey?C.line:C.acc}" stroke-width="1" stroke-dasharray="${o.solid?'0':'3 3'}"/>`+(t?lab(x+8,y+11,t,z?z.k:o.grey?C.sub:C.accInk):'')};
// legend: small swatches plus free text, one row; keys are zone ids.
const legend=(x,y,keys,maxX=1e9)=>{let g='',xx=x;keys.forEach(k=>{const z=Z[k];const iw=15+meas(z.n,8.6,400)/0.88;if(xx>x&&xx+iw>maxX){xx=x;y+=14}g+=`<rect x="${xx}" y="${y-5}" width="10" height="10" rx="2" fill="${z.f}" stroke="${z.s}"/>`+T(xx+15,y,z.n,{sz:8.6,c:C.sub,a:'start'});xx+=15+meas(z.n,8.6,400)+16});return g};
const outline={
 box:(x,y,w,h)=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="6"/>`,
 pill:(x,y,w,h)=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${h/2}"/>`,
 doc:(x,y,w,h)=>`<path d="M${x+6},${y} H${x+w-10} L${x+w},${y+10} V${y+h-6} Q${x+w},${y+h} ${x+w-6},${y+h} H${x+6} Q${x},${y+h} ${x},${y+h-6} V${y+6} Q${x},${y} ${x+6},${y} Z"/><path d="M${x+w-10},${y} V${y+10} H${x+w}" fill="none"/>`,
 dia:(x,y,w,h)=>`<path d="M${x+w/2},${y} L${x+w},${y+h/2} L${x+w/2},${y+h} L${x},${y+h/2} Z"/>`,
 cyl:(x,y,w,h)=>{const ry=5;return `<path d="M${x},${y+ry} a${w/2},${ry} 0 0 1 ${w},0 v${h-2*ry} a${w/2},${ry} 0 0 1 -${w},0 z"/><path d="M${x},${y+ry} a${w/2},${ry} 0 0 0 ${w},0" fill="none"/>`}};
const inset={box:[9,5],pill:[10,4],doc:[8,5],dia:[0,0],cyl:[8,8]};
const block=(shape,x,y,w,h,o={})=>{const z=o.z?Z[o.z]:o.e?Z.data:null;return `<g fill="${z?z.f:C.node}" stroke="${z?z.s:C.line}" stroke-width="1"${o.dash?' stroke-dasharray="3 3"':''}${o.hero?` filter="url(#${ID}sh)"`:''}>${outline[shape](x,y,w,h)}</g>`};
// node: flat block plus wrapped lines; height grows to fit. lines: [{t, sz, w, c, it, mono, sub}]
const node=(shape,x,y,w,lines,o={})=>{const [px,py]=inset[shape];const maxW=w-2*px;const L=[];
 lines.forEach(l=>{const sz=l.sz||10,ww=l.w||400;wrapW(l.t,maxW,sz,ww,l.it,l.mono).forEach(s=>L.push({...l,t:s,sz,lh:sz*1.3}))});
 const th=L.reduce((a,l)=>a+l.lh,0);const h=Math.max(o.h||0,th+2*py+(shape==='cyl'?5:0));let ty=y+(h-th)/2+(shape==='cyl'?2.5:0),g='';
 L.forEach(l=>{g+=T(x+w/2,ty+l.lh/2,l.t,{sz:l.sz,w:l.w,c:l.c||(l.sub?C.sub:C.ink),it:l.it,mono:l.mono});ty+=l.lh});
 return {g:`<g class="nd" data-b="${x},${y},${w},${h},${px},${py}" data-shape="${shape}">${block(shape,x,y,w,h,o)}${g}</g>`,h}};
const n2=(t,s,sz=10,code,z)=>[{t,w:600,sz}].concat(s?[{t:s,sz:8.6,sub:1}]:[]).concat(code?[{t:code,sz:8.6,mono:1,c:z?Z[z].k:C.accInk}]:[]);
const badge=(cx,cy,n)=>`<g class="nd badge" data-b="${cx-8},${cy-8},16,16,0,0" data-shape="badge"><circle cx="${cx}" cy="${cy}" r="8" fill="${C.ink}"/>`+T(cx,cy+0.5,String(n),{sz:8.6,w:600,c:'#fff'})+`</g>`;
const tag=(cx,cy,t,o={})=>{const w=meas(t,8.6,500,0,1)/0.88+14;return `<g class="nd" data-b="${cx-w/2},${cy-8},${w},16,4,0" data-shape="pill"><rect x="${cx-w/2}" y="${cy-8}" width="${w}" height="16" rx="8" fill="${o.light?'#fff':C.dark}" stroke="${o.light?C.line:C.dark}"/>`+T(cx,cy+0.5,t,{sz:8.6,w:500,mono:1,c:o.light?C.sub:'#fff'})+`</g>`};
const svg=(W,H,title,body)=>{syncTheme();return `<svg viewBox="0 0 ${W} ${H}" text-rendering="geometricPrecision" role="img" aria-label="${esc(title)}" xmlns="http://www.w3.org/2000/svg">${defs()}<title>${esc(title)}</title>${panel(W,H)}${body}</svg>`};
const ns=(n,L)=>L.find(x=>x.label===n);

// 1. Lead: zone overview, left to right. Data and read path in accent; the write path is ink and dashed.
export function dLead(D,tall, inst=""){
  syncTheme();ID=(tall?'lT':'lW')+inst;const N=D.lead,K=D.code;let o='',a='';
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

// 2. MCP: numbered request paths, read path in accent, write path dashed ink.
export function dMcp(D, inst=""){
  syncTheme();ID='m'+inst;const L=D.lead.layers,K=D.code;const W=420;let o='',a='';const cx=[106,316],w=184;
 o+=lab(cx[0]-w/2,20,'Read path')+lab(cx[1]-w/2,20,'Write path',C.ink);
 const c0=ns('Clients',L).nodes,m=ns('MCP servers',L).nodes,cx2=ns('Cortex',L).nodes,cat=ns('Catalogs',L).nodes,sn=ns('Snowflake',L).nodes;
 const st=(n,x,y,shape,t,s,z,k,hero)=>{const r=node(shape,x-w/2,y,w,n2(t,s,10,k,z),{h:46,z,hero});return {g:r.g+badge(x-w/2,y,n),h:r.h}};
 const ys=[36,116,196,276,356];
 const R=[st(1,cx[0],ys[0],'box',c0[0].t,c0[0].s),st(2,cx[0],ys[1],'box',m[0].t,m[0].s,'agent',K.readSrv,1),st(3,cx[0],ys[2],'box',cx2.map(n=>n.t).join(', '),'Analyst: '+cx2[1].s,'agent','',1),st(4,cx[0],ys[3],'box',cat[0].t,'and '+cat[1].t,'data',K.svDoc+', '+K.cs),st(5,cx[0],ys[4],'cyl',sn[0].t,sn[0].s,'data',K.analytics)];
 const Wr=[st(1,cx[1],ys[0],'box',c0[1].t,c0[1].s),st(2,cx[1],ys[1],'box',m[1].t,m[1].s,'agent',K.writeSrv,1),st(3,cx[1],ys[2],'cyl',sn[1].t,sn[1].s,'data',K.core)];
 R.forEach(r=>o+=r.g);Wr.forEach(r=>o+=r.g);
 [0,1,2,3].forEach(i=>a+=arrow(`M${cx[0]},${ys[i]+R[i].h} V${ys[i+1]-2}`,'c'));[0,1].forEach(i=>a+=arrow(`M${cx[1]},${ys[i]+Wr[i].h} V${ys[i+1]-2}`,'w'));
 o+=node('box',cx[1]-w/2,280,w,[{t:'No write tools on the read path: the read server serves catalogs only.',sz:8.8,c:C.sub}],{h:54,dash:1}).g;
 o+=legend(14,430,['data','agent']);
 return svg(W,444,D.lead.title,a+o)};

// 3. Data domains: crow's-foot ERD, flat tables with mono names and hairline rows.
export function dErd(D, inst=""){
  syncTheme();ID='e'+inst;const E=D.erd.entities;const W=452;let o='',a='';
 D.erd.areas.forEach((t,i)=>{const cols=3;const x=8+(i%cols)*148,y=10+Math.floor(i/cols)*34;
  o+=node('pill',x,y,140,[{t,sz:8.2,w:500,c:i===4?Z.sec.k:(i===0?Z.data.k:C.sub)}],{h:28,z:i===4?'sec':(i===0?'data':null)}).g});
 const ent=(x,y,w,e,cat,hero)=>{const rows=cat?[]:e.rows;const ml=cat?[]:wrapW(e.meaning,w-16,8.6,400,1);const h=36+rows.length*15+(cat?0:8+ml.length*11);
  let g=`<g class="nd" data-b="${x},${y},${w},${h},6,3" data-shape="box"><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="6" fill="${cat?Z.data.f:'#fff'}" stroke="${cat?Z.data.s:C.line}"${hero?` filter="url(#${ID}sh)"`:''}/><path d="M${x+0.5},${y+33} H${x+w-0.5}" stroke="${cat?Z.data.s:C.line}"/>`;
  g+=T(x+8,y+11,e.layer,{sz:8.6,w:500,ls:.4,c:cat?Z.data.k:C.sub,a:'start',mono:1})+T(x+8,y+24,e.name,{sz:e.name.length>18?8.6:9.4,w:600,a:'start',mono:1});
  rows.forEach((r,i)=>{const yy=y+45+i*15;if(i)g+=`<path d="M${x+8},${yy-7.5} H${x+w-8}" stroke="${C.soft}"/>`;g+=T(x+8,yy,r[0],{sz:9,a:'start',mono:1})+((r[1]||r[2])?T(x+w-8,yy,r[1]||r[2],{sz:8.6,w:500,c:r[1]?C.accInk:C.sub,a:'end',mono:1}):'')});
  ml.forEach((l,i)=>g+=T(x+8,y+h-8-(ml.length-1-i)*11,l,{sz:8.6,it:1,c:C.sub,a:'start'}));return {g:g+'</g>',h}};
 const w=132,xs=[8,160,312],y=96;const hs=[];[0,1,2].forEach(i=>{const r=ent(xs[i],y,w,E[i],0,i>0);o+=r.g;hs.push(r.h)});
 const cf=(x1,x2,yy)=>{const s=C.ink;let g=`<line x1="${x1}" y1="${yy}" x2="${x2}" y2="${yy}" stroke="${s}" stroke-width="1"/>`;
  g+=`<line x1="${x1+5}" y1="${yy-5}" x2="${x1+5}" y2="${yy+5}" stroke="${s}"/><line x1="${x1+9}" y1="${yy-5}" x2="${x1+9}" y2="${yy+5}" stroke="${s}"/>`;
  g+=`<path d="M${x2-8},${yy} L${x2},${yy-5} M${x2-8},${yy} L${x2},${yy+5}" stroke="${s}" fill="none"/><circle cx="${x2-12}" cy="${yy}" r="3" fill="#fff" stroke="${s}"/>`;return g};
 // many to one: crow's foot on the RAW side (several loads), bars on the CORE side (one row)
 const mo=(x1,x2,yy)=>{const s=C.ink;return `<line x1="${x1}" y1="${yy}" x2="${x2}" y2="${yy}" stroke="${s}" stroke-width="1"/><path d="M${x1+8},${yy} L${x1},${yy-5} M${x1+8},${yy} L${x1},${yy+5}" stroke="${s}" fill="none"/><circle cx="${x1+11}" cy="${yy}" r="3" fill="#fff" stroke="${s}"/><line x1="${x2-5}" y1="${yy-5}" x2="${x2-5}" y2="${yy+5}" stroke="${s}"/><line x1="${x2-9}" y1="${yy-5}" x2="${x2-9}" y2="${yy+5}" stroke="${s}"/>`};
 a+=mo(xs[0]+w,xs[1],y+45)+cf(xs[1]+w,xs[2],y+60);
 const y2=y+Math.max(...hs)+38;const s1=ent(xs[1],y2,w,E[4],1),s2=ent(xs[2],y2,w,E[3],1);o+=s1.g+s2.g;
 a+=arrow(`M${xs[1]+w/2},${y+hs[1]} V${y2-2}`,'d')+arrow(`M${xs[2]+w/2},${y+hs[2]} V${y2-2}`,'d');
 o+=tag(xs[1]+w/2,y+hs[1]+19,'indexes',{light:1})+tag(xs[2]+w/2,y+hs[2]+19,'reads',{light:1});
 const ky=y2;o+=T(8,ky+4,'Key',{sz:9,w:600,a:'start'});
 o+=`<line x1="8" y1="${ky+19}" x2="32" y2="${ky+19}" stroke="${C.ink}"/><line x1="12" y1="${ky+14}" x2="12" y2="${ky+24}" stroke="${C.ink}"/><line x1="16" y1="${ky+14}" x2="16" y2="${ky+24}" stroke="${C.ink}"/>`+T(38,ky+19,'exactly one',{sz:8.6,c:C.sub,a:'start'});
 o+=`<line x1="8" y1="${ky+36}" x2="32" y2="${ky+36}" stroke="${C.ink}"/><path d="M24,${ky+36} L32,${ky+31} M24,${ky+36} L32,${ky+41}" stroke="${C.ink}" fill="none"/><circle cx="20" cy="${ky+36}" r="3" fill="#fff" stroke="${C.ink}"/>`+T(38,ky+36,'zero or many',{sz:8.6,c:C.sub,a:'start'});
 o+=`<line x1="8" y1="${ky+53}" x2="32" y2="${ky+53}" stroke="${C.conn}" stroke-dasharray="4 3"/>`+T(38,ky+53,'derived or read',{sz:8.6,c:C.sub,a:'start'})+T(8,ky+70,'Real object names, no row data',{sz:8.6,c:C.sub,a:'start'});
 const nl=wrapW(D.erd.searchNote,W-16,8.6,400,1);const H=Math.max(y2+s1.h,ky+77)+14+nl.length*12;nl.forEach((l,i)=>o+=T(8,H-12-(nl.length-1-i)*12,l,{sz:8.6,it:1,c:C.sub,a:'start'}));
 return svg(W,H,D.erd.title,a+o)};

// 4. Pipelines: BPMN-style lanes with hairline separators and mono lane labels, events, gateway, tasks.
export function dPipes(D, inst=""){
  syncTheme();ID='p'+inst;const Pp=D.pipes,K=D.code;const W=400,H=328;let o='',a='';
 const lanes=[['Sources',0,72],['Airflow on office PC',72,138],['Snowflake',210,96]];
 lanes.forEach(([t,y,h],i)=>{if(i)o+=`<line x1="1" y1="${y}" x2="${W-1}" y2="${y}" stroke="${C.line}"/>`;o+=lab(10,y+11,t,i<2?Z.src.k:Z.data.k)});
 const ev=(cx,cy,r,sw)=>`<circle cx="${cx}" cy="${cy}" r="${r}" fill="#fff" stroke="${C.ink}" stroke-width="${sw}"/>`;
  o+=ev(24,40,9,1.2)+T(24,59,'Start',{sz:8.6,c:C.sub});
  const t1=node('box',46,24,118,n2('Monarch · Gmail · Drive · YT','',9.6),{h:34,z:'src'});o+=t1.g;
  o+=T(180,28,'EAI_MONARCH · EAI_GMAIL',{sz:8.6,mono:1,c:Z.src.k,a:'start'})+T(180,40,'drive_poll · moonshots_*',{sz:8.6,c:C.sub,a:'start'})+T(180,52,'MyChart FHIR planned',{sz:8.6,c:Z.sec.k,a:'start'});
 o+=T(10,96,Pp.airflow.s,{sz:8.6,it:1,c:C.sub,a:'start'});
 o+=`<g class="nd" data-b="83,124,34,34,0,0" data-shape="dia">${block('dia',83,124,34,34)}<path d="M94,135 l12,12 M106,135 l-12,12" stroke="${C.ink}" stroke-width="1.2"/></g>`;
 const ex=node('box',214,102,166,n2('Extract and land','',9.6,K.land,'src'),{h:34,z:'src',hero:1});o+=ex.g;
 const oc=node('box',214,150,166,n2(Pp.ocr.t,Pp.ocr.s,9.6,K.ocr,'src'),{z:'src'});o+=oc.g;
 o+=tag(158,120,'Monarch / Gmail',{light:1})+tag(158,184,'Drive docs',{light:1});
 const ld=node('box',236,234,144,n2('RAW and CORE','',9.6,K.raw+', '+K.core,'data'),{h:46,z:'data',hero:1});o+=ld.g;
 const db=node('box',138,241,82,n2(Pp.dbt.t,'',9.6),{h:32,z:'data'});o+=db.g;
 const ts=node('box',74,241,50,n2('Tests','',9.6),{h:32});o+=ts.g;
 o+=ev(36,257,10,2.6)+T(14,280,'ANALYTICS,',{sz:8.6,c:C.sub,a:'start'})+T(14,292,'MCP servers',{sz:8.6,c:C.sub,a:'start'})+T(138,298,K.dbt,{sz:8.6,mono:1,c:Z.data.k,a:'start'})+legend(14,316,['src','data']);
 a+=arrow('M33,41 H44','c')+arrow(`M105,${24+t1.h} V122`,'c')+arrow('M117,141 H124 Q128,141 128,137 V124 Q128,120 132,120 H212','c')+arrow('M100,158 V180 Q100,184 104,184 H212','c');
 a+=arrow('M380,120 H386 Q390,120 390,124 V253 Q390,257 386,257 H382','c')+arrow(`M297,${150+oc.h} V232`,'c')+arrow('M236,257 H222','c')+arrow('M138,257 H126','c')+arrow('M74,257 H48','c');
 return svg(W,H,Pp.title,a+o)};

// 5. Security: nested access zones as dashed outlines, accent only on the MCP split.
export function dSec(D, inst=""){
  syncTheme();ID='s'+inst;const S=D.security,K=D.code;const W=380,H=342;let o='';
 o+=group(8,8,364,310,'Roles',{z:'sec'})+group(20,70,340,240,'MCP split',{z:'agent'})+group(32,152,316,150,'Cortex gate',{z:'sec'})+group(44,214,292,80,'Catalogs',{z:'data'});
 S.roles.forEach((t,i)=>o+=node('pill',18+i*116,28,108,[{t,sz:9.6,w:500}],{h:28,z:'sec'}).g);
 o+=node('box',36,88,148,n2(S.write,'',10,K.writeSrv,'agent'),{h:50,z:'agent',hero:1}).g+node('box',196,88,148,n2(S.read.t,S.read.s,10,K.readSrv,'agent'),{h:50,z:'agent',hero:1}).g;
 o+=`<path d="M70,172 l14,5 v9 q0,9 -14,14 q-14,-5 -14,-14 v-9 z" fill="#fff" stroke="${Z.sec.k}" stroke-width="1.1"/>`;
 o+=node('box',100,168,220,n2(S.gate.t,S.gate.s),{h:38,z:'sec',hero:1}).g;
 o+=node('cyl',60,232,124,n2(S.cat[0],'',9.6,K.svDoc,'data'),{h:46,z:'data'}).g+node('cyl',196,232,124,n2(S.cat[1],'',9.6,K.cs,'data'),{h:46,z:'data'}).g;
 o+=legend(14,332,['sec','agent','data']);
 return svg(W,H,S.title,o)};

// 6. Stack: flat layer bands, the agent interface in accent.
export function dStack(D, inst=""){
  syncTheme();ID='k'+inst;const L=D.stack.layers;const W=360;let o='';const x=10,w=340,h=34;
 const zs=['agent','src','data','src',null];L.forEach((l,i)=>{const y=10+i*(h+6);const z=zs[i]?Z[zs[i]]:null;const one=l.nodes.length==1&&l.nodes[0].s;
  o+=`<g class="nd" data-b="${x},${y},${w},${h},8,2" data-shape="box"><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="6" fill="${z?z.f:'#fff'}" stroke="${z?z.s:C.line}"${i==0||i==2?` filter="url(#${ID}sh)"`:''}/>`+
  (one?T(x+12,y+12,l.nodes[0].t,{sz:9.6,w:600,a:'start'})+T(x+12,y+25,l.nodes[0].s,{sz:8.6,c:C.sub,a:'start'}):T(x+12,y+h/2,l.nodes.map(n=>n.t).join(' and '),{sz:9.6,w:600,a:'start'}))+
  T(x+w-12,y+h/2,l.label.toUpperCase(),{sz:8.6,w:500,ls:.4,mono:1,c:z?z.k:C.sub,a:'end'})+`</g>`});
 return svg(W,10+L.length*(h+6)+4,D.stack.title,o)};

// 7. Snowflake features: capability matrix, hairline cells, accent icons.
export function dFeat(D, inst=""){
  syncTheme();ID='f'+inst;const R=D.features.rows;const W=448;let o='';const rh=38,hx=84,cw=177,pad=6;const FH=pad*2+R.length*(rh+6)-6;
 const icon=(k,x,y,s)=>{switch(k){
  case 'hex':return `<rect x="${x}" y="${y+1}" width="13" height="12" rx="3" fill="none" stroke="${s}" stroke-width="1.2"/><path d="M${x+3},${y+7} h7" stroke="${s}" stroke-width="1.2"/>`;
  case 'grid':return `<rect x="${x-1}" y="${y}" width="15" height="14" rx="2" fill="none" stroke="${s}" stroke-width="1.2"/><path d="M${x-1},${y+5} h15 M${x+5},${y} v14" stroke="${s}" stroke-width="1.1"/>`;
  case 'search':return `<circle cx="${x+6}" cy="${y+6}" r="5" fill="none" stroke="${s}" stroke-width="1.3"/><path d="M${x+10},${y+10} l4,4" stroke="${s}" stroke-width="1.5"/>`;
  case 'layers':return `<path d="M${x+7},${y} l8,4 l-8,4 l-8,-4 z M${x-1},${y+7} l8,4 l8,-4 M${x-1},${y+10} l8,4 l8,-4" fill="none" stroke="${s}" stroke-width="1.1"/>`;
  case 'clock':return `<circle cx="${x+7}" cy="${y+7}" r="6.5" fill="none" stroke="${s}" stroke-width="1.2"/><path d="M${x+7},${y+3} v4 l3,2" stroke="${s}" stroke-width="1.2" fill="none"/>`;
  default:return `<path d="M${x},${y+7} h6 l4,-6 h4 M${x+6},${y+7} l4,6 h4" fill="none" stroke="${s}" stroke-width="1.2"/>`}};
 R.forEach((r,i)=>{const y=pad+i*(rh+6);o+=`<g class="nd" data-b="${pad},${y},${hx-14},${rh},4,2" data-shape="box">`+T(pad+6,y+rh/2,r.label.toUpperCase(),{sz:8.6,w:500,ls:.4,mono:1,c:C.sub,a:'start'})+'</g>';
  const span=r.cells.length==1;r.cells.forEach((c,j)=>{const x=hx+j*(cw+6),w=span?cw*2+6:cw;
   const z=r.z?Z[r.z]:null;o+=`<g class="nd" data-b="${x+24},${y},${w-24},${rh},4,2" data-shape="box"><rect x="${x}" y="${y}" width="${w}" height="${rh}" rx="6" fill="${z?z.f:'#fff'}" stroke="${z?z.s:C.line}"${i==0?` filter="url(#${ID}sh)"`:''}/>`+icon(c.i,x+9,y+12,z?z.k:C.sub);
   o+=T(x+30,c.s?y+13:y+rh/2,c.t,{sz:9.2,w:600,a:'start'})+(c.s?T(x+30,y+26,c.s,{sz:8.6,c:c.m&&z?z.k:C.sub,a:'start',mono:c.m}):'')+'</g>'})});
 o+=legend(pad+6,FH+10,['agent','data']);
 return svg(W,FH+20,D.features.title,o)};

export function dWorkflow(D, inst=""){
  syncTheme();ID='wf'+inst;const Wf=D.workflow;const W=440;let o='',a='';
  o+=T(14,18,'Merge ≠ apply ≠ deploy',{sz:11,w:600,a:'start'})+T(14,34,'Michael’s go gates apply, unpause, grants',{sz:8.6,c:C.sub,a:'start'});
  Wf.steps.forEach((s,i)=>{const y=52+i*42;const z=i<2?'src':i===2?null:i===3?'data':'src';
   o+=badge(24,y+16,i+1)+node('box',44,y,360,n2(s.t,s.s,10),{h:34,z:z||undefined,hero:i===3||i===5}).g;
   if(i<Wf.steps.length-1)a+=arrow(`M24,${y+34} V${y+40}`,'c')});
  o+=legend(14,52+Wf.steps.length*42+6,['src','data']);
  return svg(W,52+Wf.steps.length*42+28,Wf.title,a+o)};

export function dDatabase(D, dbId, inst=""){
  syncTheme();ID='dbx'+inst+(dbId||'');const db=D.databases?.[dbId]||D.databases?.ledger;if(!db)return '';
  const W=440;let o='';
  o+=T(14,18,db.name,{sz:12,w:600,a:'start'})+T(14,34,db.status+' · '+db.oneLine,{sz:8.6,c:C.sub,a:'start'});
  const cols=Math.min(4,Math.max(2,db.schemas.length));
  db.schemas.forEach((s,i)=>{const x=14+(i%cols)*106,y=52+Math.floor(i/cols)*36;
   o+=node('pill',x,y,98,[{t:s,sz:8.8,w:500,mono:1}],{h:26,z:db.status.startsWith('Planned')?'sec':'data'}).g});
  const schemaRows=Math.ceil(db.schemas.length/cols);let y=52+schemaRows*36+16;
  o+=lab(14,y,'Objects / path',C.sub);y+=10;
  (db.objects||[]).forEach((obj,i)=>{const yy=y+i*40;
   o+=node('box',14,yy,412,[{t:obj.layer+' · '+obj.name,sz:9.4,w:600,mono:1},{t:obj.note,sz:8.6,sub:1}],{h:34,z:i===0?'data':null}).g});
  o+=legend(14,y+(db.objects||[]).length*40+10,db.status.startsWith('Planned')?['sec','data']:['data']);
  return svg(W,y+(db.objects||[]).length*40+30,db.name+' schemas',o)};

export function dAirflow(D, inst=""){
  syncTheme();ID='af'+inst;const A=D.airflow;const W=420;let o='';
  o+=T(14,18,A.headline,{sz:12,w:600,a:'start'})+T(14,34,'Office-PC Airflow pin · git ≠ unpaused',{sz:8.6,c:C.sub,a:'start'});
  A.counts.forEach((c,i)=>{const x=14+i*132;o+=node('box',x,48,120,[{t:c.n,sz:16,w:600,c:Z.src.k},{t:c.label,sz:8.6,sub:1}],{h:48,z:'src',hero:i===2}).g});
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
  rows.forEach((r,i)=>{const y=118+i*22;o+=T(18,y,r[0],{sz:8.8,mono:1,c:Z.src.k,a:'start'})+T(248,y,r[1],{sz:8.6,c:C.sub,a:'start'})});
  o+=legend(14,118+rows.length*22+8,['src'],W-20);
  return svg(W,118+rows.length*22+28,A.title,o)};

export function dDbt(D, inst=""){
  syncTheme();
  ID='db'+inst;const L=D.dbt.layers;const W=400;let o='',a='';
  L.forEach((l,i)=>{const y=16+i*52;const z=i<2?'data':i<4?'data':'src';
   o+=node(i<2?'cyl':'box',40,y,320,n2(l.t,l.s,10),{h:40,z,hero:i===3||i===4}).g;
   if(i<L.length-1)a+=arrow(`M200,${y+40} V${y+50}`,'c')});
  o+=legend(14,16+L.length*52+4,['data','src']);
  return svg(W,16+L.length*52+24,D.dbt.title,a+o)};

export function dRearch(inst=""){
  syncTheme();ID='r'+inst;let o='',a='';['Terraform','dbt','Airflow','Snowflake'].forEach((t,i)=>{const xx=12+i*88;o+=node(i==3?'cyl':'box',xx,22,72,n2(t),{h:42,e:i==3}).g;if(i<3)a+=arrow(`M${xx+72},43 H${xx+86}`,'c')});
 return svg(370,86,'Re-architecture preview: Terraform, dbt and Airflow managing a Snowflake platform',a+o)};


export function dMychart(D,tall, inst=""){
  syncTheme();
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
