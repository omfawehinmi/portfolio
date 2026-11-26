import { ERD_MODELS } from '../../data/erdModels.js'

// Flat architecture diagrams. Object names are real structural names from the
// platform repos (see README SOURCES). Do not invent names.
// Visual language: Liam ERD Style A (green accent) for ALL diagrams — process
// flows and ERDs share chrome (white cards, soft shadows, gray borders, green
// accent). Process flows keep flow structure (not table/column ERD layout).
// Light-mode only. Page chrome / cards / nav stay on site tokens.
/* Liam Style A — shared by process flows + ERDs */
const LIAM={
  green:'#12b76a',
  soft:'rgba(18,183,106,0.12)',
  header:'#f3f5f8',
  border:'#e2e5eb',
  type:'#9aa3b2',
  ink:'#1a1f2c',
  conn:'#98a2b3',
  schema:'#6b7280',
};
const C={
  ink:LIAM.ink,
  sub:LIAM.schema,
  line:LIAM.border,
  conn:LIAM.conn,
  acc:LIAM.green,
  tint:'#e8f8ef',
  accInk:'#0f7a4c',
  dark:LIAM.ink,
  dot:'transparent',
  panel:'transparent',
  soft:LIAM.header,
  node:'#ffffff',
  disc:LIAM.header,
};
/* Zone tints stay light; strokes lean Liam gray with green hero accents */
const Z={
  data:{f:'#ffffff',s:'#86d4ad',k:'#0f7a4c',n:'Snowflake data and catalogs',tint:'#eefaf3'},
  agent:{f:'#ffffff',s:'#c4b5e8',k:'#5b3fa6',n:'MCP and agents',tint:'#f6f3fc'},
  sec:{f:'#ffffff',s:'#e0c08a',k:'#8a4b12',n:'Roles and access',tint:'#fdf6ea'},
  src:{f:'#ffffff',s:'#86d4ad',k:'#0f6b4c',n:'Sources and scheduling',tint:'#eefaf3'}
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
const defs=()=>`<defs><filter id="${ID}sh" x="-50%" y="-50%" width="200%" height="220%"><feDropShadow dx="0" dy="1.5" stdDeviation="2" flood-color="#101828" flood-opacity="0.10"/></filter>
 <filter id="${ID}shH" x="-50%" y="-50%" width="200%" height="220%"><feDropShadow dx="0" dy="3" stdDeviation="3.2" flood-color="#101828" flood-opacity="0.14"/></filter>
 <marker id="${ID}a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M1.5,1.5 L8.5,5 L1.5,8.5 Z" fill="${LIAM.conn}"/></marker>
 <marker id="${ID}c" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M1.5,1.5 L8.5,5 L1.5,8.5 Z" fill="${LIAM.green}"/></marker>
 <marker id="${ID}w" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M1.5,1.5 L8.5,5 L1.5,8.5 Z" fill="${LIAM.ink}"/></marker></defs>`;
// kinds: default Liam gray, 'c' green accent path, 'w' write path (ink, dashed), 'd' dashed grey
const arrow=(d,k)=>{const col=k==='c'?LIAM.green:k==='w'?LIAM.ink:LIAM.conn;return `<path class="edge" d="${d}" fill="none" stroke="${col}" stroke-width="${COMPACT?1.15:1.35}"${k==='w'||k==='d'?' stroke-dasharray="4 3.5"':''} stroke-linecap="round" stroke-linejoin="round" marker-end="url(#${ID}${k==='c'?'c':k==='w'?'w':'a'})"/>`};
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
const block=(shape,x,y,w,h,o={})=>{
  const z=o.z?Z[o.z]:o.e?Z.data:null;
  // Liam chrome: white cards that pop (always shadowed); hero gets green border
  const fill=shape==='disc'?C.disc:(z?(z.tint||C.node):C.node);
  const hero=!!(o.hero||o.lift);
  const stroke=hero?LIAM.green:(z?z.s:LIAM.border);
  const sw=hero?1.75:(DENSITY<0.95?1.1:1.25);
  const filt=hero?` filter="url(#${ID}shH)"`:` filter="url(#${ID}sh)"`;
  return `<g class="nd-face" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"${o.dash?' stroke-dasharray="3.5 3.5"':''}${filt}>${outline[shape](x,y,w,h)}</g>`;
};
/** Small real-platform marks for bottom-right node badges (not letter chips). */
const platMark=(kind,cx,cy,s)=>{
  const r=s/2;
  if(kind==='snowflake'){
    // Filled Snowflake crystal (brand blue) — recognizable at badge size
    const arms=[];
    for(let i=0;i<6;i++){
      const a=(i*60)*Math.PI/180;
      const x2=cx+Math.cos(a)*r*0.78, y2=cy+Math.sin(a)*r*0.78;
      const px=Math.cos(a+Math.PI/2)*r*0.14, py=Math.sin(a+Math.PI/2)*r*0.14;
      const mx=cx+Math.cos(a)*r*0.28, my=cy+Math.sin(a)*r*0.28;
      arms.push(`M${cx+px},${cy+py} L${x2+px*0.35},${y2+py*0.35} L${x2-px*0.35},${y2-py*0.35} L${cx-px},${cy-py}`);
      arms.push(`M${mx+px*1.6},${my+py*1.6} L${mx},${my} L${mx-px*1.6},${my-py*1.6}`);
    }
    return `<g fill="#29B5E8"><circle cx="${cx}" cy="${cy}" r="${r*0.16}"/><path d="${arms.join(' ')}"/></g>`;
  }
  if(kind==='airflow'){
    // Apache Airflow plane mark (brand blue) — clearer silhouette
    return `<g fill="#017CEE"><path d="M${cx-r*0.05},${cy+r*0.55} L${cx-r*0.05},${cy-r*0.15} L${cx-r*0.55},${cy+r*0.05} L${cx-r*0.55},${cy-r*0.15} L${cx-r*0.05},${cy-r*0.45} L${cx-r*0.05},${cy-r*0.72} L${cx+r*0.12},${cy-r*0.45} L${cx+r*0.72},${cy-r*0.55} L${cx+r*0.72},${cy-r*0.28} L${cx+r*0.2},${cy-r*0.18} L${cx+r*0.55},${cy+r*0.42} L${cx+r*0.32},${cy+r*0.55} L${cx+r*0.02},${cy+r*0.05} L${cx-r*0.05},${cy+r*0.55} Z"/></g>`;
  }
  if(kind==='dbt'){
    // dbt hex + dot
    const hr=r*0.72;
    return `<g fill="none" stroke="#FF694B" stroke-width="${Math.max(1.1,s*0.1)}"><path d="M${cx},${cy-hr} L${cx+hr*0.86},${cy-hr*0.5} L${cx+hr*0.86},${cy+hr*0.5} L${cx},${cy+hr} L${cx-hr*0.86},${cy+hr*0.5} L${cx-hr*0.86},${cy-hr*0.5} Z"/><circle cx="${cx}" cy="${cy}" r="${r*0.18}" fill="#FF694B" stroke="none"/></g>`;
  }
  if(kind==='mcp'){
    // MCP plug / port mark
    return `<g fill="none" stroke="#7C5CFF" stroke-width="${Math.max(1.15,s*0.1)}" stroke-linecap="round"><rect x="${cx-r*0.45}" y="${cy-r*0.35}" width="${r*0.9}" height="${r*0.7}" rx="${r*0.12}"/><path d="M${cx-r*0.2},${cy-r*0.35} V${cy-r*0.55} M${cx+r*0.2},${cy-r*0.35} V${cy-r*0.55}"/><circle cx="${cx-r*0.18}" cy="${cy+r*0.05}" r="${r*0.08}" fill="#7C5CFF" stroke="none"/><circle cx="${cx+r*0.18}" cy="${cy+r*0.05}" r="${r*0.08}" fill="#7C5CFF" stroke="none"/></g>`;
  }
  if(kind==='github'){
    // GitHub Mark–inspired silhouette (simplified)
    return `<g fill="#24292F"><circle cx="${cx}" cy="${cy}" r="${r*0.72}"/><path d="M${cx-r*0.28},${cy+r*0.15} q${r*0.28},${r*0.45} ${r*0.56},0" fill="#fff"/><circle cx="${cx-r*0.22}" cy="${cy-r*0.08}" r="${r*0.1}" fill="#fff"/><circle cx="${cx+r*0.22}" cy="${cy-r*0.08}" r="${r*0.1}" fill="#fff"/></g>`;
  }
  return '';
};
const PLATFORM={
  snowflake:{bg:'#E8F7FC', stroke:'#29B5E8'},
  airflow:{bg:'#E6F4FB', stroke:'#017CEE'},
  dbt:{bg:'#FFF0EB', stroke:'#FF694B'},
  mcp:{bg:'#F3EEFF', stroke:'#7C5CFF'},
  github:{bg:'#F3F4F6', stroke:'#24292F'},
};
const platformBadge=(x,y,w,h,plat)=>{
  if(!plat||!PLATFORM[plat]) return '';
  const s=Math.max(15, Math.round(17*Math.min(1,DENSITY+0.1)));
  const bx=x+w-s*0.22, by=y+h-s*0.22;
  const p=PLATFORM[plat];
  return `<g class="nd-plat" pointer-events="none" aria-hidden="true"><circle cx="${bx}" cy="${by}" r="${s/2}" fill="${p.bg}" stroke="${p.stroke}" stroke-width="1.1" filter="url(#${ID}sh)"/>${platMark(plat,bx,by,s*0.9)}</g>`;
};
// node: soft Issuing-style block; interactive .nd group for hover/click enlarge.
const node=(shape,x,y,w,lines,o={})=>{const [px,py]=insetFor(shape);const maxW=w-2*px;const L=[];
 const lhMul=DENSITY<0.95?1.05:(COMPACT?1.1:1.25);
 lines.forEach(l=>{const sz=fs(l.sz||10),ww=l.w||500;wrapW(l.t,maxW,sz,ww,l.it,l.mono).forEach(s=>L.push({...l,t:s,sz,lh:sz*lhMul}))});
 const th=L.reduce((a,l)=>a+l.lh,0);const minH=o.h?Math.round(o.h*Math.min(1,DENSITY+0.08)):0;const h=Math.max(minH,th+2*py+(shape==='cyl'?(COMPACT?2:4):0));let ty=y+(h-th)/2+(shape==='cyl'?2:0),g='';
 L.forEach(l=>{g+=T(x+w/2,ty+l.lh/2,l.t,{sz:l.sz,w:l.w||500,c:l.c||(l.sub?C.sub:C.ink),it:l.it,mono:l.mono});ty+=l.lh});
 const nid=o.id||`${shape}-${Math.round(x)}-${Math.round(y)}`;
 const plat=platformBadge(x,y,w,h,o.platform);
 return {g:`<g class="nd" tabindex="0" role="button" data-node="${esc(nid)}" data-b="${x},${y},${w},${h},${px},${py}" data-shape="${shape}">${block(shape,x,y,w,h,o)}${g}${plat}</g>`,h,x,y,w}};
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
const svg=(W,H,title,body,cls='')=>`<svg class="bx-arch bx-arch--liam${cls&&!String(cls).includes('liam')?` ${cls}`:cls?` ${cls}`:''}" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMin meet" text-rendering="geometricPrecision" role="img" aria-label="${esc(title)}" xmlns="http://www.w3.org/2000/svg">${defs()}<title>${esc(title)}</title>${panel(W,H)}${body}</svg>`;
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
  const af=n2(N.airflow.t,N.airflow.s,10,K.airflow,'src');const ah=node('box',104,0,112,af,{h:66}).h;o+=node('box',104,Math.round(141-ah/2),112,af,{h:66,z:'src',hero:1,platform:'airflow'}).g;
  o+=group(236,32,152,250,'',{z:'data'});
  [['Landing schema',46,K.raw],['Conformed schema',120,K.core],['Analytics schema',194,K.analytics]].forEach(([t,y,k])=>o+=node('cyl',252,y,120,n2(t,'',10,k,'data'),{h:52,z:'data',hero:k===K.core,platform:'snowflake'}).g);
  o+=node('box',412,90,118,n2('Semantic views','',10,K.svOps,'data'),{h:44,z:'data',platform:'snowflake'}).g+node('box',412,166,118,n2('Cortex Search services','',10,K.cs,'data'),{h:48,z:'data',platform:'snowflake'}).g;
  o+=node('box',550,88,104,[{t:'MCP',sz:8.6,mono:1,c:Z.agent.k}].concat(n2('Read MCP server')),{h:50,z:'agent',hero:1,platform:'mcp'}).g+node('box',550,166,104,[{t:'MCP',sz:8.6,mono:1,c:Z.agent.k}].concat(n2('Write MCP server')),{h:50,z:'agent',hero:1,platform:'mcp'}).g;
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
 row('Scheduler',50,yy=>o+=node('box',50,yy,230,n2(N.airflow.t,N.airflow.s,10,K.airflow,'src'),{h:50,z:'src',hero:1,platform:'airflow'}).g);
 a+=arrow(`M165,${y-16} V${y-2}`,'c');const sy=y;
 o+=group(6,y+12,318,82,'',{z:'data'});
 row('Snowflake',58,yy=>[['Landing schema',K.raw],['Conformed schema',K.core],['Analytics schema',K.analytics]].forEach(([t,k],i)=>o+=node('cyl',14+i*102,yy+4,96,n2(t,'',9.6,k,'data'),{h:52,z:'data',hero:i==1,platform:'snowflake'}).g));
 a+=arrow(`M250,${y-10} V${y-2}`,'c');
 row('Catalogs',42,yy=>{o+=node('box',12,yy,148,n2('Semantic views','',10,K.svOps,'data'),{h:42,z:'data'}).g+node('box',170,yy,148,n2('Cortex Search services','',10,K.cs,'data'),{h:42,z:'data'}).g});
 a+=arrow(`M86,${y-16} V${y-2}`,'c');const my=y;
 row('MCP servers',46,yy=>{o+=node('box',12,yy,148,n2('Read MCP server','',10,K.readSrv,'agent'),{h:46,z:'agent',hero:1,platform:'mcp'}).g+node('box',170,yy,148,n2('Write MCP server','',10,K.writeSrv,'agent'),{h:46,z:'agent',hero:1,platform:'mcp'}).g});
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
  const t1=node('box',44,18,120,n2('Monarch · Gmail · Drive · YT','',8.8),{h:26,z:'src',platform:'airflow'});o+=t1.g;
  o+=T(176,24,'EAI_MONARCH · EAI_GMAIL',{sz:fs(7.8),mono:1,c:Z.src.k,a:'start'})+T(176,35,'pipeline_load_window · drive_poll',{sz:fs(7.8),c:C.sub,a:'start'})+T(176,46,'MyChart FHIR → HOUSEHOLD_HEALTH',{sz:fs(7.8),c:Z.data.k,a:'start'});
 o+=T(10,78,Pp.airflow.s,{sz:fs(7.8),it:1,c:C.sub,a:'start'});
 o+=discNode(100,122,14,'','◇',{id:'gateway',glyphSz:11}).g;
 const ex=node('box',210,92,168,n2('Extract and land','',8.8,K.land,'src'),{h:28,z:'src'});o+=ex.g;
 const oc=node('box',210,132,168,n2(Pp.ocr.t,Pp.ocr.s,8.6,K.ocr,'src'),{h:28,z:'src'});o+=oc.g;
 o+=tag(154,108,'Monarch / Gmail',{light:1})+tag(154,158,'Drive docs',{light:1});
 const ld=node('box',232,210,144,n2('RAW and CORE','',8.8,K.raw+', '+K.core,'data'),{h:34,z:'data',platform:'snowflake'});o+=ld.g;
 const db=node('box',140,216,78,n2(Pp.dbt.t,'',8.6),{h:26,z:'data',platform:'dbt'});o+=db.g;
 const ts=node('box',78,216,48,n2('Tests','',8.6),{h:26});o+=ts.g;
 o+=discNode(34,229,10,'', '●',{id:'end',glyphSz:8}).g+T(12,250,'ANALYTICS,',{sz:fs(7.8),c:C.sub,a:'start'})+T(12,261,'MCP servers',{sz:fs(7.8),c:C.sub,a:'start'})+T(140,268,K.dbt,{sz:fs(7.8),mono:1,c:Z.data.k,a:'start'})+legend(14,286,['src','data']);
 a+=arrow('M29,32 H42','c')+arrow(`M104,${18+t1.h} V106`,'c')+arrow('M114,122 H120 Q124,122 124,118 V108 Q124,104 128,104 H208','c')+arrow('M100,136 V150 Q100,154 104,154 H208','c');
 a+=arrow('M378,106 H384 Q388,106 388,110 V225 Q388,229 384,229 H378','c')+arrow(`M294,${132+oc.h} V208`,'c')+arrow('M232,227 H220','c')+arrow('M140,229 H128','c')+arrow('M78,229 H44','c');
 return svg(W,H,Pp.title,a+o)};

// 5. Security + warehouses/cost: roles/MCP/gate on top; warehouse + caps below (Issuing nodes).
export function dSec(D, inst=""){
  begin(true,inst);ID='s'+inst;const S=D.security,K=D.code;const W=520,H=360;let o='',a='';
  o+=lab(14,14,'Access')+lab(14,168,'Warehouses & cost');
  // Roles row
  (S.roles||[]).forEach((t,i)=>o+=node('pill',14+i*168,28,156,[{t,sz:8.4,w:500}],{h:26,z:'sec',id:`role-${i}`}).g);
  // MCP split
  o+=node('box',14,68,200,n2(S.write||'MCP_HOUSEHOLD_LEDGER','write path',8.6,K.writeSrv,'agent'),{h:44,z:'agent',hero:1,platform:'mcp'}).g;
  o+=node('box',230,68,200,n2(S.read?.t||'MCP_HOUSEHOLD_LEDGER_GROK',S.read?.s||'no write tools',8.6,K.readSrv,'agent'),{h:44,z:'agent',hero:1,platform:'mcp'}).g;
  o+=node('box',14,124,200,n2(S.gate?.t||'R_CORTEX_GATE',S.gate?.s||'sole CORTEX_USER',8.6),{h:28,z:'sec'}).g;
  o+=node('cyl',230,122,120,n2(S.cat?.[0]||'SV_*','',8.2,K.svDoc,'data'),{h:32,z:'data'}).g;
  o+=node('cyl',360,122,120,n2(S.cat?.[1]||'CS_*','',8.2,K.cs,'data'),{h:32,z:'data'}).g;
  a+=arrow('M114,54 V68','c')+arrow('M330,54 V68','c')+arrow('M114,112 V124','c')+arrow('M330,112 V122','c');
  // Warehouses
  const wh=S.warehouses||[];
  wh.forEach((w,i)=>{
    const x=14+i*126;
    o+=node('box',x,184,118,n2(w.t,w.s,8.4),{h:48,z:i===2?'agent':'data',hero:i<2,id:`wh-${i}`,platform:i===2?'mcp':'snowflake'}).g;
    if(i)a+=arrow(`M${x-8},208 H${x}`,'');
  });
  // Cost controls
  const cost=S.cost||[];
  cost.forEach((c,i)=>{
    const x=14+i*126;
    o+=node('pill',x,250,118,[{t:c.t,sz:8,w:600},{t:c.s,sz:7.6,sub:1}],{h:36,z:'sec',id:`cost-${i}`}).g;
  });
  a+=arrow('M248,232 V250','d');
  o+=legend(14,308,['sec','agent','data']);
  o+=T(14,338,'No dollar spend shown — monitors, budgets, and batching only',{sz:fs(8),c:C.sub,a:'start'});
  return svg(W,H,S.title||'Roles, warehouses, and cost controls',a+o)};

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

/** Liam ERD Style A (green) — full per-DB table inventory; matches Style A sample chrome. */
export function dDatabase(D, dbId, inst=""){
  begin(false,inst);ID='liam'+inst+(dbId||'ledger');
  const model=ERD_MODELS[dbId]||ERD_MODELS.ledger;
  if(!model) return '';
  const isTile=String(inst||'').includes('tile')||inst==='t'||inst==='tile';
  // Tile: denser so the green Liam chrome fills the Domains thumbnail (not Issuing blue).
  const scale=isTile?0.55:0.92;
  // Single-line header (name LEFT + schema RIGHT) — matches Style A sample.
  const headH=Math.round(36*scale), rowH=Math.round(26*scale);
  const padX=Math.round(12*scale);
  const nameSz=Math.max(8.5,11*scale), typeSz=Math.max(7.5,10*scale), schemaSz=Math.max(7,9*scale);
  const tables={}; let maxX=0,maxY=0; let o='',a='';
  o+=T(20,Math.round(18*scale),model.title,{sz:Math.max(11,14*scale),w:600,a:'start',c:LIAM.ink});
  o+=T(20,Math.round(34*scale),model.subtitle,{sz:Math.max(8,10*scale),c:LIAM.schema,a:'start'});

  const tableGeom=(t)=>{
    const w=Math.round((t.w||220)*scale);
    const x=Math.round(t.x*scale)+4;
    const y=Math.round(t.y*scale)+Math.round(8*scale);
    const h=headH+t.cols.length*rowH+Math.round(4*scale);
    return {x,y,w,h};
  };

  const colY=(t,colName)=>{
    const g=tableGeom(t);
    const idx=t.cols.findIndex(c=>c.name===colName);
    const i=idx<0?0:idx;
    return g.y+headH+rowH*i+rowH/2;
  };

  model.tables.forEach((t)=>{tables[t.id]=t;});
  (model.edges||[]).forEach((e)=>{
    const aT=tables[e.from], bT=tables[e.to];
    if(!aT||!bT) return;
    const ag=tableGeom(aT), bg=tableGeom(bT);
    const y1=colY(aT,e.fromCol), y2=colY(bT,e.toCol);
    let x1,x2;
    if(ag.x+ag.w+8 < bg.x){ x1=ag.x+ag.w; x2=bg.x; }
    else if(bg.x+bg.w+8 < ag.x){ x1=ag.x; x2=bg.x+bg.w; }
    else { x1=ag.x+ag.w/2; x2=bg.x+bg.w/2; }
    // Orthogonal-ish cubic like Style A (no Issuing arrowheads, Liam gray connectors)
    const mx=(x1+x2)/2;
    a+=`<path class="edge" d="M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}" fill="none" stroke="${LIAM.conn}" stroke-width="${Math.max(1.1,1.25*scale)}"/>`;
  });

  model.tables.forEach((t,ti)=>{
    const {x,y,w,h}=tableGeom(t);
    maxX=Math.max(maxX,x+w); maxY=Math.max(maxY,y+h);
    const hi=ti===0;
    let g=`<g class="nd" tabindex="0" role="button" data-node="${esc(t.name)}" data-b="${x},${y},${w},${h},10,6" data-shape="box">`;
    g+=`<g class="nd-face" filter="url(#${ID}sh)"><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" fill="#fff" stroke="${hi?LIAM.green:LIAM.border}" stroke-width="${hi?2:1}"/></g>`;
    g+=`<rect x="${x}" y="${y}" width="${w}" height="${headH}" rx="8" fill="${LIAM.header}"/>`;
    g+=`<rect x="${x}" y="${y+headH-8}" width="${w}" height="8" fill="${LIAM.header}"/>`;
    g+=`<path d="M${x},${y+headH} H${x+w}" stroke="${LIAM.border}"/>`;
    const schemaW=meas(t.schema,schemaSz,500,0,1)/0.9+8;
    const nameMax=Math.max(48,w-2*padX-schemaW-6);
    const nameLines=wrapW(t.name,nameMax,nameSz,600,0,1).slice(0,1);
    g+=T(x+padX,y+headH/2,nameLines[0]||t.name,{sz:nameSz,w:600,a:'start',c:LIAM.ink,mono:1});
    g+=T(x+w-padX,y+headH/2,t.schema,{sz:schemaSz,c:LIAM.schema,a:'end',mono:1});
    t.cols.forEach((c,i)=>{
      const yy=y+headH+i*rowH;
      if(i) g+=`<path d="M${x},${yy} H${x+w}" stroke="${LIAM.border}"/>`;
      if(hi && (c.key==='pk'||c.key==='fk')) g+=`<rect x="${x}" y="${yy}" width="${w}" height="${rowH}" fill="${LIAM.soft}"/>`;
      const ix=x+padX+4, iy=yy+rowH/2;
      if(c.key==='pk'){
        g+=`<circle cx="${ix}" cy="${iy}" r="${5*Math.min(1,scale+0.15)}" fill="none" stroke="${LIAM.green}" stroke-width="1.3"/><circle cx="${ix}" cy="${iy}" r="${1.8*Math.min(1,scale+0.15)}" fill="${LIAM.green}"/>`;
      } else if(c.key==='fk'){
        g+=`<path d="M${ix-4},${iy-3} h5 a3,3 0 0 1 0,6 h-3" fill="none" stroke="${LIAM.green}" stroke-width="1.3"/>`;
      } else {
        g+=`<circle cx="${ix}" cy="${iy}" r="2.2" fill="${LIAM.type}"/>`;
      }
      const typeLabel=String(c.type||'');
      const typeW=meas(typeLabel,typeSz,400,0,1)/0.9+6;
      const colNameMax=Math.max(40,w-padX-16-padX-typeW-4);
      const colName=(wrapW(c.name,colNameMax,nameSz-0.5,400,0,1)[0])||c.name;
      g+=T(x+padX+16,iy,colName,{sz:nameSz-0.5,a:'start',c:LIAM.ink,mono:1});
      g+=T(x+w-padX,iy,typeLabel,{sz:typeSz,a:'end',c:LIAM.type,mono:1});
    });
    // No platform logos on ERD table cards — logos BR are process-flow only
    o+=g+'</g>';
  });

  const legendY=maxY+Math.round(24*scale);
  o+=`<circle cx="28" cy="${legendY}" r="5" fill="none" stroke="${LIAM.green}" stroke-width="1.3"/><circle cx="28" cy="${legendY}" r="1.8" fill="${LIAM.green}"/>`;
  o+=T(40,legendY,'PK',{sz:Math.max(9,11*scale),c:LIAM.schema,a:'start'});
  o+=`<path d="M70,${legendY-3} h6 a3.2,3.2 0 0 1 0,6 h-3" fill="none" stroke="${LIAM.green}" stroke-width="1.3"/>`;
  o+=T(92,legendY,'FK',{sz:Math.max(9,11*scale),c:LIAM.schema,a:'start'});
  o+=T(130,legendY,'Liam light · metadata only · no row values',{sz:Math.max(9,10*scale),c:LIAM.schema,a:'start'});
  const fullW=Math.max(980, maxX+28);
  const fullH=legendY+Math.round(24*scale);
  // Domains tile thumbnail: crop to top-left Liam green chrome (not a tiny full-map / Issuing overview)
  if(isTile){
    const cropW=Math.min(fullW, 1020);
    const cropH=Math.min(fullH, 380);
    return svg(cropW,cropH,`${model.title} ERD`,a+o,'bx-arch--liam');
  }
  return svg(fullW,fullH,`${model.title} ERD`,a+o,'bx-arch--liam')};

export function dAirflow(D, inst=""){
  begin(true,inst);ID='af'+inst;const A=D.airflow;const W=420;let o='';
  o+=T(12,14,A.headline,{sz:fs(11),w:600,a:'start'})+T(12,27,'Office-PC Airflow pin · git ≠ unpaused',{sz:fs(7.8),c:C.sub,a:'start'});
  A.counts.forEach((c,i)=>{const x=12+i*134;o+=node('box',x,38,126,[{t:c.n,sz:fs(12),w:600,c:Z.src.k},{t:c.label,sz:fs(7.6),sub:1}],{h:30,z:'src',platform:'airflow'}).g});
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
   o+=node(i<2?'cyl':'box',36,y,308,n2(l.t,l.s,8.6),{h:nh,z,platform:i<4?'dbt':'airflow'}).g;
   if(i<L.length-1)a+=arrow(`M190,${y+nh} V${y+step-2}`,'c')});
  o+=legend(12,12+L.length*step+2,['data','src']);
  return svg(W,12+L.length*step+18,D.dbt.title,a+o)};

export function dRearch(inst=""){
  begin(true,inst);ID='r'+inst;let o='',a='';
  const plats=['github','dbt','airflow','snowflake'];
  ['Terraform','dbt','Airflow','Snowflake'].forEach((t,i)=>{const xx=12+i*88;o+=node(i==3?'cyl':'box',xx,16,72,n2(t,'',9),{h:32,e:i==3,platform:plats[i]}).g;if(i<3)a+=arrow(`M${xx+72},32 H${xx+86}`,'c')});
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
