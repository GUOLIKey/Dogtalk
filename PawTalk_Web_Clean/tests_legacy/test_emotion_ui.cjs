// DOM contract checks, not a substitute for browser layout/device testing.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
class Node{constructor(){this.children=[];this.textContent='';this.hidden=false;this.value='';this.classList={toggle(){},add(){},remove(){}};}append(...nodes){this.children.push(...nodes);}replaceChildren(...nodes){this.children=nodes;}addEventListener(){}scrollIntoView(){}pause(){}removeAttribute(){}get text(){return this.textContent+this.children.map(x=>x.text||'').join('');}}
const html=fs.readFileSync(__dirname+'/static/index.html','utf8'),ids=[...html.matchAll(/id="([^"]+)"/g)].map(x=>x[1]);
assert.equal(new Set(ids).size,ids.length);const nodes=Object.fromEntries(ids.map(x=>[x,new Node()]));
let code=fs.readFileSync(__dirname+'/static/app.js','utf8');for(const x of code.matchAll(/\$\('([^']+)'\)/g))assert(nodes[x[1]],'Missing ID '+x[1]);
code=code.split('// Press-and-hold recording.')[0].replace('refreshStatus();history(false);setInterval(refreshStatus,30000);','');
const context=vm.createContext({document:{getElementById:id=>nodes[id],createElement:()=>new Node()},window:{},console,URL,clearTimeout,setTimeout});vm.runInContext(code,context);
context.sample={emotion:{enabled:false,available:false}};vm.runInContext('renderEmotion(sample)',context);assert.equal(nodes.emotionCard.hidden,true);
context.sample.emotion={enabled:true,available:false,message:'模型未准备'};vm.runInContext('renderEmotion(sample)',context);assert.equal(nodes.emotionCard.hidden,false);assert.equal(nodes.emotionMessage.textContent,'模型未准备');assert.equal(nodes.emotionDetails.hidden,true);
context.sample.emotion={enabled:true,available:true,message:'实验',arousal:{usable:false,display:'暂不确定',scores:{Low:.34,Medium:.33,High:.33}},valence:{usable:true,display:'正向',scores:{Negative:.1,Neutral:.1,Positive:.8}}};vm.runInContext('renderEmotion(sample)',context);assert.equal(nodes.emotionAxes.children.length,2);assert(nodes.emotionAxes.text.includes('暂不确定'));assert(nodes.emotionScores.text.includes('0.340'));
vm.runInContext("status={emotion:{ready:true}};session=null;busy=false;controls()",context);assert.equal(nodes.emotionToggle.disabled,false);
vm.runInContext("session={};controls()",context);assert.equal(nodes.emotionToggle.disabled,true);
context.review={arousal:{name:'低',basis:'observation',reason:'动作很少',evidence:[{turn:1,quote:'安静趴着'}]},valence:{name:'暂不确定',basis:'uncertain',reason:'不足',evidence:[]}};
vm.runInContext("renderEmotionReview($('finalEmotion'),review)",context);assert(nodes.finalEmotion.text.includes('根据回答作出的推断'));assert(nodes.finalEmotion.text.includes('安静趴着'));
vm.runInContext('renderModeStats({})',context);assert.equal(nodes.modeStats.children.length,3);
console.log('Emotion UI contracts passed: IDs, off/on/error, scores, locked toggle, evidence, grouped metrics.');
