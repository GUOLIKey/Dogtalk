const assert=require('node:assert/strict');
const HoldRecorder=require('./static/recorder.js');
function setup(extra={}){let tracksStopped=0,blobs=[],errors=[],states=[];const stream={getTracks:()=>[{stop:()=>tracksStopped++}]};class FakeRecorder{static isTypeSupported(t){return t.startsWith('audio/webm');}constructor(s,o){this.state='inactive';this.mimeType=o?.mimeType||'audio/webm';}start(){this.state='recording';}stop(){if(this.state==='inactive')return;this.state='inactive';this.ondataavailable?.({data:new Blob(['test audio'],{type:this.mimeType})});this.onstop?.();}}const m=new HoldRecorder({getUserMedia:async()=>stream,Recorder:FakeRecorder,onBlob:b=>blobs.push(b),onError:e=>errors.push(e),onState:s=>states.push(s),...extra});return {m,stream,blobs,errors,states,stopped:()=>tracksStopped};}
(async()=>{
 let t=setup();await t.m.begin();t.m.end();assert.equal(t.blobs.length,1);assert.equal(t.stopped(),1);assert.equal(t.m.state,'idle');t.m.end();assert.equal(t.blobs.length,1);
 t=setup();await t.m.begin();t.m.end(true);assert.equal(t.blobs.length,0);assert.equal(t.stopped(),1);
 let grant;t=setup({getUserMedia:()=>new Promise(resolve=>grant=resolve)});const pending=t.m.begin();t.m.end();grant(t.stream);await pending;assert.equal(t.blobs.length,0);assert.equal(t.stopped(),1);assert.equal(t.m.state,'idle');
 t=setup({getUserMedia:async()=>{throw Object.assign(Error('denied'),{name:'NotAllowedError'});}});await t.m.begin();assert.equal(t.errors.length,1);assert.equal(t.m.state,'idle');
 t=setup({Recorder:class{constructor(){throw Error('codec unavailable');}}});await t.m.begin();assert.equal(t.stopped(),1);assert.equal(t.errors.length,1);
 t=setup({maxMs:5});await t.m.begin();await new Promise(r=>setTimeout(r,20));assert.equal(t.blobs.length,1);assert.equal(t.stopped(),1);
 t=setup({maxBytes:1});await t.m.begin();t.m.end();assert.equal(t.blobs.length,0);assert.equal(t.errors.length,1);assert.equal(t.stopped(),1);
 t=setup();await t.m.begin();await t.m.begin();t.m.end();assert.equal(t.blobs.length,1);assert.equal(t.stopped(),1);
 console.log('8 microphone lifecycle checks passed: release, cancel, permission race, denial, codec failure, time limit, byte limit, duplicate press.');
})().catch(e=>{console.error(e);process.exitCode=1;});
