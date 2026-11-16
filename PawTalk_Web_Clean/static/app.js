'use strict';
const $ = id => document.getElementById(id);
let session = null, status = {}, busy = false, objectUrl = null, mic = null, pendingJob = null, versionMismatch = false;
const presentation = window.PawTalkPresentation;
const statusNames = {possible:'待验证', supported:'有观察支持', excluded:'有反证排除',subsumed:'被具体目标涵盖'};
function el(tag, text, cls) {const n=document.createElement(tag); if(text!==undefined)n.textContent=text; if(cls)n.className=cls; return n;}
function toast(message) {$('toastMessage').textContent=message;$('toast').hidden=false;}
$('dismissToast').onclick=()=>{$('toast').hidden=true;};
function syncPresentation(s) {
 const ui=presentation.derive(s);
 $('pageTitle').textContent=ui.title;
 $('inputDetails').open=!ui.compactInput;
 $('dropzone').hidden=!!s;$('micPanel').hidden=!!s;
 $('modelDetails').hidden=!s;$('modelSummary').textContent=ui.models;
 $('reasoningDetails').hidden=!ui.showReasoning;
 $('conversationDetails').hidden=!s?.dialogue?.history?.length;
 if(s){$('intentCard').hidden=!ui.showIntent;if(!s.analysis.ready)$('modelDetails').open=true;}
}

async function api(path, data) {const r=await fetch(path,data===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});const out=await r.json();if(!r.ok)throw Error(out.error||'请求失败');return out;}
function controls() {
 $('talkView').setAttribute?.('aria-busy',String(busy));
 for(const id of ['experimentMode','preObservation','answerKind','audioFile','changeFile','generateIntents','startChat','answerBtn','unknownBtn','stopChat','llmSelect','source','revealBtn','nextBtn','exportBtn','chatAnswer','tabTalk','tabRecords'])$(id).disabled=busy;
 $('correctCompleted').disabled=busy||!session?.dialogue?.history?.length;
 $('correctAnswer').disabled=busy||!session?.dialogue?.history?.length||!!session?.result;$('regenerate').hidden=!session?.result||['concluded','coexisting_supported'].includes(session?.result?.result_status);$('regenerate').disabled=busy;
 $('resumeUpload').hidden=!pendingJob||busy;$('resumeUpload').disabled=busy;
 $('preObservation').disabled=busy||!!session?.dialogue;
 $('generateIntents').disabled=busy||!session?.analysis?.ready||!$('llmSelect').value||session?.contaminated;
 $('experimentMode').disabled=busy||!!session?.initial_intents?.length;
 $('llmSelect').disabled=busy||!!session?.initial_intents?.length;$('source').disabled=busy||!!session?.initial_intents?.length;
 $('micBtn').disabled=busy||!mic; $('dropzone').classList.toggle('busy',busy);
 for(const button of $('historyList').querySelectorAll('button'))button.disabled=busy;
}
async function task(message, fn) {if(versionMismatch){toast("网页与后端版本不一致，请关闭旧运行窗口后重新启动，并打开新窗口提供的地址。");return;}if(busy)return;busy=true;$('toast').hidden=true;$('working').textContent=message;$('working').hidden=false;controls();try{await fn();}catch(e){toast(e.message);}finally{busy=false;$('working').hidden=true;controls();}}
function table(target, scores, names={}) {const t=el('table'),head=el('tr');head.append(el('th','类别'),el('th','原始分数'));t.append(head);for(const [label,value] of Object.entries(scores||{})){const r=el('tr');r.append(el('td',names[label]||label),el('td',Number(value).toFixed(5)));t.append(r);}target.append(t);}
function bullets(target, values) {target.replaceChildren();const ul=el('ul');for(const v of values)ul.append(el('li',v));target.append(ul);}
function refs(items) {return (items||[]).map(x=>`${x.answer_id||'第 '+x.turn+' 轮'}：“${x.quote}”`);}
function reset() {$('toast').hidden=true;$('reasoningDetails').open=false;$('conversationDetails').open=false;session=null;$('preObservation').value='';$('uploadProgress').hidden=true;$('uploadProgress').textContent='';for(const id of ['audioBox','sceneCard','emotionCard','ruleCard','intentCard','chatCard','resultCard'])$(id).hidden=true;$('audioFile').value='';$('chatAnswer').value='';$('reference').textContent='';$('player').pause();$('player').removeAttribute('src');if(objectUrl)URL.revokeObjectURL(objectUrl);objectUrl=null;$('modelDetails').open=false;syncPresentation(null);controls();}
function render(s) {
 const previousQuestion=session?.dialogue?.decision?.question?.text, hadResult=!!session?.result;
 session=s;
 if(s.pre_observations?.length)$('preObservation').value=s.pre_observations.map(x=>x.answer).join('\n');
 $('requestQuestion').hidden=!s.request_understanding;$('requestQuestion').textContent=s.request_understanding?'你想了解：'+s.request_understanding.user_question:'';
 $('interimExplanation').hidden=!s.dialogue?.decision?.explanation||!!s.result;$('interimExplanation').textContent=s.dialogue?.decision?.explanation||'';
 $('audioBox').hidden=false;$('filename').textContent=s.filename;$('audioMeta').textContent=s.analysis?.context?.duration?`${s.analysis.context.duration.toFixed(1)} 秒`:'';
 const a=s.analysis,c=a.context,e=a.emotion;
 $('sceneCard').hidden=false;$('sceneStatus').textContent=c.available?'场景分析完成':c.message;
 if(c.diagnostic)$('sceneStatus').textContent += ` 文件：${c.diagnostic.filename}；索引 ${c.diagnostic.index_count} 条；索引同名 ${c.diagnostic.same_filename_in_index} 条。${c.diagnostic.note}`;
 $('scenePreferred').textContent=c.available?`分类器原始首选：${c.original_preferred.name} · 决策分数 ${c.original_preferred.score.toFixed(5)}`:'';
 $('sceneScores').replaceChildren();if(c.available)table($('sceneScores'),c.scores,c.labels);
 $('sceneSource').textContent=c.available?`${c.model} · ${c.source} · ${c.duration.toFixed(2)} 秒 · ${c.score_kind} · 标签 ${c.label_version} · 模型 ${c.model_version} · 缓存版本 ${c.cache_version||'不适用/未知'}`:'';
 $('emotionCard').hidden=false;$('emotionMessage').textContent=e.available?e.message:e.message||'情绪未完成';$('emotionAxes').replaceChildren();
 if(e.available)for(const [axis,name] of [['arousal','唤醒程度'],['valence','正负倾向']]){const x=e[axis],box=el('div',undefined,'emotionAxis');box.append(el('h3',name),el('p',`${x.score_kind==='svm_decision_function_ovr'?'分类器首选':'最高类'} ${x.label} · usable=${x.usable} · ${x.usability_note||(x.usable?'可作辅助线索':'未达到使用门槛，保留不确定')}`));table(box,x.scores);box.append(el('p',`分数最大项 ${x.top_score_label||x.label} · 分差=${x.margin.toFixed(4)} · agreement=${x.agreement==null?'不适用（整段推理）':x.agreement.toFixed(4)} · ${x.score_kind}`,'fine'));$('emotionAxes').append(box);}
 $('emotionSource').textContent=e.available?`${e.model} · ${e.source} · ${e.duration.toFixed(2)} 秒 · ${e.model_version||e.model_commit||'历史版本未知'} · ${e.recipe}`:'';
 $('ruleCard').hidden=false;const es=s.emotion_summary,rc=s.rule_context;
 $('quadrantSummary').textContent=es?`${es.quadrant_id||es.status} · ${es.name} · ${es.reason_codes.join(' / ')}`:'尚未生成派生摘要';
 $('ruleSummary').textContent=rc?`原始首选入口 ${rc.anchor_scene||'不可用'}；主规则 ${rc.primary_rule_id||'无'}；辅助 ${rc.matched_rule_ids.slice(1).join('、')||'无'}；回退 ${rc.fallback_id||'无'}；规则版本 ${rc.rules_version}`:'';
 $('ruleRaw').textContent=JSON.stringify({emotion_summary:es,rule_context:rc},null,2);
 $('analysisRaw').textContent=JSON.stringify(a,null,2);$('roundArchive').textContent=JSON.stringify({initial_candidates:s.initial_intents,previous_rounds:s.candidate_rounds||[],answer_revisions:s.dialogue?.answer_revisions||[],decision:s.dialogue?.decision,result:s.result},null,2);
 $('intentCard').hidden=false;const blocked=!a.ready||s.contaminated;$('blocked').hidden=!blocked;
 $('blocked').textContent=!a.ready?'两路分析尚未齐全。请按错误信息准备模型后重新上传；本次没有生成意图。':'这段录音曾查看标签，不能用于正常盲测问答，请换用未看标签的录音。';
 $('intentSetup').hidden=blocked||s.phase!=='analyzed';$('intentCandidates').replaceChildren();if(s.generation_reason)$('intentCandidates').append(el('p',s.generation_reason,'fine'));
 const states=new Map((s.dialogue?.states||[]).map(x=>[x.intent_id,x]));
 for(const i of s.initial_intents){const current=states.get(i.intent_id)||i,box=el('article',undefined,'intent');box.dataset.status=current.status;box.append(el('small',`${i.intent_id} · ${statusNames[current.status]}`),el('h3',i.title),el('p',i.description));
 const clues=el('ul');for(const x of i.supporting_signals)clues.append(el('li',`${x.source}/${x.label}：${x.explanation}`));box.append(clues,el('p','初始不确定：'+i.uncertainties.join('；'),'fine'));
 box.append(el('p','来源规则：'+(i.source_rule_ids||[]).join('、')+(i.outside_rule_pool_reason?'；池外依据：'+i.outside_rule_pool_reason:''),'fine'));if(current.subtype||i.subtype)box.append(el('p','已确认细化：'+JSON.stringify(current.subtype||i.subtype),'fine'));if(current.reason)box.append(el('p',current.reason));for(const quote of refs(current.evidence))box.append(el('blockquote',quote));$('intentCandidates').append(box);}
 $('startChat').hidden=blocked||s.phase!=='intents_ready'||!!s.dialogue||!!s.result;
 $('chatCard').hidden=!s.dialogue||!!s.result;$('chatLog').replaceChildren();
 if(s.dialogue){const d=s.dialogue;$('turnBadge').textContent=`${s.question_budget?.used??d.history.length} / ${d.max_questions} · 剩余 ${s.question_budget?.remaining??0} 问`;for(const h of d.history){const item=el('div',undefined,'chatTurn');item.append(el('strong',`${h.turn}. ${h.question}`),el('p',h.answer));$('chatLog').append(item);}$('questionBox').hidden=!!s.result||!d.decision.question;$('currentQuestion').textContent=d.decision.question?.text||'';}
 $('resultCard').hidden=!s.result;
 $('resultAlternatives').replaceChildren();
 if(s.result){const r=s.result;
 if(r.uncertain&&r.alternatives){$('resultAlternatives').append(el('h3',r.alternatives.length?'可能的解释':'暂时没有足够线索'));for(const x of r.alternatives){const box=el('article',undefined,'intent');box.append(el('h3',x.title),el('p',x.reason),el('p','仍缺少：'+(x.missing_information||[]).join('；'),'fine'));for(const quote of refs(x.evidence))box.append(el('blockquote',quote));$('resultAlternatives').append(box);}}
$('resultTitle').textContent=r.uncertain?'暂不确定':`可能是：${r.title}`;$('resultReason').textContent=presentation.resultSummary(r);bullets($('resultEvidence'),refs(r.supporting_observations).length?refs(r.supporting_observations):['尚无足以支持唯一结论的观察。']);
 $('resultClues').textContent=`场景原始首选：${c.original_preferred?.name||'不可用'}；情绪分类首选：${e.arousal?.label||'不可用'} / ${e.valence?.label||'不可用'}（usable=${e.arousal?.usable} / ${e.valence?.usable}）。完整分数与来源保留在上方及导出中，均是预测。`;
 bullets($('resultExcluded'),r.excluded_intents.length?r.excluded_intents.map(i=>`${i.title}：${i.reason}；${refs(i.evidence).join('；')}`):['没有候选获得明确排除依据。']);bullets($('resultUncertainty'),r.uncertainties);$('revealBtn').hidden=!s.has_reference;$('reference').textContent=s.scene_reference?`场景参考：${s.scene_reference}（不能作为意图标准答案）`:'';}
 syncPresentation(s);controls();
 if(s.result&&!hadResult)$('resultTitle').focus?.();
 else if(s.dialogue?.decision?.question?.text && previousQuestion!==s.dialogue.decision.question.text)$('currentQuestion').focus?.();
}
const JOB_STORAGE_KEY='pawtalk_pending_upload_scene_six_v1';
function rememberJob(id) {pendingJob=id;try{if(id)globalThis.localStorage?.setItem(JOB_STORAGE_KEY,id);else globalThis.localStorage?.removeItem(JOB_STORAGE_KEY);}catch{}controls();}
function storedJob() {try{return globalThis.localStorage?.getItem(JOB_STORAGE_KEY)||null;}catch{return null;}}
async function jobFetch(path, options={}) {
 const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),15000);
 try{const r=await fetch(path,{...options,signal:controller.signal});const out=await r.json();if(!r.ok)throw Error(out.error||'上传状态请求失败');return out;}
 catch(e){if(e.name==='AbortError')throw Error('状态连接超过 15 秒，后台任务可能仍在执行。可点击“恢复查看上传任务”。');throw e;}
 finally{clearTimeout(timer);}
}
function displayProgress(job) {
 $('modelDetails').hidden=!job.partial?.context;$('modelSummary').textContent='分析中';
 $('uploadProgress').hidden=false;$('uploadProgress').textContent=`${job.filename} · 已用 ${job.elapsed_seconds} 秒 · ${job.message}`;
 if(job.elapsed_seconds>=60&&['queued','running'].includes(job.state))$('uploadProgress').textContent+=' 首次模型加载可能较慢；这不是录音长度，也不表示已超时。可以刷新后恢复查看，不必重复上传。';
 const c=job.partial?.context;
 if(c){if(!c.available)$('modelDetails').open=true;$('sceneCard').hidden=false;$('sceneStatus').textContent=c.available?'场景分析已完成，情绪分支仍在处理。':c.message;
 if(c.diagnostic)$('sceneStatus').textContent+=` 文件：${c.diagnostic.filename}；索引 ${c.diagnostic.index_count} 条；同名 ${c.diagnostic.same_filename_in_index} 条。${c.diagnostic.note}`;
 $('scenePreferred').textContent=c.available?`原始首选：${c.original_preferred.name} · 决策分数 ${c.original_preferred.score.toFixed(5)}`:'';
 $('sceneScores').replaceChildren();if(c.available)table($('sceneScores'),c.scores,c.labels);$('sceneSource').textContent=c.available?c.source:'';}
}
async function watchUpload(id) {
 let failures=0;
 while(pendingJob===id){
  let job;
  try{job=await jobFetch('/api/upload/status?id='+encodeURIComponent(id));failures=0;}
  catch(e){failures++;$('uploadProgress').hidden=false;$('uploadProgress').textContent=`状态连接暂时中断，正在重试（${failures}/3）。后台推理不会因为网页断开而取消。`;
   if(failures>=3)throw Error(e.message+' 请点击恢复查看上传任务，或稍后在探索记录中查看。');
   await new Promise(r=>setTimeout(r,2000));continue;}
  displayProgress(job);
  if(job.state==='completed'){rememberJob(null);render(job.session);$('uploadProgress').textContent=`处理已结束，用时 ${job.elapsed_seconds} 秒。结果已保存到探索记录。`;return;}
  if(['failed','interrupted'].includes(job.state)){rememberJob(null);throw Error(job.message);}
  await new Promise(r=>setTimeout(r,1500));
 }
}
async function resumeUpload() {if(!pendingJob)return;await task('正在恢复上传任务状态…',()=>watchUpload(pendingJob));}
async function upload(file) {
 if(mic?.active)mic.end(true);
 await task('正在接收音频，分析将在后台执行…',async()=>{
  const observed=$('preObservation').value;reset();$('preObservation').value=observed;objectUrl=URL.createObjectURL(file);$('player').src=objectUrl;$('player').hidden=false;
  const job=await jobFetch('/api/upload/start',{method:'POST',headers:{'X-Filename':encodeURIComponent(file.name||'microphone.webm')},body:file});
  rememberJob(job.job_id);displayProgress(job);await watchUpload(job.job_id);
 });
}
async function interview(path,extra={}) {const updated=await api(path,{id:session.id,revision:session.dialogue?.revision,...extra});render(updated);$('chatAnswer').value='';}
async function history() {const out=await api('/api/history');$('stats').textContent=`完成 ${out.intent_summary.completed} 次 · 暂定结论 ${out.intent_summary.concluded} 次 · 不确定 ${out.intent_summary.uncertain} 次`;$('legacyStats').textContent=JSON.stringify({legacy_scene_summary:out.legacy_scene_summary,intent_summary:out.intent_summary},null,2);$('historyList').replaceChildren();if(!out.records.length)$('historyList').append(el('p','还没有记录','empty'));for(const r of out.records){const card=el('article',undefined,'card');card.append(el('h3',r.filename),el('p',new Date(r.created*1000).toLocaleString()),el('p',r.legacy?'旧版记录':(r.result?.uncertain?'暂不确定':r.result?.title||presentation.phaseName(r.phase))));if(!r.legacy){const b=el('button','查看记录','secondary');b.onclick=()=>{if(busy)return;reset();$('player').hidden=true;render(r);view(false);};card.append(b);}else{const pre=el('pre',JSON.stringify({label_version:r.label_version,analysis:r.analysis,result:r.result},null,2));const detail=el('details');detail.append(el('summary','旧版结果'),pre);card.append(detail);}$('historyList').append(card);}}
function view(records) {if(records&&mic?.active)mic.end(true);$('tabRecords').setAttribute?.('aria-current',records?'page':'false');$('tabTalk').setAttribute?.('aria-current',records?'false':'page');$('recordsView').hidden=!records;$('talkView').hidden=records;$('tabRecords').classList.toggle('active',records);$('tabTalk').classList.toggle('active',!records);}
$('audioFile').onchange=e=>{if(e.target.files[0])upload(e.target.files[0]);};
$('dropzone').ondragover=e=>e.preventDefault();$('dropzone').ondrop=e=>{e.preventDefault();if(e.dataTransfer.files[0])upload(e.dataTransfer.files[0]);};$('dropzone').onkeydown=e=>{if(['Enter',' '].includes(e.key)&&!busy){e.preventDefault();$('audioFile').click();}};
$('resumeUpload').onclick=resumeUpload;
function newAnalysis(){reset();$('dropzone').focus?.();}
$('changeFile').onclick=newAnalysis;$('nextBtn').onclick=newAnalysis;$('llmSelect').onchange=controls;
$('generateIntents').onclick=()=>task('DeepSeek 正在理解你的描述与问题，综合声音线索…',()=>interview('/api/intents/generate',{flow:'context_first',model:$('llmSelect').value,source:$('source').value,experiment_mode:$('experimentMode').value,pre_observation:$('preObservation').value}));
$('correctAnswer').onclick=()=>task('正在重评更正后的观察…',async()=>{const last=session.dialogue.history.at(-1);await interview('/api/chat/correct',{answer_id:last.answer_id,answer:$('chatAnswer').value});});
$('correctCompleted').onclick=()=>task('正在重评更正后的观察…',()=>interview('/api/chat/correct',{answer_id:session.dialogue.history.at(-1).answer_id,answer:$('correctionText').value}));
$('regenerate').onclick=()=>task('正在创建新的候选轮次…',()=>interview('/api/intents/regenerate'));
$('startChat').onclick=()=>task('DeepSeek 正在生成第一问…',()=>interview('/api/chat/start'));
$('answerBtn').onclick=()=>task('DeepSeek 正在结合全部历史评估三个意图…',()=>interview('/api/chat/answer',{answer:$('chatAnswer').value,kind:$('answerKind').value}));
$('unknownBtn').onclick=()=>task('正在保留未知观察并准备下一步…',()=>interview('/api/chat/answer',{answer:'不知道 / 没观察到',kind:'unknown'}));
$('stopChat').onclick=()=>task('正在保存不确定结论…',()=>interview('/api/chat/stop'));
$('revealBtn').onclick=()=>task('读取场景参考标签…',async()=>{const r=await api('/api/reveal',{id:session.id});$('reference').textContent=`场景参考：${r.scene_reference}。${r.note}`;});
$('tabTalk').onclick=()=>view(false);$('tabRecords').onclick=()=>task('读取记录…',async()=>{await history();view(true);});
$('exportBtn').onclick=()=>task('导出记录…',async()=>{const out=await api('/api/export'),url=URL.createObjectURL(new Blob([JSON.stringify(out,null,2)],{type:'application/json'})),a=el('a');a.href=url;a.download='PawTalk_scene_six_v1_records.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
if(navigator.mediaDevices?.getUserMedia&&window.MediaRecorder&&window.HoldRecorder){mic=new HoldRecorder({getUserMedia:c=>navigator.mediaDevices.getUserMedia(c),Recorder:MediaRecorder,onState:s=>{$('micBtn').classList.toggle('recording',s==='recording');$('micBtn').setAttribute?.('aria-label',s==='recording'?'正在录音，松开结束':'按住录音');$('micHint').textContent={idle:'',requesting:'等待麦克风许可…',recording:'正在录音，松开结束。',stopping:'正在保存录音…'}[s];$('cancelMic').hidden=s==='idle';},onError:e=>toast('麦克风：'+e.message),onBlob:b=>upload(new File([b],`microphone.${b.type.includes('mp4')?'m4a':b.type.includes('ogg')?'ogg':'webm'}`,{type:b.type}))});
 $('micBtn').onpointerdown=e=>{if(busy)return;e.preventDefault();$('micBtn').setPointerCapture(e.pointerId);mic.begin();};$('micBtn').onpointerup=()=>mic.end();$('micBtn').onpointercancel=()=>mic.end(true);$('micBtn').onkeydown=e=>{if([' ','Enter'].includes(e.key)&&!e.repeat&&!busy){e.preventDefault();mic.begin();}};$('micBtn').onkeyup=e=>{if([' ','Enter'].includes(e.key))mic.end();};$('cancelMic').onclick=()=>mic.end(true);window.addEventListener('blur',()=>mic.end(true));document.addEventListener('visibilitychange',()=>{if(document.hidden)mic.end(true);});
}else $('micHint').textContent='此浏览器不支持录音，请上传文件。';
(async()=>{try{status=await api('/api/status');const expected=document.body?.dataset?.appVersion;versionMismatch=!!expected&&expected!==status.app_version;$('versionWarning').hidden=!versionMismatch;$('versionWarning').textContent=versionMismatch?`网页是 ${expected}，后端仍是 ${status.app_version}。请关闭旧运行窗口，再启动 00，并打开它提供的新地址。当前暂停提交，避免使用旧问答逻辑。`:'';$('buildVersion').textContent=status.app_version;const needsSetup=!(status.live_audio.ready||status.audio_count>0)||!status.emotion.ready||!status.deepseek_configured;$('connectionDot').dataset.state=needsSetup?'attention':'ready';$('serviceNotice').hidden=!needsSetup;$('serviceNotice').textContent=needsSetup?'部分服务尚未就绪，请查看连接状态。':'';$('modelStatus').textContent=`场景：${status.audio_count} 条缓存；${status.live_audio.message}；${status.cache_status?.message||''}`;$('emotionStatus').textContent=status.emotion.message;$('llmStatus').textContent=status.deepseek_configured?'DeepSeek 已配置，实际调用时验证':status.llm_error||'DeepSeek 未配置：运行 04';$('llmSelect').replaceChildren();for(const model of status.llm_models){const option=el('option',model);option.value=model;$('llmSelect').append(option);}if(!status.llm_models.length){const option=el('option','先配置 DeepSeek');option.value='';$('llmSelect').append(option);}}catch(e){toast(e.message);}controls();pendingJob=storedJob();if(pendingJob)await resumeUpload();})();
