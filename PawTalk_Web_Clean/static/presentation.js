/* Pure presentation state: no network, storage, DOM mutation or model inference. */
(function (root) {
  'use strict';
  const PHASE_NAMES = Object.freeze({analyzed:'待分析',intents_ready:'待补充',interview:'问答中',completed:'已完成',blocked:'分析未完成'});
  function derive(session) {
    const hasResult = !!session?.result;
    const hasDialogue = !!session?.dialogue;
    const hasCandidates = !!session?.initial_intents?.length;
    return {
      title: hasResult ? '分析结果' : hasDialogue ? '补充观察' : '新分析',
      compactInput: hasDialogue || hasCandidates || hasResult,
      showIntent: !!session && !hasDialogue && !hasResult,
      showReasoning: hasCandidates || !!session?.generation_reason || !!session?.candidate_rounds?.length,
      models: !session ? '' : session.analysis?.ready ? '已完成' : '未完成',
    };
  }
  function resultSummary(result) {
    const text=result.explanation || result.message || '';
    const coexist=(result.coexisting_intent_ids||[]).length > 1;
    return text + (coexist ? '\n也可能存在并存的意图。' : '');
  }
  const api=Object.freeze({derive,resultSummary,phaseName:phase=>PHASE_NAMES[phase]||'未完成'});
  if(typeof module!=='undefined' && module.exports)module.exports=api;
  else root.PawTalkPresentation=api;
})(globalThis);
