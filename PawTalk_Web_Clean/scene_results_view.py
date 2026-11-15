"""Readable local reports from measured metrics; never estimates missing scores."""
import argparse
import html
import json
from pathlib import Path
import webbrowser

HERE=Path(__file__).resolve().parent
RESULTS=HERE/'results'/'scene_six_v1'
TITLES={'original_eight':'原八分类模型（八类任务）','mapped_eight_to_six':'原八类预测映射六类（六类基线）','retrained_six':'重新训练六分类模型（新任务）'}

def fmt(value):return '未运行' if value is None else f'{value:.4f}'

def select_result(root=RESULTS):
    root=Path(root)
    completed=[p.parent for p in root.glob('trained_run_*/RUN_COMPLETE.json') if (p.parent/'metrics.json').is_file()]
    if completed:return max(completed,key=lambda p:(p/'RUN_COMPLETE.json').stat().st_mtime_ns)
    old=root/'historical_audit'
    if not (old/'metrics.json').is_file():raise FileNotFoundError('还没有评估文件，请先运行 09_TRAIN_SIX_AND_SHOW_F1.bat。')
    return old

def render(directory):
    from scene_schema import NAMES, IDS, LABEL_VERSION
    directory=Path(directory);data=json.loads((directory/'metrics.json').read_text(encoding='utf-8'))
    if data.get('label_version')!=LABEL_VERSION:raise ValueError('评估标签版本不匹配。')
    p=data['provenance'];kind=p.get('kind');synthetic=p.get('data_kind')=='synthetic_test_fixture'
    is_trained=kind=='paired_model_rerun' and p.get('data_kind')=='real_dev'
    if synthetic:banner='仅为合成测试数据；以下数字不代表真实数据集表现。'
    elif is_trained:banner='真实 DEV 同折比较已完成；六类模型结果见第三行。'
    else:banner='这是旧 OOF 预测的标签映射统计；真正六分类重训尚未完成。'
    primary=data['subsets']['primary_ge_1s'];all_dev=data['subsets']['all_dev']
    if is_trained and 'retrained_six' not in primary:raise ValueError('训练结果缺少六分类指标。')
    notes=[banner,'主指标：时长 >= 1 秒的 DEV OOF Macro-F1，不是训练集拟合分数。',
           '不同目标的八类和六类分数不能相减并声称原任务提升；六类基线与六类重训才用于同任务比较。',
           '没有意图真值，不把场景分数解释为意图准确率。不读取或评分锁定留出集。']
    if is_trained:
        audit=p.get('split_audit',{})
        notes += ['划分：'+str(audit.get('split_policy')),'原划分跨折关联组：'+str(audit.get('original_cross_fold_groups')),
                  '已提供犬只 ID 样本数：'+str(audit.get('dog_id_coverage'))+'；来源录音 ID 样本数：'+str(audit.get('source_recording_id_coverage'))]
        notes += audit.get('limitations',[])
    elif p.get('leakage_status'):notes.append(p['leakage_status'])
    def table(headers,rows):
        esc=lambda v:html.escape(str(v))
        return '<table><thead><tr>'+''.join('<th>'+esc(v)+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'
    text=['PawTalk 六类场景评估结果',*notes,'','主评估子集 >= 1 秒：']
    sections=['<h1>PawTalk 场景模型评估</h1>','<div class="notice">'+html.escape(banner)+'</div>','<ul>'+''.join('<li>'+html.escape(n)+'</li>' for n in notes[1:])+'</ul>']
    for name,sub in [('主评估：时长 ≥ 1 秒',primary),('补充评估：全部 DEV',all_dev)]:
        rows=[]
        for key,title in TITLES.items():
            m=sub.get(key,{})
            rows.append([title,m.get('n','—'),fmt(m.get('accuracy')),fmt(m.get('macro_f1'))])
            if sub is primary:text.append(f'{title}: Accuracy={fmt(m.get("accuracy"))}, Macro-F1={fmt(m.get("macro_f1"))}, n={m.get("n","—")}')
        sections+=['<h2>'+name+'</h2>',table(['方案','样本数','Accuracy','Macro-F1'],rows)]
    if is_trained:
        delta=primary['retrained_six']['macro_f1']-primary['mapped_eight_to_six']['macro_f1']
        line=f'同一六类目标、同一划分：重训相对映射基线的 Macro-F1 差值 = {delta:+.4f}（不是显著性结论）。'
        sections.append('<p>'+html.escape(line)+'</p>');text+=['',line]
    for key in ('mapped_eight_to_six','retrained_six'):
        if key not in primary:continue
        m=primary[key];sections.append('<h2>'+TITLES[key]+'：主子集明细</h2>')
        rows=[]
        for f in m['folds']:rows.append([f['fold'],f['n'],fmt(f['accuracy']),fmt(f['macro_f1'])])
        sections.append(table(['折','样本数','Accuracy','Macro-F1'],rows))
        fs=m['fold_summary'];sections.append('<p>各折 Macro-F1 均值 ± 样本标准差：'+fmt(fs['macro_f1']['mean'])+' ± '+fmt(fs['macro_f1']['std_sample'])+'；与合并 OOF 指标分别报告。</p>')
        rows=[]
        for label in m['labels']:
            c=m['per_class'][label];rows.append([IDS[label],NAMES[label],fmt(c['precision']),fmt(c['recall']),fmt(c['f1-score']),c['support']])
        sections.append(table(['ID','类别','Precision','Recall','F1','样本数'],rows))
        sections.append('<h3>混淆矩阵：行是真实场景，列是预测场景</h3>')
        if m['confusion_matrix'] is not None:sections.append(table(['真实 / 预测']+[IDS[l] for l in m['labels']],[[IDS[l],*r] for l,r in zip(m['labels'],m['confusion_matrix'])]))
    text+=['','详细结果目录：'+str(directory.resolve()),'本文件和 report.html 不包含 DeepSeek 密钥。']
    summary='\n'.join(text)+'\n';(directory/'summary.txt').write_text(summary,encoding='utf-8')
    document='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PawTalk 场景 F1</title><style>body{font-family:"Microsoft YaHei",Arial,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;color:#163e36;background:#fafbf8;line-height:1.7}table{border-collapse:collapse;width:100%;background:white;margin:16px 0}th,td{border:1px solid #d6dfda;padding:9px;text-align:left}th{background:#e7eeea}.notice{background:#fff1d6;border-left:5px solid #ac7018;padding:18px;font-weight:bold}h2{margin-top:38px}li{margin:6px 0}</style>'+''.join(sections)+'</html>'
    report=directory/'report.html';report.write_text(document,encoding='utf-8')
    return report,summary

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--results');parser.add_argument('--no-browser',action='store_true');a=parser.parse_args()
    report,summary=render(Path(a.results) if a.results else select_result());print(summary)
    if not a.no_browser:webbrowser.open(report.resolve().as_uri())

if __name__=='__main__':
    try:main()
    except Exception as exc:print('[FAILED] '+str(exc));raise SystemExit(1)
