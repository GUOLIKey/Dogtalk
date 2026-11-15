"""Plot saved measured confusion counts; never generates or estimates model scores."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scene_schema import IDS

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);a=p.parse_args();root=Path(a.results)
    result=json.loads((root/'metrics.json').read_text())
    for subset,schemes in result['subsets'].items():
        for name,m in schemes.items():
            if name=='original_eight' or m['confusion_matrix'] is None:continue
            counts=np.array(m['confusion_matrix']);fig,ax=plt.subplots(figsize=(9,7))
            im=ax.imshow(counts,cmap='Blues');fig.colorbar(im,ax=ax,label='Recording count')
            labels=[IDS[l] for l in m['labels']];ax.set_xticks(range(6),labels);ax.set_yticks(range(6),labels)
            for i in range(6):
                for j in range(6):ax.text(j,i,str(counts[i,j]),ha='center',va='center',color='white' if counts[i,j]>counts.max()*.55 else 'black')
            ax.set_xlabel('Predicted scene');ax.set_ylabel('Reference scene')
            title='Eight-class predictions mapped to six scenes' if name=='mapped_eight_to_six' else 'Retrained six-scene model'
            ax.set_title(title+'\n'+subset+f' | n={m["n"]} | Macro-F1={m["macro_f1"]:.4f}')
            fig.text(.1,.03,'S01 Alerting | S02 Food | S03 Play-related | S04 Rest | S05 Attention | S06 Shower',fontsize=9)
            fig.tight_layout(rect=(0,.05,1,1));fig.savefig(root/(subset+'_'+name+'_confusion.png'),dpi=160);plt.close(fig)

if __name__=='__main__':main()
