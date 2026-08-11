"""
GEdit-Bench 统计脚本（适配版）- 去掉 megfile 依赖

用法：
    python eval/gen/gedit/calculate_statistics.py \
        --model_name unimm --backbone gpt4o \
        --save_path /path/to/score_dir --language all
"""
import os, argparse, json, numpy as np, math, pandas as pd
from collections import defaultdict

GROUPS = ["background_change","color_alter","material_alter","motion_change",
          "ps_human","style_change","subject-add","subject-remove",
          "subject-replace","text_change","tone_transfer"]

def analyze_scores(save_path_dir, model_name, language):
    s, q, o, si, qi, oi = {}, {}, {}, {}, {}, {}
    for g in GROUPS:
        # 优先找 per-language 文件，fallback 到 all 文件（scoring 以 --instruction_language all 跑时生成）
        csv_path = os.path.join(save_path_dir, f"{model_name}_{g}_{language}_vie_score.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(save_path_dir, f"{model_name}_{g}_all_vie_score.csv")
        if not os.path.exists(csv_path):
            print(f"  Warning: missing {g}, skip"); continue
        df = pd.read_csv(csv_path)
        sl,ql,ol,sil,qil,oil = [],[],[],[],[],[]
        for _, row in df.iterrows():
            if language != "all" and row['instruction_language'] != language: continue
            sv,qv = float(row['sementics_score']), float(row['quality_score'])
            ov = math.sqrt(sv*qv)
            sl.append(sv); ql.append(qv); ol.append(ov)
            if row['intersection_exist']:
                sil.append(sv); qil.append(qv); oil.append(ov)
        s[g]=np.mean(sl) if sl else float('nan')
        q[g]=np.mean(ql) if ql else float('nan')
        o[g]=np.mean(ol) if ol else float('nan')
        si[g]=np.mean(sil) if sil else float('nan')
        qi[g]=np.mean(qil) if qil else float('nan')
        oi[g]=np.mean(oil) if oil else float('nan')
    valid=[g for g in GROUPS if g in s and not math.isnan(s[g])]
    for d in [s,q,o,si,qi,oi]:
        key='avg'; vals=[d[g] for g in valid] if valid else []
        d[key]=np.mean(vals) if vals else float('nan')
    return s,q,o,si,qi,oi

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--model_name",type=str,default="unimm")
    parser.add_argument("--backbone",type=str,default="gpt4o")
    parser.add_argument("--save_path",type=str,required=True)
    parser.add_argument("--language",type=str,default="all",choices=["all","en","cn"])
    parser.add_argument("--json-output", type=str, default=None)
    args=parser.parse_args()
    langs=["all","en","cn"] if args.language=="all" else [args.language]
    save_dir=os.path.join(args.save_path,args.model_name,args.backbone)
    summary = {}
    for lang in langs:
        print(f"\n{'='*10} {args.backbone} | {args.model_name} | {lang} {'='*10}")
        s,q,o,si,qi,oi=analyze_scores(save_dir,args.model_name,lang)
        print(f"\n{'group':32s}  SC     PQ     Overall")
        for g in GROUPS:
            if g in s: print(f"  {g:30s}  {s[g]:.3f}  {q[g]:.3f}  {o[g]:.3f}")
        print(f"  {'Average':30s}  {s.get('avg',float('nan')):.3f}  {q.get('avg',float('nan')):.3f}  {o.get('avg',float('nan')):.3f}")
        print(f"\nIntersection:")
        for g in GROUPS:
            if g in si: print(f"  {g:30s}  {si[g]:.3f}  {qi[g]:.3f}  {oi[g]:.3f}")
        print(f"  {'Average':30s}  {si.get('avg',float('nan')):.3f}  {qi.get('avg',float('nan')):.3f}  {oi.get('avg',float('nan')):.3f}")
        summary[lang] = {"sc": float(s["avg"]), "pq": float(q["avg"]),
                         "overall": float(o["avg"])}
    if args.json_output:
        output_path = os.path.abspath(args.json_output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, allow_nan=False)
        print(f"Machine-readable summary saved to {output_path}")
