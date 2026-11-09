"""Bonus: exhaustive improving endpoint flips and shortlist swaps."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from Assignment2_23CS10012_greedy_task1 import LOW,HIGH,ROOT,BUDGETS,save_json,load_inputs,shortlist,rss
from Assignment2_23CS10012_paper_task2 import State,FixedBase

def improve(P,a,s,ids,pool,selected,endpoints,direction=1,tolerance=1e-8):
    start=time.perf_counter();current=a.copy();current[selected]=endpoints;selected=set(map(int,selected))
    state=State(P,current,s);initial=float(state.z.sum());moves=[];evaluations=0;scans=0
    while True:
        scans+=1;old=float(state.z.sum());best_gain=tolerance;best=None
        union=sorted(selected|set(map(int,pool)));base=FixedBase(state,union)
        for v in sorted(selected):
            endpoint=LOW if state.a[v]==HIGH else HIGH
            f=base.evaluate([v],[endpoint]);evaluations+=1;gain=direction*(f-old)
            if gain>best_gain:best_gain=gain;best=('flip',v,v,endpoint,f)
            for w in pool:
                w=int(w)
                if w in selected:continue
                for endpoint in (LOW,HIGH):
                    f=base.evaluate([v,w],[a[v],endpoint]);evaluations+=1;gain=direction*(f-old)
                    if gain>best_gain:best_gain=gain;best=('swap',v,w,endpoint,f)
        if best is None:break
        kind,v,w,endpoint,f=best
        if kind=='swap':
            state.accept(v,a[v]);selected.remove(v);selected.add(w)
        state.accept(w,endpoint);verification=state.verify()
        if direction*(float(state.z.sum())-old)<=tolerance:raise ArithmeticError('Accepted bonus move did not improve')
        moves.append(dict(kind=kind,removed=int(ids[v]) if kind=='swap' else None,inserted=int(ids[w]),alpha=endpoint,f=float(state.z.sum()),gain=best_gain,**verification))
    indices=sorted(selected)
    return dict(initial=initial,f=float(state.z.sum()),added_gain=direction*(float(state.z.sum())-initial),
                ids=ids[indices],indices=indices,alpha=state.a[indices],moves=moves,scans=scans,evaluations=evaluations,
                already_local=not moves,local_optimal_tolerance=tolerance,seconds=time.perf_counter()-start,peak_mib=rss())

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dataset',type=Path,default=ROOT/'data/facebook_combined.txt');parser.add_argument('--output',type=Path,default=ROOT/'results');parser.add_argument('--profile',choices=['U','P'],default='U');parser.add_argument('--direction',choices=['max','min'],default='max')
    args=parser.parse_args();ids,degree,P,a,profiles,meta=load_inputs(args.dataset);s=profiles[args.profile];direction=1 if args.direction=='max' else -1
    greedy=json.loads((args.output/f'fast_{args.profile}_{args.direction}_all.json').read_text());results={}
    for k in BUDGETS:
        rows=greedy['rows'][:k]
        results[str(k)]=improve(P,a,s,ids,shortlist(ids,degree),[r['index'] for r in rows],[r['alpha'] for r in rows],direction)
        save_json(args.output/f'bonus_{args.profile}_{args.direction}.json',dict(results=results,complete=k==BUDGETS[-1]))
if __name__=='__main__':main()
