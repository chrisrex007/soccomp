"""Task 3: exhaustive endpoint search on the required candidate pool."""
from __future__ import annotations
import argparse
from itertools import combinations,product
from pathlib import Path
import time
import numpy as np
from Assignment2_23CS10012_greedy_task1 import LOW,HIGH,ROOT,save_json,load_inputs,shortlist,rss
from Assignment2_23CS10012_paper_task2 import State,FixedBase,fast,baseline,random_trials

def exhaustive(P,a,s,ids,degree,output=None,pool_size=15,ks=(2,3)):
    start=time.perf_counter();state=State(P,a,s);initial=float(state.z.sum());n=len(a)
    top_degree=shortlist(ids,degree,pool_size)
    singleton=np.max(state.scores(np.arange(n)),axis=1)-initial
    top_gain=np.lexsort((ids,-singleton))[:pool_size]
    C=np.union1d(top_degree,top_gain);base=FixedBase(state,C);setup=time.perf_counter()-start;results={}
    for k in ks:
        tick=time.perf_counter();count=subsets=0;best=-np.inf;winner=None
        for S in combinations(C.tolist(),k):
            subsets+=1
            for values in product((LOW,HIGH),repeat=k):
                f=base.evaluate(S,values);count+=1
                if f>best:
                    best=f;winner=(f,S,values)
        results[str(k)]=dict(f=winner[0],indices=winner[1],ids=ids[list(winner[1])],alpha=winner[2],
                            configurations=count,subsets=subsets,seconds=time.perf_counter()-tick,
                            total_seconds=time.perf_counter()-start,peak_mib=rss())
    result=dict(initial=initial,C_indices=C,C_ids=ids[C],top_degree=ids[top_degree],top_gain=ids[top_gain],
                overlap=len(np.intersect1d(top_degree,top_gain)),pool_size=len(C),setup_seconds=setup,
                references=results,complete=True)
    if output:save_json(output,result)
    return result,C

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dataset',type=Path,default=ROOT/'data/facebook_combined.txt');parser.add_argument('--output',type=Path,default=ROOT/'results');parser.add_argument('--profile',choices=['U','P'],default='U');parser.add_argument('--restricted-comparisons',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    ids,degree,P,a,profiles,meta=load_inputs(args.dataset);s=profiles[args.profile]
    result,C=exhaustive(P,a,s,ids,degree,args.output/f'groundtruth_{args.profile}.json')
    if args.restricted_comparisons:
        fast(P,a,s,ids,C,3,out=args.output/f'fast_{args.profile}_max_C.json')
        for kind in ('I','II'):baseline(P,a,s,ids,degree,kind,3,pool=C,out=args.output/f'{kind}_{args.profile}_max_C.json')
        random_trials(P,a,s,ids,3,pool=C,pool_code=2,out=args.output/f'random_{args.profile}_max_C.json')
if __name__=='__main__':main()
