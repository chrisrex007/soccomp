"""Task 2: rank-one greedy, baselines, non-convex local search and witnesses."""
from __future__ import annotations
import argparse
from itertools import product
from pathlib import Path
import time
import numpy as np
from scipy import linalg
import networkx as nx
from Assignment2_23CS10012_greedy_task1 import (LOW,HIGH,ROOT,TIE,Solver,system,save_json,rss,rng,graph_matrix,load_inputs,shortlist,choose)

class State:
    """Mutable inverse and equilibrium; candidate scoring needs no new solve."""
    def __init__(self,P,a,s):
        self.P=P;self.a=a.copy();self.s=s;self.G=linalg.inv(system(P,a).toarray())
        self.z=self.G@(a*s);self.p=P@self.z;self.c=self.G.sum(axis=0)
        self.diag=np.einsum('ij,ji->i',P.toarray(),self.G)
    def scores(self,pool):
        pool=np.asarray(pool,dtype=int);delta=np.array([LOW,HIGH])[None,:]-self.a[pool,None]
        den=1+delta*self.diag[pool,None]
        if np.any(den<=0):raise ArithmeticError('Nonpositive Sherman-Morrison denominator')
        gains=delta*(self.s[pool]-self.p[pool])[:,None]*self.c[pool,None]/den
        return float(self.z.sum())+gains
    def accept(self,v,endpoint):
        delta=endpoint-self.a[v];g=self.G[:,v].copy();h=np.asarray(self.P.getrow(v)@self.G).ravel()
        den=1+delta*h[v]
        if den<=0:raise ArithmeticError('Nonpositive update denominator')
        beta=delta/den;Pg=self.P@g;res=self.s[v]-self.p[v];cv=self.c[v]
        self.z+=beta*res*g;self.p+=beta*res*Pg;self.c-=beta*cv*h;self.diag-=beta*Pg*h
        self.G-=beta*np.outer(g,h);self.a[v]=endpoint

class FixedBase:
    """Woodbury evaluations against an immutable inverse."""
    def __init__(self,state,pool=None):
        self.state=state;self.lookup=None
        if pool is not None:
            pool=np.asarray(pool,dtype=int)
            self.lookup={int(v):i for i,v in enumerate(pool)}
            self.H=np.asarray(state.P[pool,:]@state.G[:,pool])
    def evaluate(self,indices,values):
        state=self.state;S=np.array(indices,dtype=int)
        if not len(S):return float(state.z.sum())
        delta=np.asarray(values)-state.a[S]
        if self.lookup is None:H=np.asarray(state.P[S,:]@state.G[:,S])
        else:
            idx=[self.lookup[int(v)] for v in S];H=self.H[np.ix_(idx,idx)]
        K=np.eye(len(S))+delta[:,None]*H
        w=linalg.solve(K,delta*(state.s[S]-state.p[S]),check_finite=False)
        return float(state.z.sum()+state.c[S]@w)

def fast(P,a,s,ids,pool,k,direction=1,out=None):
    start=time.perf_counter();state=State(P,a,s);setup=time.perf_counter()-start;initial=float(state.z.sum());selected=[];rows=[]
    for step in range(1,k+1):
        remaining=[int(v) for v in pool if v not in selected];scores=state.scores(remaining)
        choices=[(v,e,float(scores[i,j])) for i,v in enumerate(remaining) for j,e in enumerate((LOW,HIGH))]
        v,e,_=choose(choices,direction);state.accept(v,e);selected.append(v)
        actual=float(state.z.sum())
        rows.append(dict(k=step,added=int(ids[v]),index=v,alpha=e,f=actual,
                         gain=direction*(actual-(rows[-1]['f'] if rows else initial)),seconds=time.perf_counter()-start,
                         setup_seconds=setup,peak_mib=rss()))
        if out:save_json(out,dict(initial=initial,rows=rows,complete=step==k))
    return dict(initial=initial,rows=rows,complete=True)

def baseline(P,a,s,ids,degree,kind,k=30,direction=1,pool=None,out=None):
    start=time.perf_counter();pool=np.arange(len(a)) if pool is None else np.asarray(pool)
    values=s if direction==1 else 1-s
    if kind=='I':score=values
    elif kind=='II':
        neighbor_sum=degree*(P@values)
        score=np.divide(values,neighbor_sum,out=np.where(values>0,np.inf,0.),where=neighbor_sum!=0)*degree/degree.sum()
    else:raise ValueError(kind)
    order=pool[np.lexsort((ids[pool],-score[pool]))];solver=Solver();initial=float(solver.solve(P,a,s).sum());current=a.copy();rows=[]
    for step,v in enumerate(order[:k],1):
        current[v]=HIGH;z=solver.solve(P,current,s);f=float(z.sum())
        rows.append(dict(k=step,added=int(ids[v]),index=int(v),alpha=HIGH,f=f,gain=direction*(f-(rows[-1]['f'] if rows else initial)),seconds=time.perf_counter()-start,peak_mib=rss(),total_solves=solver.calls))
    result=dict(initial=initial,rows=rows,complete=True)
    if out:save_json(out,result)
    return result

def random_trials(P,a,s,ids,k=30,direction=1,pool=None,pool_code=0,trials=20,out=None):
    start=time.perf_counter();state=State(P,a,s);setup=time.perf_counter()-start
    pool=np.arange(len(a)) if pool is None else np.array(pool);all_rows=[]
    for trial in range(trials):
        tick=time.perf_counter();order=rng(1000+100*pool_code+trial).permutation(pool)[:k];chosen=[];endpoints=[];rows=[]
        base=FixedBase(state,order)
        for step,v in enumerate(order,1):
            candidates=[(int(v),e,base.evaluate(chosen+[int(v)],endpoints+[e])) for e in (LOW,HIGH)]
            _,e,f=choose(candidates,direction);chosen.append(int(v));endpoints.append(e)
            rows.append(dict(k=step,added=int(ids[v]),index=int(v),alpha=e,f=f,seconds=time.perf_counter()-tick))
        all_rows.append(rows)
        if out:save_json(out,dict(initial=float(state.z.sum()),setup_seconds=setup,trials=all_rows,complete=False))
    rows=[]
    for j in range(k):
        fs=[r[j]['f'] for r in all_rows];times=[r[j]['seconds'] for r in all_rows]
        rows.append(dict(k=j+1,f=float(np.mean(fs)),std=float(np.std(fs,ddof=1)) if trials>1 else 0.,seconds=setup+sum(times),mean_trial_seconds=float(np.mean(times))))
    result=dict(initial=float(state.z.sum()),setup_seconds=setup,rows=rows,trials=all_rows,
                complete=True,total_seconds=time.perf_counter()-start,peak_mib=rss())
    if out:save_json(out,result)
    return result

LOCAL_TOL = 1e-5  # Absolute comparison tolerance for total equilibrium opinions.

def local_starts(a):
    return [('upper',np.full(len(a),HIGH)),('initial',a.copy()),
            ('x_random',1/rng(3000).uniform(1,1/LOW,len(a)))]

def local_search(P,s,a0,ids,direction):
    """Section 6.1 coordinate search, using s-Pz to support the upper bound 1.

    Maximization minimizes complemented opinions. A move must improve the
    computed total by more than TIE; no residual certificate is claimed.
    """
    start=time.perf_counter();a=np.array(a0,dtype=float,copy=True)
    opinions=1-s if direction==1 else s
    order=np.argsort(ids,kind='stable');solver=Solver();moves=0
    z=solver.solve(P,a,opinions)
    initial=float(len(s)-z.sum()) if direction==1 else float(z.sum())
    while True:
        difference=opinions-P@z
        for v in order:
            if difference[v]>0:endpoint=LOW
            elif difference[v]<0:endpoint=HIGH
            else:continue
            if a[v]==endpoint:continue
            trial=a.copy();trial[v]=endpoint;new_z=solver.solve(P,trial,opinions)
            if float(z.sum()-new_z.sum())<=TIE:continue
            a=trial;z=new_z;moves+=1;break
        else:break
    actual=solver.solve(P,a,s)
    return dict(initial=initial,f=float(actual.sum()),alpha=a,initial_alpha=a0,
                direction='max' if direction==1 else 'min',moves=moves,status='converged',
                endpoint_count=int(np.count_nonzero((a==LOW)|(a==HIGH))),
                seconds=time.perf_counter()-start,peak_mib=rss())

def unbudgeted(P,a,s,ids,direction,out=None):
    runs=[];starts=local_starts(a)
    result=dict(method='local',schema_version=2,runs=runs,objective_tolerance=LOCAL_TOL,complete=False)
    if out:save_json(out,result)
    for number,(name,a0) in enumerate(starts):
        if out:print(f'Local search: starting {name} ({number+1}/{len(starts)})',flush=True)
        run=local_search(P,s,a0,ids,direction);run.update(start=number,start_name=name);runs.append(run)
        if out:save_json(out,result)
        if out:print(f"Local search: {name}: {run['status']}, moves={run['moves']}, f={run['f']:.10f}",flush=True)
    result['start_spread']=float(np.ptp([run['f'] for run in runs]))
    result['optimum']=dict(f=runs[0]['f'],start_name='upper')
    result['complete']=bool(result['start_spread']<=LOCAL_TOL)
    if out:save_json(out,result)
    return result

def optimized_set(base,S,direction=1):
    choices=[(base.evaluate(S,values),values) for values in product((LOW,HIGH),repeat=len(S))]
    return max(choices,key=lambda x:direction*x[0])

def paper_example():
    ids,_,P=graph_matrix(nx.complete_graph([1,2,3]));a=np.full(3,.1);s=np.array([1.,0.,0.]);base=FixedBase(State(P,a,s))
    sets=[[],[0],[2],[0,1],[0,2],[0,1,2]];fs=[optimized_set(base,S)[0] for S in sets]
    expected=[1,29/11,2999/2009,86971/31009,86971/31009,2999/1001]
    assert np.max(abs(np.array(fs)-expected))<1e-10
    return dict(sets=[[int(ids[v]) for v in S] for S in sets],f=fs,
                nonsubmodular=dict(small=fs[4]-fs[1],large=fs[5]-fs[3]),
                nonsupermodular=dict(small=fs[2]-fs[0],large=fs[4]-fs[1]))

def witnesses(P,a,s,ids,limit=10000):
    base=FixedBase(State(P,a,s));generator=rng(4000);found={};cache={}
    def value(S):
        key=tuple(sorted(S))
        if key not in cache:cache[key]=optimized_set(base,list(key))
        return cache[key]
    for attempt in range(limit):
        sample=generator.choice(len(a),3,replace=False).tolist()
        A=[] if attempt%2==0 else [sample[0]];B=A+[sample[1]];v=sample[2]
        sets=[sorted(A),sorted(A+[v]),sorted(B),sorted(B+[v])];vals=[value(S)[0] for S in sets]
        small=vals[1]-vals[0];large=vals[3]-vals[2];gap=large-small
        kind='nonsubmodular' if gap>1e-6 else 'nonsupermodular' if gap < -1e-6 else None
        if kind and kind not in found:
            found[kind]=dict(A=ids[A],B=ids[B],v=int(ids[v]),sets=[ids[S] for S in sets],
                             assignments=[list(value(S)[1]) for S in sets],f=vals,
                             marginal_small=small,marginal_large=large,attempt=attempt+1)
        if len(found)==2:break
    return dict(witnesses=found,attempts=attempt+1,complete=len(found)==2)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=ROOT/'data/facebook_combined.txt');parser.add_argument('--output',type=Path,default=ROOT/'results')
    parser.add_argument('--method',choices=['fast','I','II','random','local','witness','example'],default='fast');parser.add_argument('--profile',choices=['U','P'],default='U');parser.add_argument('--direction',choices=['max','min'],default='max');parser.add_argument('--pool',choices=['all','shortlist'],default='all');parser.add_argument('--k',type=int,default=30)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.method=='local' and args.pool!='all':parser.error('Local search is unbudgeted and requires --pool all')
    if args.method=='example':save_json(args.output/'paper_example.json',paper_example());return
    ids,degree,P,a,profiles,meta=load_inputs(args.dataset);s=profiles[args.profile];direction=1 if args.direction=='max' else -1
    pool=np.arange(len(a)) if args.pool=='all' else shortlist(ids,degree)
    if args.method in ('fast','I','II','random') and not 1<=args.k<=len(pool):
        parser.error('Budget must be between 1 and the candidate count')
    filename=args.output/f'{args.method}_{args.profile}_{args.direction}_{args.pool}.json'
    if args.method=='fast':fast(P,a,s,ids,pool,args.k,direction,filename)
    elif args.method in ('I','II'):baseline(P,a,s,ids,degree,args.method,args.k,direction,pool,filename)
    elif args.method=='random':random_trials(P,a,s,ids,args.k,direction,pool,int(args.pool=='shortlist'),out=filename)
    elif args.method=='local':
        result=unbudgeted(P,a,s,ids,direction,filename)
        if not result['complete']:raise ArithmeticError(f'Local starts disagree; inspect {filename}')
    elif args.method=='witness':
        result=witnesses(P,a,s,ids);save_json(filename,result)
        if not result['complete']:raise ArithmeticError(f'Marginal witnesses not found; inspect {filename}')
if __name__=='__main__':main()
