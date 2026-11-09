"""Task 2: rank-one greedy, original-paper baselines, PGD and witnesses."""
from __future__ import annotations
import argparse
from itertools import product
import time
import numpy as np
from scipy import linalg
from scipy.sparse.linalg import bicgstab, spsolve
import networkx as nx
from Assignment2_23CS10012_greedy_task1 import (LOW,HIGH,BUDGETS,ROOT,TIE,Solver,system,save_json,rss,rng,graph_matrix,load_inputs,shortlist,choose)

class State:
    """Mutable inverse and equilibrium. Every scoring operation is solve-free."""
    def __init__(self,P,a,s):
        self.P=P;self.a=a.copy();self.s=s;self.G=linalg.inv(system(P,a).toarray(),check_finite=True)
        self.z=self.G@(a*s);self.p=P@self.z;self.c=self.G.sum(axis=0)
        # Sparse-dense product only during setup; do not retain a second dense matrix.
        self.diag=np.einsum('ij,ji->i',P.toarray(),self.G)
        columns=np.unique(np.linspace(0,len(a)-1,min(8,len(a)),dtype=int))
        inverse_residual=system(P,a)@self.G[:,columns]-np.eye(len(a))[:,columns]
        if np.max(abs(inverse_residual))>1e-10:raise ArithmeticError('Inverse column residual check failed')
        self.solver=Solver();self.verifications=0;self.rebuilds=0
        self.verify()
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
    def verify(self):
        independent=self.solver.solve(self.P,self.a,self.s);self.verifications+=1
        error=float(np.max(abs(independent-self.z)))
        residual=float(np.max(abs(system(self.P,self.a)@self.z-self.a*self.s)))
        if error>2e-8 or residual>1e-10:raise ArithmeticError(f'Rank update drift: {error}, {residual}')
        return dict(max_error=error,residual=residual)

class FixedBase:
    """Woodbury evaluations against an immutable inverse; no large copy per trial."""
    def __init__(self,state,pool=None):
        self.state=state
        self.lookup=None
        if pool is not None:
            pool=np.asarray(pool,dtype=int)
            self.lookup={int(v):i for i,v in enumerate(pool)}
            self.H=np.asarray(state.P[pool,:]@state.G[:,pool])
    def evaluate(self,indices,values,vector=False):
        state=self.state;S=np.array(indices,dtype=int)
        if not len(S):return state.z.copy() if vector else float(state.z.sum())
        delta=np.asarray(values)-state.a[S]
        if self.lookup is None:
            H=np.asarray(state.P[S,:]@state.G[:,S])
        else:
            idx=[self.lookup[int(v)] for v in S];H=self.H[np.ix_(idx,idx)]
        K=np.eye(len(S))+delta[:,None]*H
        b=delta*(state.s[S]-state.p[S]);w=linalg.solve(K,b,check_finite=False)
        if np.max(abs(K@w-b))>1e-9:raise ArithmeticError('Small-system residual too large')
        if vector:return state.z+state.G[:,S]@w
        return float(state.z.sum()+state.c[S]@w)

def fast(P,a,s,ids,pool,k,direction=1,out=None):
    start=time.perf_counter();state=State(P,a,s);setup=time.perf_counter()-start;initial=float(state.z.sum());selected=[];rows=[]
    for step in range(1,k+1):
        remaining=[int(v) for v in pool if v not in selected];scores=state.scores(remaining)
        choices=[(v,e,float(scores[i,j])) for i,v in enumerate(remaining) for j,e in enumerate((LOW,HIGH))]
        v,e,f=choose(choices,direction);state.accept(v,e);diagnostics=state.verify();selected.append(v)
        actual=float(state.z.sum())
        if abs(actual-f)>1e-8:raise ArithmeticError('Candidate and accepted objective disagree')
        rows.append(dict(k=step,added=int(ids[v]),index=v,alpha=e,f=actual,
                         gain=direction*(actual-(rows[-1]['f'] if rows else initial)),seconds=time.perf_counter()-start,
                         setup_seconds=setup,peak_mib=rss(),verification_solves=state.verifications,**diagnostics))
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
    start=time.perf_counter();state=State(P,a,s);base=FixedBase(state);setup=time.perf_counter()-start
    pool=np.arange(len(a)) if pool is None else np.array(pool);all_rows=[]
    for trial in range(trials):
        tick=time.perf_counter();order=rng(1000+100*pool_code+trial).permutation(pool)[:k];chosen=[];endpoints=[];rows=[]
        base=FixedBase(state,order)
        for step,v in enumerate(order,1):
            candidates=[(int(v),e,base.evaluate(chosen+[int(v)],endpoints+[e])) for e in (LOW,HIGH)]
            _,e,f=choose(candidates,direction);chosen.append(int(v));endpoints.append(e)
            rows.append(dict(k=step,added=int(ids[v]),index=int(v),alpha=e,f=f,seconds=time.perf_counter()-tick,peak_mib=rss()))
        verification=a.copy();verification[chosen]=endpoints
        if abs(Solver().solve(P,verification,s).sum()-rows[-1]['f'])>1e-7:raise ArithmeticError('Random verification failed')
        rows[-1]['seconds']=time.perf_counter()-tick
        all_rows.append(rows)
        if out:save_json(out,dict(initial=float(state.z.sum()),setup_seconds=setup,trials=all_rows,complete=trial+1==trials))
    rows=[]
    for j in range(k):
        fs=[r[j]['f'] for r in all_rows];times=[r[j]['seconds'] for r in all_rows]
        rows.append(dict(k=j+1,f=float(np.mean(fs)),std=float(np.std(fs,ddof=1)) if trials>1 else 0.,seconds=setup+sum(times),mean_trial_seconds=float(np.mean(times)),peak_mib=rss()))
    result=dict(initial=float(state.z.sum()),setup_seconds=setup,rows=rows,trials=all_rows,complete=True,total_seconds=time.perf_counter()-start)
    if out:save_json(out,result)
    return result

def pg_value_gradient(P,s,x,gradient=True):
    a=1/x;solver=Solver();z=solver.solve(P,a,s);f=float(z.sum())
    if not gradient:return f,z
    # B=X*M, so B^T q=1 is M^T*(X q)=1; avoid forming a poorly scaled B.
    M=system(P,a);adjoint,info=bicgstab(M.T,np.ones(len(x)),rtol=2e-13,atol=2e-13,maxiter=10000)
    if info!=0 or np.max(abs(M.T@adjoint-1))>1e-9:adjoint=spsolve(M.T,np.ones(len(x)))
    q=adjoint/x;grad=-q*(z-P@z)
    return f,z,grad

def bellman_reference(P,s,direction):
    """Endpoint policy improvement, certified afterwards by Bellman residual."""
    start=time.perf_counter();solver=Solver();a=np.full(len(s),.5)
    for it in range(1000):
        z=solver.solve(P,a,s);p=P@z
        new=np.where(direction*(s-p)>=0,HIGH,LOW)
        if np.array_equal(new,a):break
        a=new
    else:raise ArithmeticError('Reference policy iteration did not stabilize')
    z=solver.solve(P,a,s);p=P@z
    alternatives=np.stack([LOW*s+(1-LOW)*p,s])
    H=np.max(alternatives,axis=0) if direction==1 else np.min(alternatives,axis=0)
    residual=float(np.max(abs(z-H)));bound=len(s)*residual/LOW
    if bound>1e-5:raise ArithmeticError(f'Insufficient reference accuracy: {bound}')
    return dict(f=float(z.sum()),alpha=a,bellman_residual=residual,objective_error_bound=bound,iterations=it+1,seconds=time.perf_counter()-start)

def projected_gradient(P,s,x0,direction,maxiter=10000):
    start=time.perf_counter();x=x0.copy();eta=1.;history=[];status='iteration_limit'
    f,z,grad=pg_value_gradient(P,s,x);g=-direction*grad
    for iteration in range(maxiter):
        eta=min(eta*2,1e12);F=-direction*f
        accepted=False
        for backtrack in range(80):
            y=np.clip(x-eta*g,1,1/LOW);d=y-x
            if np.max(abs(d))<1e-13:break
            fy,zy=pg_value_gradient(P,s,y,False)
            if -direction*fy <= F+1e-4*np.dot(g,d)+1e-11:
                accepted=True;break
            eta*=.5
        mapping=float(np.max(abs(x-np.clip(x-g,1,1/LOW))))
        if not accepted:
            status='projected_stationary' if mapping<1e-9 else 'line_search_failed';break
        oldg=g;oldx=x;x=y
        f,z,grad=pg_value_gradient(P,s,x);g=-direction*grad
        if iteration%25==0 or iteration==maxiter-1:
            history.append(dict(iteration=iteration+1,f=f,eta=eta,mapping=mapping,seconds=time.perf_counter()-start))
        # A Bellman residual checks global quality of this very PGD policy.
        p=P@z;H=np.maximum(LOW*s+(1-LOW)*p,s) if direction==1 else np.minimum(LOW*s+(1-LOW)*p,s)
        bound=len(s)*float(np.max(abs(H-z)))/LOW
        if bound<1e-5:status='reference_certified';break
        displacement=x-oldx;change=g-oldg;den=np.dot(displacement,change)
        if den>0:eta=float(np.clip(np.dot(displacement,displacement)/den,.0001,1e12))/2
    a=1/x;z=Solver().solve(P,a,s)
    return dict(f=float(z.sum()),alpha=a,iterations=iteration+1,status=status,history=history,
                endpoint_count=int(np.count_nonzero((abs(a-LOW)<1e-8)|(abs(a-HIGH)<1e-8))),seconds=time.perf_counter()-start,peak_mib=rss())

def unbudgeted(P,a,s,direction,out=None,maxiter=10000):
    reference=bellman_reference(P,s,direction)
    starts=[1/a,np.full(len(a),(1+1/LOW)/2),rng(3000).uniform(1,1/LOW,len(a))]
    runs=[]
    for number,x in enumerate(starts):
        result=projected_gradient(P,s,x,direction,maxiter);result['start']=number
        result['reference_gap']=direction*(reference['f']-result['f']);runs.append(result)
        if out:save_json(out,dict(reference=reference,runs=runs,complete=number==2))
    return dict(reference=reference,runs=runs,complete=True)

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
    state=State(P,a,s);base=FixedBase(state);generator=rng(4000);found={};cache={}
    def value(S):
        key=tuple(sorted(S))
        if key not in cache:cache[key]=optimized_set(base,list(key))
        return cache[key]
    for attempt in range(limit):
        # Both empty-to-singleton and singleton-to-pair comparisons.
        sample=generator.choice(len(a),3,replace=False).tolist()
        A=[] if attempt%2==0 else [sample[0]]; B=A+[sample[1]];v=sample[2]
        sets=[sorted(A),sorted(A+[v]),sorted(B),sorted(B+[v])];vals=[value(S)[0] for S in sets]
        small=vals[1]-vals[0];large=vals[3]-vals[2];gap=large-small
        kind='nonsubmodular' if gap>1e-6 else 'nonsupermodular' if gap < -1e-6 else None
        if kind and kind not in found:
            verified=[];assignments=[];error_bound=0.
            for S in sets:
                f,es=value(S);trial=a.copy();trial[S]=es;z_fresh=Solver().solve(P,trial,s);actual=float(z_fresh.sum())
                residual=float(np.max(abs(system(P,trial)@z_fresh-trial*s)))
                error_bound+=len(a)*(residual+10*np.finfo(float).eps)/LOW
                if abs(actual-f)>1e-7:raise ArithmeticError('Witness solve mismatch')
                verified.append(actual);assignments.append(list(es))
            verified_gap=(verified[3]-verified[2])-(verified[1]-verified[0])
            if abs(verified_gap)<=10*error_bound or (verified_gap>0)!=(kind=='nonsubmodular'):continue
            found[kind]=dict(numerical_error_bound=error_bound,A=ids[A],B=ids[B],v=int(ids[v]),sets=[ids[S] for S in sets],assignments=assignments,f=verified,
                             marginal_small=verified[1]-verified[0],marginal_large=verified[3]-verified[2],attempt=attempt+1)
        if len(found)==2:break
    return dict(witnesses=found,attempts=attempt+1,complete=len(found)==2)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dataset',type=__import__('pathlib').Path,default=ROOT/'data/facebook_combined.txt');parser.add_argument('--output',type=__import__('pathlib').Path,default=ROOT/'results')
    parser.add_argument('--method',choices=['fast','I','II','random','pgd','witness','example'],default='fast');parser.add_argument('--profile',choices=['U','P'],default='U');parser.add_argument('--direction',choices=['max','min'],default='max');parser.add_argument('--pool',choices=['all','shortlist'],default='all');parser.add_argument('--k',type=int,default=30)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.method=='example':save_json(args.output/'paper_example.json',paper_example());return
    ids,degree,P,a,profiles,meta=load_inputs(args.dataset);s=profiles[args.profile];direction=1 if args.direction=='max' else -1
    pool=np.arange(len(a)) if args.pool=='all' else shortlist(ids,degree)
    filename=args.output/f'{args.method}_{args.profile}_{args.direction}_{args.pool}.json'
    if args.method=='fast':fast(P,a,s,ids,pool,args.k,direction,filename)
    elif args.method in ('I','II'):baseline(P,a,s,ids,degree,args.method,args.k,direction,pool,filename)
    elif args.method=='random':random_trials(P,a,s,ids,args.k,direction,pool,int(args.pool=='shortlist'),out=filename)
    elif args.method=='pgd':unbudgeted(P,a,s,direction,filename)
    elif args.method=='witness':save_json(filename,witnesses(P,a,s,ids))
if __name__=='__main__':main()
