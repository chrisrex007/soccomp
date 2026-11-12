"""Task 4: validation, reproducible experiment driver, analysis tables and plots.

Commands: checks, prepare, run, analyze. Use --help for paths.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
from itertools import product, combinations
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import urllib.request
import numpy as np
import networkx as nx
from Assignment2_23CS10012_greedy_task1 import (ROOT,LOW,HIGH,BUDGETS,Solver,system,save_json,load_inputs,graph_matrix,rng,naive,iterate,small_checks)
from Assignment2_23CS10012_paper_task2 import (State,FixedBase,fast,baseline,random_trials,local_search,local_starts,LOCAL_TOL,bellman_reference,paper_example,witnesses)
from Assignment2_23CS10012_groundtruth_task3 import exhaustive
from Assignment2_23CS10012_bonus_task4 import improve

PREFIX='Assignment2_23CS10012_'
SCENARIOS=[('U','max'),('P','max'),('U','min')]
METHODS=['fast','I','II','random']

def checks(output):
    """Independent small-graph checks. No Facebook results are synthesized here."""
    start=time.perf_counter();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    small_checks(output);save_json(output/'paper_example.json',paper_example());records=[]
    for graph in [nx.path_graph(6),nx.complete_graph(5)]:
        ids,degree,P=graph_matrix(graph);n=len(ids);a=rng(n).uniform(LOW,HIGH,n);s=rng(n+30).random(n);solver=Solver()
        z=solver.solve(P,a,s);iterated,steps=iterate(P,a,s)
        assert np.max(abs(z-iterated))<1e-8
        assert abs(solver.solve(P,a,1-s).sum()+z.sum()-n)<1e-10
        state=State(P,a,s);base=FixedBase(state,np.arange(n))
        for S in [(0,),(0,2),(0,2,4)]:
            for es in product((LOW,HIGH),repeat=len(S)):
                trial=a.copy();trial[list(S)]=es;direct=solver.solve(P,trial,s)
                assert abs(base.evaluate(S,es)-direct.sum())<1e-8
                assert np.max(abs(base.evaluate(S,es,True)-direct))<1e-8
        for v,e in [(0,HIGH),(2,LOW),(0,a[0]),(4,HIGH),(2,HIGH)]:
            state.accept(v,e);state.verify()
            independent=np.linalg.inv(system(P,state.a).toarray())
            assert np.max(abs(state.c-independent.sum(axis=0)))<1e-8
            assert np.max(abs(state.diag-np.diag(P@independent)))<1e-8
        slow=naive(P,a,s,ids,np.arange(n),3);quick=fast(P,a,s,ids,np.arange(n),3)
        assert slow['rows'][-1]['candidate_solves']==2*sum(n-i for i in range(3))
        for left,right in zip(slow['rows'],quick['rows']):
            assert (left['index'],left['alpha'])==(right['index'],right['alpha'])
            assert abs(left['f']-right['f'])<1e-8
        # Enumerate all endpoints using independent dense solves (no State or
        # local-search update code), for both objectives and all four starts.
        endpoint_values=[float(np.linalg.solve(system(P,np.array(es)).toarray(),np.array(es)*s).sum())
                         for es in product((LOW,HIGH),repeat=n)]
        local_runs=[]
        for direction in (1,-1):
            exact=max(endpoint_values) if direction==1 else min(endpoint_values)
            ref=bellman_reference(P,s,direction);assert abs(ref['f']-exact)<1e-8
            objectives=[]
            for name,a0 in local_starts(a):
                result=local_search(P,s,a0,ids,direction)
                gap=direction*(exact-result['f'])
                assert result['status']=='certified' and abs(gap)<=LOCAL_TOL, (n,direction,name,result['status'],gap)
                assert result['objective_error_bound']<=LOCAL_TOL
                assert np.all((result['alpha']>=LOW)&(result['alpha']<=HIGH))
                trial=a0.copy();previous=float(np.linalg.solve(system(P,trial).toarray(),trial*s).sum())
                for row in result['history']:
                    assert int(ids[row['index']])==row['node']
                    assert trial[row['index']]==row['alpha_before']
                    trial[row['index']]=row['alpha_after']
                    actual=float(np.linalg.solve(system(P,trial).toarray(),trial*s).sum())
                    assert abs(actual-row['f'])<1e-8
                    assert direction*(actual-previous)>0 and row['gain']>row['gain_uncertainty']
                    previous=actual
                assert np.array_equal(trial,result['alpha'])
                if name=='upper':
                    assert result['moves']<=n and result['endpoint_count']==n
                    assert len({row['node'] for row in result['history']})==result['moves']
                    assert all(row['alpha_before']==HIGH and row['alpha_after']==LOW for row in result['history'])
                complement=local_search(P,1-s,a0,ids,-direction)
                assert complement['status']=='certified'
                assert abs(complement['f']+result['f']-n)<LOCAL_TOL
                objectives.append(result['f'])
                local_runs.append(dict(direction=direction,start_name=name,f=result['f'],exact=exact,gap=gap,
                                       moves=result['moves'],status=result['status'],objective_error_bound=result['objective_error_bound']))
            assert np.ptp(objectives)<=LOCAL_TOL
        reference,C=exhaustive(P,a,s,ids,degree,pool_size=n)
        for k in (2,3):
            expected=__import__('math').comb(n,k)*2**k
            assert reference['references'][str(k)]['configurations']==expected
            direct_best=-np.inf
            for subset in combinations(range(n),k):
                for values in product((LOW,HIGH),repeat=k):
                    trial=a.copy();trial[list(subset)]=values
                    direct_best=max(direct_best,float(solver.solve(P,trial,s).sum()))
            assert abs(reference['references'][str(k)]['f']-direct_best)<1e-8
        for kind in ('I','II'):
            record=baseline(P,a,s,ids,degree,kind,k=3)
            assert all(row['alpha']==HIGH for row in record['rows'])
            scores=s if kind=='I' else degree/degree.sum()*s/(degree*(P@s))
            order=np.lexsort((ids,-scores))[:3]
            assert [r['index'] for r in record['rows']]==list(order)
            minimum=baseline(P,a,s,ids,degree,kind,k=3,direction=-1)
            complement=baseline(P,a,1-s,ids,degree,kind,k=3)
            assert [r['index'] for r in minimum['rows']]==[r['index'] for r in complement['rows']]
            assert abs(minimum['rows'][-1]['f']+complement['rows'][-1]['f']-n)<1e-8
        random_result=random_trials(P,a,s,ids,k=3,trials=20)
        assert len(random_result['trials'])==20
        for rows in random_result['trials']:
            assert len({r['index'] for r in rows})==3
            trial=a.copy()
            for row in rows:
                v=row['index'];values=[]
                for endpoint in (LOW,HIGH):
                    trial[v]=endpoint;values.append(float(solver.solve(P,trial,s).sum()))
                assert abs(row['f']-max(values))<1e-8
                trial[v]=row['alpha']
        rows=quick['rows'];bonus=improve(P,a,s,ids,np.arange(n),[r['index'] for r in rows],[r['alpha'] for r in rows])
        assert bonus['added_gain']>=-1e-9
        # Exhaustively check the final neighborhood via independent equilibrium solves.
        final=a.copy();chosen=bonus['indices'];final[chosen]=bonus['alpha'];f=solver.solve(P,final,s).sum()
        for v in chosen:
            for w in range(n):
                if w in chosen and w!=v:continue
                for e in (LOW,HIGH):
                    trial=final.copy();trial[v]=a[v];trial[w]=e
                    assert solver.solve(P,trial,s).sum()<=f+1e-8
        evidence=witnesses(P,a,s,ids,limit=200)
        for kind,witness in evidence['witnesses'].items():
            for S,es,f in zip(witness['sets'],witness['assignments'],witness['f']):
                trial=a.copy();trial[list(S)]=es
                assert abs(solver.solve(P,trial,s).sum()-f)<1e-8
            gap=witness['marginal_large']-witness['marginal_small']
            assert (gap>0 if kind=='nonsubmodular' else gap<0)
        records.append(dict(nodes=n,edges=graph.number_of_edges(),iteration_steps=steps,
                            naive_fast_match=True,local=local_runs,exhaustive=reference,bonus=bonus))
    local_edges=local_edge_checks()
    result=dict(passed=True,graphs=records,local_edge_checks=local_edges,seconds=time.perf_counter()-start,python=platform.python_version())
    save_json(output/'validation.json',result);print('All independent small-graph checks passed',flush=True)
    return result

def local_edge_checks():
    """Boundary, ties, original-ID ordering and non-convergence regressions."""
    ids,_,P=graph_matrix(nx.path_graph([10,30]));s=np.array([1.,0.]);upper=np.full(2,HIGH)
    # Here z=s at alpha=1: using s-z would falsely stop immediately.
    values={}
    for direction in (-1,1):
        result=local_search(P,s,upper,ids,direction)
        expected=LOW if direction==-1 else 2-LOW
        assert result['status']=='certified' and abs(result['f']-expected)<1e-10
        assert result['moves']==1 and result['history'][0]['node']==(10 if direction==-1 else 30)
        blocked=local_search(P,s,upper,ids,direction,max_moves=0)
        assert blocked['status']=='move_limit' and blocked['objective_error_bound']>LOCAL_TOL
        values['max' if direction==1 else 'min']=result['f']
        for constant in (0.,.5,1.):
            a0=np.array([.2,.7]);flat=local_search(P,np.full(2,constant),a0,ids,direction)
            assert flat['status']=='certified' and flat['moves']==0
            assert np.array_equal(flat['alpha'],a0) and abs(flat['f']-2*constant)<1e-10
    ids,_,P=graph_matrix(nx.path_graph([10,20,30]));s=np.array([1.,0.,1.]);a0=np.full(3,HIGH)
    first=local_search(P,s,a0,ids,-1);perm=np.array([2,0,1])
    second=local_search(P[perm,:][:,perm],s[perm],a0[perm],ids[perm],-1)
    assert first['status']==second['status']=='certified'
    assert first['history'][0]['node']==10
    assert [row['node'] for row in first['history']]==[row['node'] for row in second['history']]
    assert abs(first['f']-second['f'])<1e-10
    return dict(two_node=values,upper_boundary=True,constant_ties=True,original_id_order=True,move_limit=True)

def prepare(dataset):
    dataset=Path(dataset);dataset.parent.mkdir(parents=True,exist_ok=True)
    if dataset.exists():load_inputs(dataset);return
    url='https://snap.stanford.edu/data/facebook_combined.txt.gz'
    with urllib.request.urlopen(url,timeout=60) as response:compressed=response.read()
    raw=gzip.decompress(compressed);tmp=dataset.with_suffix('.tmp');tmp.write_bytes(raw)
    load_inputs(tmp);tmp.replace(dataset)
    save_json(dataset.parent/'source.json',dict(url=url,compressed_sha256=hashlib.sha256(compressed).hexdigest(),raw_sha256=hashlib.sha256(raw).hexdigest()))

def experiment_hash(dataset):
    h=hashlib.sha256(Path(dataset).read_bytes())
    for path in sorted(ROOT.glob(PREFIX+'*.py')):h.update(path.read_bytes())
    return h.hexdigest()

def run(dataset,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    ids,degree,P,a,profiles,meta=load_inputs(dataset);fingerprint=experiment_hash(dataset)
    manifest=output/'manifest.json'
    if manifest.exists():
        previous=json.loads(manifest.read_text())
        if previous.get('experiment_hash') not in (None,fingerprint):raise RuntimeError('Inputs/source changed: choose a new results directory')
    meta['experiment_hash']=fingerprint;save_json(manifest,meta)
    np.savez(output/'inputs.npz',ids=ids,alpha=a,U=profiles['U'],P=profiles['P'])
    checks(output/'checks')
    equilibrium={}
    for profile,s in profiles.items():
        z=Solver().solve(P,a,s);zi,steps=iterate(P,a,s);error=float(np.max(abs(z-zi)))
        assert error<1e-8
        equilibrium[profile]=dict(f=float(z.sum()),iteration_error=error,steps=steps)
    save_json(output/'equilibrium.json',equilibrium)
    commands=read(output,'commands.json') if (output/'commands.json').exists() else []
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR='/tmp/soccomp-matplotlib',XDG_CACHE_HOME='/tmp/soccomp-cache')
    def worker(script,extra,expected):
        cmd=[sys.executable,str(ROOT/(PREFIX+script+'.py')),'--dataset',str(Path(dataset).resolve()),'--output',str(output.resolve())]+extra
        target=output/expected
        if target.exists() and json.loads(target.read_text()).get('complete'):
            commands.append(dict(command=cmd,reused=True));return
        print('Running',script,' '.join(extra),flush=True);tick=time.perf_counter()
        subprocess.run(cmd,env=env,check=True)
        if not target.exists() or not json.loads(target.read_text()).get('complete'):raise RuntimeError(f'Worker did not finish {expected}')
        commands.append(dict(command=cmd,reused=False,wall_seconds=time.perf_counter()-tick));save_json(output/'commands.json',commands)
    for profile in ('U','P'):
        worker('greedy_task1',['--profile',profile],f'naive_{profile}.json')
        worker('paper_task2',['--method','fast','--profile',profile,'--pool','shortlist','--k','10'],f'fast_{profile}_max_shortlist.json')
        slow=json.loads((output/f'naive_{profile}.json').read_text());quick=json.loads((output/f'fast_{profile}_max_shortlist.json').read_text())
        assert slow['rows'][-1]['candidate_solves']==3910
        for left,right in zip(slow['rows'],quick['rows']):
            if (left['index'],left['alpha'])!=(right['index'],right['alpha']) or abs(left['f']-right['f'])>1e-7:raise ArithmeticError('Task 1 and fast greedy mismatch')
    for profile,direction in SCENARIOS:
        for method in METHODS:
            worker('paper_task2',['--method',method,'--profile',profile,'--direction',direction],f'{method}_{profile}_{direction}_all.json')
    for profile in ('U','P'):
        for direction in ('max','min'):
            worker('paper_task2',['--method','local','--profile',profile,'--direction',direction],f'local_{profile}_{direction}_all.json')
            local_result=read(output,f'local_{profile}_{direction}_all.json')
            if (len(local_result['runs'])!=4 or local_result['start_spread']>LOCAL_TOL
                or any(r['status']!='certified' or r['objective_error_bound']>LOCAL_TOL
                       or abs(r['reference_gap'])>LOCAL_TOL for r in local_result['runs'])):
                raise ArithmeticError(f'Local search needs convergence diagnosis: {profile} {direction}; see saved histories')
        worker('paper_task2',['--method','witness','--profile',profile],f'witness_{profile}_max_all.json')
        worker('groundtruth_task3',['--profile',profile,'--restricted-comparisons'],f'random_{profile}_max_C.json')
    for profile,direction in SCENARIOS:worker('bonus_task4',['--profile',profile,'--direction',direction],f'bonus_{profile}_{direction}.json')
    save_json(output/'commands.json',commands)
    save_json(output/'complete.json',dict(complete=True,experiment_hash=fingerprint))

def read(output,name):return json.loads((Path(output)/name).read_text())

def jaccard(left,right):
    left=set(left);right=set(right)
    return len(left&right)/len(left|right) if left|right else 1.

def completed_snapshot(dataset,output):
    """Analysis may read an older completed run, but must identify its provenance."""
    manifest=read(output,'manifest.json');marker=read(output,'complete.json')
    if not marker.get('complete') or not manifest.get('experiment_hash') or marker.get('experiment_hash')!=manifest['experiment_hash']:
        raise RuntimeError('Missing or inconsistent completed experiment fingerprint')
    if hashlib.sha256(Path(dataset).read_bytes()).hexdigest()!=manifest['dataset_sha256']:
        raise RuntimeError('Dataset does not match the saved experiment')
    return manifest

def analyze(dataset,output):
    """Compute comparisons and export CSV tables and PNG plots."""
    output=Path(output);meta=completed_snapshot(dataset,output)
    for profile in ('U','P'):
        for direction in ('max','min'):
            data=read(output,f'local_{profile}_{direction}_all.json')
            if (not data.get('complete') or data.get('method')!='local' or len(data['runs'])!=4
                or data['start_spread']>LOCAL_TOL or any(r['status']!='certified'
                    or r['objective_error_bound']>LOCAL_TOL or abs(r['reference_gap'])>LOCAL_TOL for r in data['runs'])):
                raise RuntimeError(f'Incomplete or uncertified local search: {profile} {direction}')
    for profile,direction in SCENARIOS:
        for method in METHODS:
            data=read(output,f'{method}_{profile}_{direction}_all.json')
            if not data.get('complete') or [r['k'] for r in data['rows']]!=list(range(1,31)):
                raise RuntimeError(f'Incomplete 30-round result: {method} {profile} {direction}')
            if method=='random' and (len(data['trials'])!=20 or any(len(t)!=30 for t in data['trials'])):
                raise RuntimeError(f'Incomplete Random trials: {profile} {direction}')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ids,_,P,_,_,_=load_inputs(dataset)
    with np.load(output/'inputs.npz') as inputs:
        if not np.array_equal(ids,inputs['ids']):raise RuntimeError('Saved node IDs differ from the dataset')
        a=inputs['alpha'];profiles={p:inputs[p] for p in ('U','P')}
    figures=output/'figures';figures.mkdir(exist_ok=True)
    table=[];comparisons=[];marginal_counts={};claims={}
    def figure_save(fig,name):
        fig.savefig(figures/(name+'.png'),dpi=150,bbox_inches='tight');plt.close(fig)
    for profile,direction in SCENARIOS:
        sign=1 if direction=='max' else -1
        ref=read(output,f'local_{profile}_{direction}_all.json')['optimum']['f']
        fig,(ax,zoom)=plt.subplots(1,2,figsize=(12,4.8));scenario={}
        for method in METHODS:
            data=read(output,f'{method}_{profile}_{direction}_all.json');scenario[method]=data
            rows=[data['rows'][k-1] for k in BUDGETS];ys=[r['f'] for r in rows]
            ax.plot(BUDGETS,ys,'o-',label=method)
            zoom.plot(BUDGETS,ys,'o-',label=method)
            if method=='random':
                for panel in (ax,zoom):panel.fill_between(BUDGETS,np.array(ys)-np.array([r['std'] for r in rows]),np.array(ys)+np.array([r['std'] for r in rows]),alpha=.15)
            for k in BUDGETS:
                row=data['rows'][k-1];den=sign*(ref-data['initial'])
                table.append(dict(profile=profile,direction=direction,method=method,k=k,f=row['f'],improvement=sign*(row['f']-data['initial']),std=row.get('std'),gain_fraction=sign*(row['f']-data['initial'])/den if den else None,seconds=row['seconds'],peak_mib=row['peak_mib']))
        ax.axhline(ref,color='black',ls='--',label='Certified unbudgeted local search');ax.axhline(scenario['fast']['initial'],color='gray',ls=':',label='Initial')
        ax.set(xlabel='Budget k',ylabel='Total equilibrium opinion',title=f'{profile} {direction}: global reference');ax.legend(fontsize=8)
        zoom.axhline(scenario['fast']['initial'],color='gray',ls=':',label='Initial');zoom.set(xlabel='Budget k',ylabel='Total equilibrium opinion',title='Budgeted methods (zoom)');zoom.legend(fontsize=8);fig.tight_layout();figure_save(fig,f'objective_{profile}_{direction}')
        greedy=scenario['fast'];gains=[r['gain'] for r in greedy['rows']]
        marginal_counts[f'{profile}_{direction}']=sum(gains[j]>gains[j-1]+1e-8 for j in range(1,len(gains)))
        claims[f'{profile}_{direction}']={str(k):{m:sign*(greedy['rows'][k-1]['f']-scenario[m]['rows'][k-1]['f']) for m in ('I','II')} for k in BUDGETS}
        z0=Solver().solve(P,a,profiles[profile]);neighbor=P@z0
        fig,ax=plt.subplots(figsize=(6,6))
        for endpoint,marker in ((LOW,'v'),(HIGH,'^')):
            chosen=[r['index'] for r in greedy['rows'] if r['alpha']==endpoint]
            ax.scatter(profiles[profile][chosen],neighbor[chosen],marker=marker,label=f'alpha={endpoint}')
        ax.plot([0,1],[0,1],'k:',alpha=.5);ax.set(xlabel='Innate opinion s_i',ylabel='Pre-intervention mean neighbor equilibrium',title=f'Greedy targets: {profile} {direction}');ax.legend();figure_save(fig,f'targets_{profile}_{direction}')
        bonus=read(output,f'bonus_{profile}_{direction}.json')['results']
        fig,ax=plt.subplots(figsize=(8,4));ax.plot(BUDGETS,[bonus[str(k)]['added_gain'] for k in BUDGETS],'o-');ax.set(xlabel='Budget',ylabel='Additional improvement',title=f'Flip/swap local search: {profile} {direction}');figure_save(fig,f'bonus_{profile}_{direction}')
    for profile in ('U','P'):
        fig,ax=plt.subplots(figsize=(8,4))
        for name,label in [(f'naive_{profile}.json','Fresh solves, shortlist'),(f'fast_{profile}_max_shortlist.json','Fast, shortlist'),(f'fast_{profile}_max_all.json','Fast, all nodes')]:
            data=read(output,name);ax.plot([r['k'] for r in data['rows']],[r['seconds'] for r in data['rows']],'o-',label=label)
        ax.set(xlabel='Budget',ylabel='Cumulative seconds including setup',title=f'Runtime: {profile} maximisation');ax.legend();figure_save(fig,f'runtime_{profile}')
        ground=read(output,f'groundtruth_{profile}.json')
        for pool in ('all','C'):
            for method in METHODS:
                data=read(output,f'{method}_{profile}_max_{pool}.json')
                for k in (2,3):
                    ref=ground['references'][str(k)];den=ref['f']-ground['initial']
                    trajectories=data['trials'] if method=='random' else [data['rows']]
                    ratios=[(rows[k-1]['f']-ground['initial'])/den if den else None for rows in trajectories]
                    overlaps=[jaccard([r['added'] for r in rows[:k]],ref['ids']) for rows in trajectories]
                    comparisons.append(dict(profile=profile,pool=pool,method=method,k=k,gain_ratio=float(np.mean(ratios)) if den else None,jaccard=float(np.mean(overlaps))))
    write_csv(output/'comparison_table.csv',table)
    write_csv(output/'groundtruth_comparisons.csv',comparisons)
    checks1=read(output/'checks','task1_checks.json')
    fig,axes=plt.subplots(1,3,figsize=(11,3.5))
    for ax,item in zip(axes,checks1['sweeps']):ax.plot(item['grid'],item['objectives'],'o-');ax.set(title=item['graph'],xlabel='Resistance',ylabel='Total opinion')
    figure_save(fig,'endpoint_sweeps')
    local_rows=[]
    for profile in ('U','P'):
        for direction in ('max','min'):
            for run in read(output,f'local_{profile}_{direction}_all.json')['runs']:
                local_rows.append(dict(profile=profile,direction=direction,**{k:run[k] for k in
                    ('start_name','initial','f','moves','endpoint_count','status','objective_error_bound','reference_gap','seconds','peak_mib')}))
    write_csv(output/'local_search_comparisons.csv',local_rows)
    provenance=dict(recorded_experiment_hash=meta['experiment_hash'],current_source_hash=experiment_hash(dataset),
                    dataset_sha256=meta['dataset_sha256'],results_directory=str(output.resolve()))
    save_json(output/'analysis.json',dict(**provenance,marginal_increases=marginal_counts,greedy_minus_baselines=claims))
    print('Wrote analysis tables and plots to',output,flush=True)

def write_csv(path,rows):
    with Path(path).open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['checks','prepare','run','analyze'])
    parser.add_argument('--dataset',type=Path,default=ROOT/'data/facebook_combined.txt')
    parser.add_argument('--output',type=Path,default=ROOT/'results-local')
    args=parser.parse_args()
    if args.command=='checks':checks(args.output/'checks')
    elif args.command=='prepare':prepare(args.dataset)
    elif args.command=='run':run(args.dataset,args.output)
    elif args.command=='analyze':analyze(args.dataset,args.output)
if __name__=='__main__':main()
