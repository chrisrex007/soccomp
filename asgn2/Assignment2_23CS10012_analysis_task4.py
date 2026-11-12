"""Task 4: assignment checks, fresh experiments, CSV comparisons and plots."""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
from itertools import product
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
from Assignment2_23CS10012_greedy_task1 import (ROOT,LOW,HIGH,BUDGETS,Solver,system,save_json,load_inputs,graph_matrix,rng,iterate,small_checks)
from Assignment2_23CS10012_paper_task2 import (local_search,local_starts,LOCAL_TOL,paper_example)

PREFIX='Assignment2_23CS10012_'
SCENARIOS=[('U','max'),('P','max'),('U','min')]
METHODS=['fast','I','II','random']

def checks(output):
    """The required two-node, endpoint-grid, example and unbudgeted checks."""
    start=time.perf_counter();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    small_checks(output);save_json(output/'paper_example.json',paper_example());records=[]
    for graph in [nx.path_graph(6),nx.complete_graph(5)]:
        ids,_,P=graph_matrix(graph);n=len(ids);a=rng(n).uniform(LOW,HIGH,n);s=rng(n+30).random(n)
        endpoint_values=[float(np.linalg.solve(system(P,np.array(es)).toarray(),np.array(es)*s).sum())
                         for es in product((LOW,HIGH),repeat=n)]
        runs=[]
        for direction in (1,-1):
            exact=max(endpoint_values) if direction==1 else min(endpoint_values)
            objectives=[]
            for name,a0 in local_starts(a):
                result=local_search(P,s,a0,ids,direction)
                gap=abs(exact-result['f'])
                assert gap<=LOCAL_TOL, (n,direction,name,gap)
                objectives.append(result['f'])
                runs.append(dict(direction=direction,start_name=name,f=result['f'],exact=exact,gap=gap))
            assert np.ptp(objectives)<=LOCAL_TOL
        records.append(dict(nodes=n,edges=graph.number_of_edges(),local=runs))
    result=dict(passed=True,graphs=records,seconds=time.perf_counter()-start,python=platform.python_version())
    save_json(output/'validation.json',result);print('All required small-graph checks passed',flush=True)
    return result

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
    output=Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise RuntimeError('Choose a new or empty output directory; full runs do not resume')
    output.mkdir(parents=True,exist_ok=True)
    ids,degree,P,a,profiles,meta=load_inputs(dataset);fingerprint=experiment_hash(dataset)
    manifest=output/'manifest.json'
    meta['experiment_hash']=fingerprint;save_json(manifest,meta)
    np.savez(output/'inputs.npz',ids=ids,alpha=a,U=profiles['U'],P=profiles['P'])
    checks(output/'checks')
    equilibrium={}
    for profile,s in profiles.items():
        z=Solver().solve(P,a,s);zi,steps=iterate(P,a,s);error=float(np.max(abs(z-zi)))
        assert error<1e-8
        equilibrium[profile]=dict(f=float(z.sum()),iteration_error=error,steps=steps)
    save_json(output/'equilibrium.json',equilibrium)
    commands=[]
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR='/tmp/soccomp-matplotlib',XDG_CACHE_HOME='/tmp/soccomp-cache')
    def worker(script,extra,worker_output=output):
        cmd=[sys.executable,str(ROOT/(PREFIX+script+'.py')),'--dataset',str(Path(dataset).resolve()),'--output',str(worker_output.resolve())]+extra
        print('Running',script,' '.join(extra),flush=True);tick=time.perf_counter()
        subprocess.run(cmd,env=env,check=True)
        commands.append(dict(command=cmd,wall_seconds=time.perf_counter()-tick));save_json(output/'commands.json',commands)
    for profile in ('U','P'):
        worker('greedy_task1',['--profile',profile])
        worker('paper_task2',['--method','fast','--profile',profile,'--pool','shortlist','--k','10'])
        slow=json.loads((output/f'naive_{profile}.json').read_text());quick=json.loads((output/f'fast_{profile}_max_shortlist.json').read_text())
        assert slow['rows'][-1]['candidate_solves']==3910
        assert len(slow['rows'])==len(quick['rows'])==10
        for left,right in zip(slow['rows'],quick['rows']):
            if (left['index'],left['alpha'])!=(right['index'],right['alpha']) or abs(left['f']-right['f'])>1e-7:raise ArithmeticError('Task 1 and fast greedy mismatch')
    for profile,direction in SCENARIOS:
        for method in METHODS:
            worker('paper_task2',['--method',method,'--profile',profile,'--direction',direction])
            if method=='random':
                for k in (5,10):
                    worker('paper_task2',['--method','random','--profile',profile,'--direction',direction,'--k',str(k)],output/'measurements'/f'k{k}')
    for profile in ('U','P'):
        for direction in ('max','min'):
            worker('paper_task2',['--method','local','--profile',profile,'--direction',direction])
        worker('paper_task2',['--method','witness','--profile',profile])
        worker('groundtruth_task3',['--profile',profile,'--restricted-comparisons'])
    for profile,direction in SCENARIOS:worker('bonus_task4',['--profile',profile,'--direction',direction])
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
            for k in (5,10,30):
                measured=read(output/'measurements'/f'k{k}',f'random_{profile}_{direction}_all.json') if method=='random' and k<30 else data
                row=measured['rows'][k-1];den=sign*(ref-measured['initial'])
                seconds=measured['total_seconds'] if method=='random' else row['seconds']
                peak=measured['peak_mib'] if method=='random' else row['peak_mib']
                table.append(dict(profile=profile,direction=direction,method=method,k=k,f=row['f'],improvement=sign*(row['f']-measured['initial']),std=row.get('std'),gain_fraction=sign*(row['f']-measured['initial'])/den if den else None,seconds=seconds,peak_mib=peak))
        ax.axhline(ref,color='black',ls='--',label='Unbudgeted local search');ax.axhline(scenario['fast']['initial'],color='gray',ls=':',label='Initial')
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
                    ('start_name','initial','f','moves','endpoint_count','status','seconds','peak_mib')}))
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
