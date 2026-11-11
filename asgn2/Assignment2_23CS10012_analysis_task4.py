"""Task 4: validation, reproducible experiment driver, analysis and PDF report.

Commands: checks, prepare, run, report, archive. Use --help for paths.
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
import tempfile
import time
import urllib.request
import zipfile
import numpy as np
import networkx as nx
from Assignment2_23CS10012_greedy_task1 import (ROOT,LOW,HIGH,BUDGETS,Solver,system,save_json,load_inputs,graph_matrix,rng,naive,iterate,small_checks)
from Assignment2_23CS10012_paper_task2 import (State,FixedBase,fast,baseline,random_trials,local_search,local_starts,LOCAL_TOL,bellman_reference,paper_example,optimized_set,witnesses)
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

def report(dataset,output,report_path):
    """Build the actual-results report; refuse incomplete runs."""
    output=Path(output);complete=read(output,'complete.json')
    if complete['experiment_hash']!=experiment_hash(dataset):raise RuntimeError('Report inputs do not match the completed experiments')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import pandas as pd
    import textwrap
    ids,degree,P,a,profiles,meta=load_inputs(dataset)
    figures=output/'figures';figures.mkdir(exist_ok=True)
    table=[];comparisons=[];marginal_counts={};claims={};plot_paths=[]
    def figure_save(fig,name):
        path=figures/(name+'.png');fig.savefig(path,dpi=150,bbox_inches='tight');plot_paths.append(path);plt.close(fig)
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
                row=data['rows'][k-1];den=sign*(ref-data['initial'])
                table.append(dict(profile=profile,direction=direction,method=method,k=k,f=row['f'],gain_fraction=sign*(row['f']-data['initial'])/den if den else None,seconds=row['seconds'],peak_mib=row['peak_mib']))
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
    pd.DataFrame(table).to_csv(output/'comparison_table.csv',index=False)
    pd.DataFrame(comparisons).to_csv(output/'groundtruth_comparisons.csv',index=False)
    save_json(output/'analysis.json',dict(marginal_increases=marginal_counts,greedy_minus_baselines=claims))
    checks1=read(output/'checks','task1_checks.json')
    fig,axes=plt.subplots(1,3,figsize=(11,3.5))
    for ax,item in zip(axes,checks1['sweeps']):ax.plot(item['grid'],item['objectives'],'o-');ax.set(title=item['graph'],xlabel='Resistance',ylabel='Total opinion')
    figure_save(fig,'endpoint_sweeps')
    with PdfPages(report_path) as pdf:
        def page(title,paragraphs):
            lines=[]
            for paragraph in paragraphs:
                for line in str(paragraph).splitlines():
                    lines.extend(textwrap.wrap(line,width=100,replace_whitespace=False) or [''])
                lines.append('')
            for offset in range(0,len(lines),48):
                fig=plt.figure(figsize=(8.27,11.69));fig.text(.07,.96,title if offset==0 else title+' (continued)',fontsize=14,va='top')
                fig.text(.07,.91,'\n'.join(lines[offset:offset+48]),fontsize=9,va='top',linespacing=1.45,family='monospace');pdf.savefig(fig);plt.close(fig)
        page('23CS10012',[
            'Assignment 2: Opinion Maximisation on ego-Facebook',
            'Reproducible numerical implementation and experiments. All target IDs use original SNAP labels.',
            'Software: '+json.dumps(meta),
            'Execution hardware: '+json.dumps(read(output,'hardware.json')) if (output/'hardware.json').exists() else 'Hardware details: see manifest.json.',
            'Data: official SNAP facebook_combined.txt, undirected simple graph; input SHA-256 '+meta['dataset_sha256']+'. Duplicate edges collapse and self loops are removed; observed counts: '+str({k:meta[k] for k in ('nodes','edges','components','removed_loops','collapsed_duplicates')}),
            'Bounds [0.001,1]; budgets 1,2,3,4,5,10,20,30. Initial alpha uniform; U uniform; P Beta(0.5,0.5). SeedSequence([2026,stream]) with alpha=0,U=1,P=2. The same alpha is shared across profiles. Random stream=1000+100*pool_code+trial (20 trials); local-search random X start=3000.',
            'Reproduction: python '+PREFIX+'analysis_task4.py prepare; python '+PREFIX+'analysis_task4.py run; python '+PREFIX+'analysis_task4.py report. Use --dataset and --output to override paths. commands.json records each task command. requirements.txt pins tested versions. Set OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1 before Python.',
            'Timing includes algorithm initialization and numerical verification. Each main algorithm runs in a fresh Python subprocess. Peak memory is Linux total-process ru_maxrss / 1024 MiB, including native arrays. Random time is initial setup plus the sum of 20 trial prefix times; its shading is one sample standard deviation, not a confidence interval.'
        ])
        findings=[]
        for profile,direction in SCENARIOS:
            d=read(output,f'fast_{profile}_{direction}_all.json');last=d['rows'][-1]
            sign=1 if direction=='max' else -1
            findings.append(f'{profile} {direction}: initial f={d["initial"]:.6f}; greedy at k=30 gives f={last["f"]:.6f}, improvement {sign*(last["f"]-d["initial"]):.6f}. Cumulative time {last["seconds"]:.3f} s, peak RSS {last["peak_mib"]:.2f} MiB.')
        for profile in ('U','P'):
            slow=read(output,f'naive_{profile}.json')['rows'][-1];quick=read(output,f'fast_{profile}_max_shortlist.json')['rows'][-1]
            findings.append(f'{profile} matching-shortlist ten rounds: fresh solves {slow["seconds"]:.3f} s versus fast {quick["seconds"]:.3f} s ({slow["seconds"]/quick["seconds"]:.2f}x speedup), with matching targets and endpoint assignments.')
        findings.append('These comparisons describe one recorded draw of the initial resistances and each opinion profile. Random variability is conditional on those fixed inputs; it is not uncertainty over graph samples or opinion-profile draws.')
        for profile in ('U','P'):
            for direction in ('max','min'):
                data=read(output,f'local_{profile}_{direction}_all.json')
                findings.append(f'{profile} {direction} local search: '+json.dumps(dict(
                    canonical_f=data['optimum']['f'],start_spread=data['start_spread'],
                    statuses=[r['status'] for r in data['runs']],moves=[r['moves'] for r in data['runs']],
                    maximum_reference_gap=max(abs(r['reference_gap']) for r in data['runs']))))
        exact_matches=[]
        for profile in ('U','P'):
            ground=read(output,f'groundtruth_{profile}.json');greedy=read(output,f'fast_{profile}_max_all.json')
            exact_matches.append(f'{profile}: |C|={ground["pool_size"]}; absolute greedy/reference discrepancies at k=2,3 are '+str([abs(greedy['rows'][k-1]['f']-ground['references'][str(k)]['f']) for k in (2,3)]))
        findings.append('Restricted exact references: '+'; '.join(exact_matches)+'. This agreement on small budgets does not imply a global approximation guarantee.')
        local=[r['already_local'] for profile,direction in SCENARIOS for r in read(output,f'bonus_{profile}_{direction}.json')['results'].values()]
        findings.append(f'Bonus: {sum(local)} of {len(local)} tested scenario/budget starts were already locally optimal for the specified flip/shortlist-swap neighborhood. Zero improvement is reported without claiming optimality outside that neighborhood.')
        page('Main findings',findings)
        page('Task 1: equilibrium and endpoint checks',[
            'M=I-(I-A)P, b=A s; solve Mz=b with checked bicgstab (rtol and atol 2e-13), falling back to a checked sparse direct solve on failure. No specialized opinion-dynamics package is used.',
            'For two nodes, D=alpha_a+alpha_b-alpha_a*alpha_b. z_a=[alpha_a*s_a+(1-alpha_a)*alpha_b*s_b]/D; z_b is the symmetric expression. Checks: '+json.dumps(checks1['two_node']),
            'Facebook direct iteration versus independent linear solve: '+json.dumps(read(output,'equilibrium.json')),
            'For fixed other coordinates, the Sherman-Morrison expression makes f linear-fractional in one resistance, with a positive denominator and a derivative of constant sign. Thus a maximum lies at an endpoint (or every value ties). Coordinate replacement yields an endpoint optimum. Three 21-point sweeps are shown below.',
            'Fresh greedy evaluates both endpoints at every remaining candidate, retains previous assignments, and uses the 200 highest-degree candidates on the full graph. Each ten-round profile uses 3910 candidate solves; verification and fallback solves are recorded separately. Ties use objective tolerance 1e-9, then original ID and lower endpoint.',
        ])
        for profile in ('U','P'):
            rows=read(output,f'naive_{profile}.json')['rows']
            page('Task 1 checkpoints: '+profile,[json.dumps(rows[k-1]) for k in (1,2,3,4,5,10)])
        page('Task 2: fast greedy and baselines',[
            'Let G=M^-1, delta=alpha_new[v]-alpha[v], g=G[:,v], h=P[v,:]G, d=1+delta*h[v]. M_new=M+delta*e_v*P[v,:]. Sherman-Morrison gives G_new=G-delta*g*h/d. With the changed RHS, z_new=z+delta*(s[v]-(Pz)[v])*g/d; the gain is the same scalar times sum(g).',
            'Cache c=1^T G and t=diag(PG). With beta=delta/d, c_new=c-beta*c[v]*h and t_new=t-beta*(P*g)*h. All candidate evaluations use cached scalars; no candidate equilibrium solve. Each accepted state is checked by an independent solver. No division by 1-alpha is used.',
            'Both profiles match every Task 1 shortlist choice and objective through ten rounds within 1e-7; the driver aborts on a mismatch. All-node runs cover 30 rounds for U-max, P-max and U-min.',
            'Baseline I ranks innate s. Baseline II ranks degree/(2m) times s/sum_neighbor(s). Both assign selected alpha=1, as in paper Section 5. Minimisation ranks 1-s but evaluates original s. Positive/zero is infinity and 0/0 is zero; ties use IDs.',
            'Random samples 20 uniform permutations, using nested prefixes. At each step it tests both endpoints given earlier assignments and keeps the better one. It does not jointly reoptimize the selected set. Fixed-base Woodbury systems avoid new full solves. All-node and C-restricted universes are labeled separately.'
        ])
        page('Task 2 part 2: corrected non-convex local search',[
            'Following the TA clarification dated 11 November 2026, the unbudgeted experiment uses the corrected paper\'s local-search strategy in Section 6.1. The previous projected-gradient implementation is replaced. This is the basic equilibrium-based single-coordinate strategy, not the approximate or optimistic algorithms of Section 6.3.',
            'For minimisation, solve Mz=A s and compute d=s-Pz. At the first reliably improving coordinate in ascending original-ID order, set alpha_i to 0.001 if d_i>0, or to 1 if d_i<0. Recompute equilibrium after one accepted move. Exact or numerically unresolved ties keep their current alpha; opinions are not perturbed.',
            'Boundary adaptation: the paper assumes upper bounds below 1. At alpha_i=1, s_i-z_i=0 even when lowering alpha_i improves the objective. Instead use dz/dalpha_i=M^-1 e_i*(s_i-(Pz)_i), whose nonnegative inverse column has positive sum. Thus the neighbour-average test remains valid at the assignment\'s upper bound 1.',
            'The canonical run begins at all upper bounds. In exact arithmetic each coordinate moves from upper to lower at most once: every accepted minimising move decreases z componentwise, so s-Pz cannot decrease. This extends the irrevocable-update argument to upper bound 1. The implementation rejects a reverse flip in this run. The three additional initialisations are robustness comparisons and do not inherit the n-move guarantee.',
            'Starts: upper=all ones; initial=the sampled resistance vector; x_midpoint=1/500.5 in each coordinate; x_random=reciprocals of the stream-3000 uniform draw on [1,1000]. The last three preserve the original assignment starting points after converting X back to alpha.',
            'Maximisation minimises complemented innate opinions 1-s. A fresh final solve with the original s supplies the reported objective and checks f_s+f_(1-s)=n. Histories report the original objective; positive gain always means improvement in the requested direction.'
        ])
        page('Task 2: numerical acceptance and theory',[
            'Every proposed move uses a fresh equilibrium solve. With r=||Mz-A s||_infinity, an equilibrium error estimate is (r+8*machine_epsilon)/min(alpha). Accept a move only when its objective improvement exceeds the combined total-objective uncertainty of the old and new solves. If a scan finds no resolvable move, repeat with sparse direct solves before stopping.',
            'A final independent direct solve is checked against H_i(z)=max/min over a in {0.001,1} of a*s_i+(1-a)*(Pz)_i. The contraction bound is n*||H(z)-z||_infinity/0.001; the implementation adds a roundoff allowance and an equilibrium-solve error allowance. These are floating-point residual diagnostics, not rigorous interval certificates.',
            'Success requires each of the four starts to have status certified and an objective error bound <=1e-5, absolute reference gaps <=1e-5, and a total-objective spread <=1e-5. A move cap or unresolved sign is not accepted as proof of convergence. The independent simultaneous endpoint-policy routine remains a cross-check; plot reference lines use the certified canonical local-search result.',
            'The transformed objective is not convex in general. For two nodes with s=(1,0), f(x,y)=(2y-1)/(x+y-1). f(2,2)=1 exceeds [f(2,1)+f(2,3)]/2=7/8, but is below [f(1,2)+f(3,2)]/2=9/8. No convexity claim is used to justify the new algorithm.',
            'The corrected paper establishes the structure behind local and global optima and gives the all-upper-bound irrevocable strategy. Endpoint optima exist; ties can also admit interior optimal coordinates. Therefore interior coordinates in a supplementary start are reported, not forcibly rounded or treated alone as a failure.',
            'Small-graph checks compare local search with exhaustive endpoint enumeration using independent dense solves, replay each accepted move, and test constant-opinion ties, the alpha=1 boundary, complement symmetry, move limits and original-ID ordering.'
        ])
        for profile in ('U','P'):
            for direction in ('max','min'):
                data=read(output,f'local_{profile}_{direction}_all.json')
                page(f'Unbudgeted local search {profile} {direction}',[
                    'Canonical optimum: '+json.dumps(data['optimum']),
                    'Independent reference: '+json.dumps({k:v for k,v in data['reference'].items() if k!='alpha'}),
                    'Four-start objective spread: '+str(data['start_spread']),
                    *[json.dumps({k:v for k,v in r.items() if k not in ('alpha','initial_alpha','history')}) for r in data['runs']],
                    'Full initial/final alpha vectors and move histories are in the corresponding raw JSON file.'
                ])
        page('Task 2 local-search validation',[
            json.dumps([{k:v for k,v in g.items() if k in ('nodes','local')} for g in read(output/'checks','validation.json')['graphs']]),
            'Boundary and tie checks: '+json.dumps(read(output/'checks','validation.json')['local_edge_checks'])
        ])
        page('Task 2: non-submodularity and claim verdicts',[
            'The supplied published paper labels the requested example Example 4, not Example 13. Triangle s=(1,0,0), alpha=(0.1,0.1,0.1): '+json.dumps(read(output/'checks','paper_example.json')),
            'The paper prints 1.493 for f({3})-f(empty), omitting f(empty)=1. The corrected marginal is about 0.493 and the non-supermodularity violation remains.',
            'Claim 3 verdict by profile (both distinct inequalities witnessed): '+json.dumps({p:len(read(output,f"witness_{p}_max_all.json")["witnesses"])==2 for p in ('U','P')}),
            'Each Facebook witness evaluates the optimized set function by exhaustive endpoint assignments at all four sets and verifies those assignments with fresh solves. Larger marginal at B violates submodularity; smaller marginal at B violates supermodularity.',
            *[profile+': '+json.dumps(read(output,f'witness_{profile}_max_all.json')) for profile in ('U','P')],
            'Claim 4 verdict by scenario (strictly superior at EVERY requested budget): '+json.dumps({scenario:all(gap>1e-8 for by_method in budgets.values() for gap in by_method.values()) for scenario,budgets in claims.items()}),
            'Claim 4: signed greedy-minus-baseline differences at every requested budget (positive means strictly better): '+json.dumps(claims),
            'Do not generalize empirical superiority into an approximation guarantee.'
        ])
        for profile in ('U','P'):
            ground=read(output,f'groundtruth_{profile}.json')
            page('Task 3 exact restricted reference: '+profile,[json.dumps(ground),
                'For changed set S, let E select its columns, D=diag(delta), R=P[S,:]. Solve (I+D*R*G*E)w=D*(s[S]-R*z). Then f_new=f_base+(1^T*G*E)w. Enumerate all subsets and endpoint assignments; verify competitive configurations independently. Numerical ties use IDs and endpoints.',
                'T* is optimal within C to the stated numerical tolerance, not necessarily over all 4039 nodes. Hence all-node gain ratios may exceed one. Jaccard is measured against the canonical reported reference; Random averages per-trial overlaps.'
            ])
        page('Task 4 comparison table',[pd.DataFrame(table).round(6).to_string(index=False)])
        page('Task 4 reference ratios and overlaps',[pd.DataFrame(comparisons).round(6).to_string(index=False)])
        page('Task 4 greedy choices and bonus',[
            'Scatter uses innate s against the PRE-intervention mean neighbor equilibrium Pz0. The rank-one gain has positive column sum and denominator, so raising alpha helps maximisation when s exceeds the current neighbor average, and lowering helps when it is below. The pre-intervention explanatory plot can differ from current-round conditions.',
            'Counts of consecutive marginal improvements increasing by more than 1e-8: '+json.dumps(marginal_counts),
            'Bonus scan order is ascending selected original ID, then the deterministic descending-degree shortlist (ID ties), with lower endpoint tested first. Exact equal-gain moves keep the first encountered move. Improvements must exceed 1e-8.',
            'Bonus scans all flips and swaps of a selected target with an unselected member of the 200-degree shortlist, testing both inserted endpoints. Removal restores original alpha. Deterministic best improvement above 1e-8 repeats until a complete scan finds none. Each accepted move is independently checked. This is a local optimum for the specified neighborhood, not a global certificate.',
            *[f'{profile} {direction}: '+json.dumps({k:{field:value for field,value in r.items() if field in ('initial','f','added_gain','scans','evaluations','already_local','seconds','peak_mib')} for k,r in read(output,f'bonus_{profile}_{direction}.json')['results'].items()}) for profile,direction in SCENARIOS]
        ])
        for profile,direction in SCENARIOS:
            bonus=read(output,f'bonus_{profile}_{direction}.json')['results']
            count=sum(r['already_local'] for r in bonus.values())
            page(f'Bonus summary: {profile} {direction}',[
                f'Greedy was already locally optimal in {count}/{len(bonus)} tested budgets ({count/len(bonus):.1%}).',
                'Flips revise earlier endpoint choices after the surrounding equilibrium changes. Swaps let the algorithm undo an earlier node choice. These moves can escape the irreversible decisions of forward greedy; if no gain occurs, the completed neighborhood scan establishes local optimality up to 1e-8 for this neighborhood.',
                'Baseline and Random comparisons, exact restricted references, and the bonus share identical initial parameters within each scenario. Raw outputs retain all target IDs, endpoint assignments and per-round values.'
            ])
        for path in plot_paths:
            fig=plt.figure(figsize=(11.69,8.27));ax=fig.add_axes([.04,.04,.92,.92]);ax.imshow(plt.imread(path));ax.axis('off');pdf.savefig(fig);plt.close(fig)
        page('Commands producing the numerical results',[
            'Commands below are the recorded execution commands. For another location, use the included scripts with --dataset and --output pointing to the extracted input/results. The README and RUN_GUIDE give WSL commands.',
            *[' '.join(entry['command']) for entry in read(output,'commands.json') if not entry.get('reused')]
        ])
        page('References and reproducibility',[
            'Abebe, Kleinberg, Parkes, Tsourakakis. Opinion Dynamics with Varying Susceptibility to Persuasion. KDD 2018, pp.1089-1098. DOI:10.1145/3219819.3219983. Sections 2,4,5 supply endpoint, greedy and baseline definitions.',
            'SNAP ego-Facebook: https://snap.stanford.edu/data/ego-Facebook.html',
            'CS60017 Assignment: Opinion Maximisation on ego-Facebook, supplied four-page assignment, deadline 13 November 2026.',
            'Abebe, Chan, Kleinberg, Liang, Parkes, Sozio, Tsourakakis. Opinion Dynamics with Varying Susceptibility to Persuasion via Non-Convex Local Search. arXiv:2011.02583v1, 4 November 2020. https://arxiv.org/abs/2011.02583. Sections 5 and 6.1 supply the corrected structural argument and basic local-search strategy; the upper-bound-1 adaptation is described above.',
            'Raw per-round records, target IDs, assigned endpoints, seeds, reference certificates, command list, input arrays/checksums and software metadata accompany this report. Generated tables and figures derive exclusively from those completed experiments.'
        ])
    print('Wrote',report_path,flush=True)

def archive(dataset,output,report_path,destination):
    if not Path(report_path).is_file():raise RuntimeError('Generate the completed report before archiving')
    marker=read(output,'complete.json')
    if marker['experiment_hash']!=experiment_hash(dataset):raise RuntimeError('Archive inputs changed since experiment completion')
    files=list(ROOT.glob(PREFIX+'*.py'))+[Path(report_path),ROOT/'requirements.txt',ROOT/'README.md',ROOT/'RUN_GUIDE.md',Path(dataset)]
    files+=list(Path(output).rglob('*.json'))+list(Path(output).rglob('*.csv'))+list(Path(output).rglob('*.npz'))+list(Path(output).rglob('*.png'))
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as z:
        for path in files:
            if path==Path(dataset):relative=Path('data/facebook_combined.txt')
            elif path.is_relative_to(Path(output)):relative=Path('results')/path.relative_to(Path(output))
            else:relative=Path(path.name)
            z.write(path,str(Path('Assignment2_23CS10012')/relative))
    with tempfile.TemporaryDirectory(prefix='soccomp-archive-') as tmp:
        with zipfile.ZipFile(destination) as z:z.extractall(tmp)
        root=Path(tmp)/'Assignment2_23CS10012'
        for script in root.glob('*.py'):subprocess.run([sys.executable,str(script),'--help'],check=True,stdout=subprocess.DEVNULL)
        subprocess.run([sys.executable,str(root/(PREFIX+'analysis_task4.py')),'checks','--output',str(Path(tmp)/'checks')],check=True)
        code='from Assignment2_23CS10012_greedy_task1 import load_inputs, Solver; ids,d,P,a,s,m=load_inputs("data/facebook_combined.txt"); assert len(ids)==4039; assert Solver().solve(P,a,s["U"]).shape==(4039,)'
        subprocess.run([sys.executable,'-c',code],cwd=root,check=True)
    print('Archive extracted and validated:',destination,flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['checks','prepare','run','report','archive']);parser.add_argument('--dataset',type=Path,default=ROOT/'data/facebook_combined.txt');parser.add_argument('--output',type=Path,default=ROOT/'results');parser.add_argument('--report',type=Path,default=ROOT/(PREFIX+'report.pdf'));parser.add_argument('--archive',type=Path,default=ROOT/'Assignment2_23CS10012.zip')
    args=parser.parse_args()
    if args.command=='checks':checks(args.output/'checks')
    elif args.command=='prepare':prepare(args.dataset)
    elif args.command=='run':run(args.dataset,args.output)
    elif args.command=='report':report(args.dataset,args.output,args.report)
    elif args.command=='archive':archive(args.dataset,args.output,args.report,args.archive)
if __name__=='__main__':main()
