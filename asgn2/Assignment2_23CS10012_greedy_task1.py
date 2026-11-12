"""Task 1 and shared numerical primitives. Run --help for reproducible commands."""
from __future__ import annotations
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import resource
import time
import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.linalg import bicgstab, spsolve
import networkx as nx

LOW, HIGH = .001, 1.
BUDGETS = (1, 2, 3, 4, 5, 10, 20, 30)
SEED = 2026
TIE = 1e-9
ROOT = Path(__file__).resolve().parent

def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    def encode(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=encode, allow_nan=False) + '\n')
    tmp.replace(path)

def rss():
    """Linux total-process high-water RSS, converted from KiB to MiB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

def rng(stream):
    return np.random.default_rng(np.random.SeedSequence([SEED, int(stream)]))

def graph_matrix(graph):
    ids = np.array(sorted(graph.nodes()), dtype=np.int64)
    adjacency = sparse.csr_matrix(nx.to_scipy_sparse_array(graph, nodelist=ids, dtype=float, format='csr'))
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    if np.any(degree == 0): raise ValueError('Isolated vertices are unsupported by the specified random walk')
    return ids, degree, sparse.diags(1 / degree) @ adjacency

def load_inputs(dataset):
    path = Path(dataset)
    graph = nx.Graph(); loops = duplicates = lines = 0
    with path.open() as fin:
        for line in fin:
            if not line.strip() or line.startswith('#'): continue
            u, v = map(int, line.split()); lines += 1
            if u == v: loops += 1; continue
            if graph.has_edge(u, v): duplicates += 1
            graph.add_edge(u, v)
    ids, degree, P = graph_matrix(graph)
    if (len(ids), graph.number_of_edges(), nx.number_connected_components(graph)) != (4039, 88234, 1):
        raise ValueError('Input does not have the expected ego-Facebook graph counts')
    a = rng(0).uniform(LOW, HIGH, len(ids))
    profiles = {'U': rng(1).uniform(0, 1, len(ids)), 'P': rng(2).beta(.5, .5, len(ids))}
    meta = dict(dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), nodes=len(ids),
                edges=graph.number_of_edges(), components=1, raw_lines=lines,
                removed_loops=loops, collapsed_duplicates=duplicates, seed=SEED,
                streams={'alpha':0,'U':1,'P':2,'random':'1000 + pool_code*100 + trial','local_x_start':3000},
                python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                networkx=nx.__version__, pandas=version('pandas'), matplotlib=version('matplotlib'), machine=platform.machine(), processor=platform.processor(),
                cpus=os.cpu_count(), threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')})
    return ids, degree, P.tocsr(), a, profiles, meta

def system(P, a):
    return sparse.eye(len(a), format='csr') - sparse.diags(1-a) @ P

class Solver:
    def __init__(self): self.calls = 0; self.recoveries = 0
    def solve(self, P, a, s, tol=2e-13):
        self.calls += 1
        M = system(P, a); b = a*s
        z, info = bicgstab(M, b, rtol=tol, atol=tol, maxiter=10000)
        if info != 0 or not np.isfinite(z).all():
            self.recoveries += 1
            z = spsolve(M, b)
        if not np.isfinite(z).all():
            raise ArithmeticError(f'Equilibrium solve returned nonfinite values: info={info}')
        return z

def iterate(P, a, s, tolerance=2e-12):
    z = s.copy()
    for step in range(200000):
        new = a*s + (1-a)*(P@z)
        residual = np.max(np.abs(new-z))
        z = new
        if residual / np.min(a) < tolerance: return z, step+1
    raise ArithmeticError('Opinion iteration did not converge')

def shortlist(ids, degree, count=200):
    return np.lexsort((ids, -degree))[:min(count, len(ids))]

def choose(candidates, direction=1):
    """Candidates: (index, endpoint, objective). Batch ties are deterministic."""
    best = max(direction * x[2] for x in candidates)
    return min((x for x in candidates if best-direction*x[2] <= TIE), key=lambda x:(x[0],x[1]))

def naive(P, a, s, ids, pool, k, out=None):
    start=time.perf_counter(); solver=Solver(); current=a.copy()
    z=solver.solve(P,current,s); initial=float(z.sum()); selected=set(); rows=[]; candidate_calls=0
    for round_no in range(1,k+1):
        candidates=[]
        for v in pool:
            if int(v) in selected: continue
            for endpoint in (LOW,HIGH):
                trial=current.copy(); trial[v]=endpoint
                z_trial=solver.solve(P,trial,s); candidate_calls+=1
                candidates.append((int(v),endpoint,float(z_trial.sum())))
        v,endpoint,f=choose(candidates)
        current[v]=endpoint; selected.add(v)
        rows.append(dict(k=round_no,added=int(ids[v]),index=v,alpha=endpoint,f=f,
                         gain=f-(rows[-1]['f'] if rows else initial), seconds=time.perf_counter()-start,
                         peak_mib=rss(),candidate_solves=candidate_calls,total_solves=solver.calls,
                         recovery_solves=solver.recoveries))
        if out: save_json(out,dict(initial=initial,rows=rows,complete=round_no==k))
    return dict(initial=initial,rows=rows,complete=True)

def small_checks(output):
    solver=Solver(); checks=[]
    _,_,P=graph_matrix(nx.path_graph(2))
    for a,s in [(np.array([.2,.8]),np.array([.1,.9])),(np.array([LOW,1.]),np.array([1.,0.])),(np.array([1.,1.]),np.array([.3,.7]))]:
        den=a[0]+a[1]-a[0]*a[1]
        exact=np.array([a[0]*s[0]+(1-a[0])*a[1]*s[1],a[1]*s[1]+(1-a[1])*a[0]*s[0]])/den
        z=solver.solve(P,a,s)
        assert np.max(abs(z-exact))<1e-10
        checks.append(dict(alpha=a,s=s,exact=exact,computed=z))
    sweeps=[]
    for name,graph in [('path',nx.path_graph(5)),('triangle',nx.complete_graph(3)),('star',nx.star_graph(5))]:
        ids,degree,P=graph_matrix(graph); n=len(ids); a=rng(10+n).uniform(LOW,HIGH,n);s=rng(20+n).random(n)
        values=[]; grid=np.linspace(LOW,HIGH,21)
        for endpoint in grid:
            trial=a.copy();trial[0]=endpoint; values.append(float(solver.solve(P,trial,s).sum()))
        assert max(values)<=max(values[0],values[-1])+1e-9
        sweeps.append(dict(graph=name,node=0,alpha=a,s=s,grid=grid,objectives=values))
    result=dict(two_node=checks,sweeps=sweeps)
    save_json(Path(output)/'task1_checks.json',result)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=ROOT/'data/facebook_combined.txt')
    parser.add_argument('--output',type=Path,default=ROOT/'results')
    parser.add_argument('--checks',action='store_true');parser.add_argument('--profile',choices=['U','P'],default='U')
    parser.add_argument('--k',type=int,default=10)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if args.checks: small_checks(args.output);print('Task 1 small checks passed');return
    ids,degree,P,a,profiles,meta=load_inputs(args.dataset)
    if not 1<=args.k<=10: parser.error('Task 1 requires k in 1..10')
    if not (args.output/'manifest.json').exists():save_json(args.output/'manifest.json',meta)
    np.savez(args.output/'inputs.npz',ids=ids,alpha=a,U=profiles['U'],P=profiles['P'])
    naive(P,a,profiles[args.profile],ids,shortlist(ids,degree),args.k,args.output/f'naive_{args.profile}.json')
if __name__=='__main__':main()
