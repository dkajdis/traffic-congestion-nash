"""Exact minimum-cost flow and three sequential best-response algorithms."""
from heapq import heappush, heappop
from collections import Counter
import random
import numpy as np
import networkx as nx
from traffic_game import Game, BASE_EDGES, compositions


def exact(game, n, include_states=False):
    F=np.array(list(compositions(n,len(game.paths))),dtype=np.int64)
    X=F@game.P
    pc=(game.a+X*game.b)@game.P.T
    S=np.sum(X*(game.a+X*game.b),axis=1)
    phi=np.sum(game.a*X+game.b*X*(X+1)//2,axis=1)
    correction=((1-game.P)*game.b)@game.P.T
    isne=np.ones(len(F),dtype=bool)
    for p in range(len(game.paths)):
        isne &= (F[:,p]==0)|(pc[:,p]<=np.min(pc+correction[p],axis=1))
    phimin=int(phi.min());smin=int(S.min())
    result=dict(n=n,k=len(game.paths),states=len(F),phi_min=phimin,
        lex_phi_social_cost=int(S[phi==phimin].min()),social_optimum=smin,
        social_optimum_counts=F[S==smin].tolist(),
        ne_count=int(isne.sum()),ne_best=int(S[isne].min()),ne_worst=int(S[isne].max()),
        equilibria=[dict(counts=f.tolist(),social_cost=int(s),potential=int(p))
                    for f,s,p in zip(F[isne],S[isne],phi[isne])])
    if include_states:
        result['all_states']=[dict(counts=f.tolist(),social_cost=int(s),potential=int(p),is_pne=bool(z))
                              for f,s,p,z in zip(F,S,phi,isne)]
    return result


def expand_counts(counts):
    return [p for p,n in enumerate(counts) for _ in range(n)]


def mincost_flow(game,n,objective='potential'):
    """N unit-capacity parallel arcs per original edge; single OD only."""
    bound=n*(len(game.graph)-1)*max(a+b*n for u,v,a,b in game.edges)+1
    H=nx.MultiDiGraph()
    for u in game.graph:
        H.add_node(u,demand=-n if u=='O' else n if u=='T' else 0)
    for eid,(u,v,a,b) in enumerate(game.edges):
        for k in range(1,n+1):
            potential_increment=a+b*k
            social_increment=a+b*(2*k-1)
            w={'potential':potential_increment,'social':social_increment,
               'lex':bound*potential_increment+social_increment}[objective]
            H.add_edge(u,v,key=k,capacity=1,weight=int(w))
    objective_value,flow=nx.network_simplex(H)
    x=np.array([sum(flow[u][v].values()) for u,v,a,b in game.edges],dtype=np.int64)
    # Decompose aggregate DAG flow in the declared path order, at most N units.
    remaining=x.copy();counts=np.zeros(len(game.paths),dtype=np.int64)
    for p,row in enumerate(game.P):
        ids=np.flatnonzero(row)
        amount=int(remaining[ids].min())
        counts[p]=amount;remaining-=amount*row
    assert not remaining.any() and int(counts.sum())==n
    state=expand_counts(counts)
    phi=game.potential(x);social=game.total_cost(x)
    expected={'potential':phi,'social':social,'lex':bound*phi+social}[objective]
    assert expected==objective_value
    verification=game.independent_verify(state)
    if objective!='social':
        assert verification['is_pne']
    return dict(objective=objective,n=n,path_counts=counts.tolist(),edge_flows=x.tolist(),
                potential=phi,social_cost=social,equilibrium_gap=verification['max_gain'],
                is_pne=verification['is_pne'],expanded_nodes=len(H),expanded_arcs=H.number_of_edges(),
                lex_multiplier=bound,objective_value=int(objective_value))


def dijkstra_oracle(game,x,old):
    weights=game.a+game.b*(x-game.P[old]+1)
    adj={v:[] for v in game.graph}
    for eid,(u,v,a,b) in enumerate(game.edges):
        adj[u].append((v,int(weights[eid])))
    d={'O':0};heap=[(0,'O')]
    while heap:
        distance,u=heappop(heap)
        if d[u]!=distance:
            continue
        if u=='T':
            return distance
        for v,w in adj[u]:
            q=distance+w
            if q<d.get(v,float('inf')):
                d[v]=q;heappush(heap,(q,v))
    raise RuntimeError('Unreachable T')


def br(game,initial,rule='cyclic',seed=0,check_oracles=True):
    s=list(initial);x=game.flows(s);rng=random.Random(seed)
    moves=0;queries=0;path_evaluations=0;oracle_matches=0;sweeps=0
    trace=[dict(move=0,potential=game.potential(x),social_cost=game.total_cost(x),gain=0)]
    def evaluate(old):
        nonlocal queries,path_evaluations,oracle_matches
        costs=game.deviation_costs(x,old)
        queries+=1;path_evaluations+=len(game.paths)
        best=int(np.argmin(costs));gain=int(costs[old]-costs[best])
        if check_oracles:
            assert int(costs[best])==dijkstra_oracle(game,x,old)
            oracle_matches+=1
        return gain,best
    def move(i,new,gain):
        nonlocal x,moves
        before=game.potential(x);old=s[i]
        x=x-game.P[old]+game.P[new];s[i]=new;moves+=1
        after=game.potential(x)
        assert before-after==gain and gain>0
        trace.append(dict(move=moves,potential=after,social_cost=game.total_cost(x),gain=gain))
    if rule=='max_gain':
        while True:
            sweeps+=1
            best_gain=0;best_i=None;best_new=None
            for i,old in enumerate(s):
                gain,new=evaluate(old)
                if gain>best_gain:
                    best_gain,best_i,best_new=gain,i,new
            if best_gain==0:
                break
            move(best_i,best_new,best_gain)
    else:
        while True:
            sweeps+=1;changed=False
            indices=list(range(len(s)))
            if rule=='random':rng.shuffle(indices)
            elif rule!='cyclic':raise ValueError(rule)
            for i in indices:
                gain,new=evaluate(s[i])
                if gain>0:
                    move(i,new,gain);changed=True
            if not changed:break
    # Dictionary-based verifier is independent of incidence/candidate formulas.
    verified=game.independent_verify(s)
    assert verified['is_pne']
    return dict(rule=rule,n=len(s),path_counts=np.bincount(s,minlength=len(game.paths)).tolist(),
        potential=game.potential(x),social_cost=game.total_cost(x),moves=moves,
        sweeps=sweeps,driver_queries=queries,path_candidate_evaluations=path_evaluations,
        dijkstra_matches=oracle_matches,equilibrium_gap=verified['max_gain'],trace=trace)


def classify_interval(with_edge,without_edge):
    lo,hi=with_edge['ne_best'],with_edge['ne_worst']
    blo,bhi=without_edge['ne_best'],without_edge['ne_worst']
    if hi<blo:return 'robust_benefit'
    if lo>bhi:return 'robust_harm'
    # A weak sign means every with-edge equilibrium is at least as good/bad,
    # and some comparisons tie; equal singleton intervals have no difference.
    if lo==hi==blo==bhi:return 'equal'
    if hi==blo:return 'weak_benefit'
    if lo==bhi:return 'weak_harm'
    return 'overlap'
