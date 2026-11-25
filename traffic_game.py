"""Finite, unweighted atomic congestion game. All computations use integers."""
from dataclasses import dataclass
from itertools import product
from collections import Counter
import random
import networkx as nx
import numpy as np


BASE_EDGES = [
    ('O', 'A', 2, 2), ('O', 'B', 7, 1), ('A', 'C', 3, 1),
    ('A', 'D', 12, 0), ('B', 'D', 2, 2), ('B', 'E', 8, 1),
    ('C', 'T', 14, 0), ('C', 'D', 1, 0), ('D', 'T', 1, 2),
    ('E', 'T', 3, 1),
]


def compositions(n, k):
    """All k-component nonnegative integer vectors summing to n."""
    if k == 1:
        yield (n,)
    else:
        for first in range(n + 1):
            for tail in compositions(n - first, k - 1):
                yield (first,) + tail


@dataclass
class Game:
    edges: list

    def __post_init__(self):
        self.graph = nx.DiGraph()
        self.graph.add_edges_from((u, v) for u, v, a, b in self.edges)
        assert nx.is_directed_acyclic_graph(self.graph)
        assert all(a >= 0 and b >= 0 for u, v, a, b in self.edges)
        # Deterministic DFS order follows the declared edge insertion order.
        self.paths = list(nx.all_simple_paths(self.graph, 'O', 'T'))
        self.P = np.array([[int((u, v) in set(zip(p, p[1:])))
                            for u, v, a, b in self.edges] for p in self.paths], dtype=np.int64)
        self.a = np.array([e[2] for e in self.edges], dtype=np.int64)
        self.b = np.array([e[3] for e in self.edges], dtype=np.int64)

    def flows(self, profile):
        return np.bincount(profile, minlength=len(self.paths)) @ self.P

    def potential(self, x):
        return int(np.sum(self.a*x + self.b*x*(x+1)//2))

    def total_cost(self, x):
        return int(x @ (self.a + self.b*x))

    def deviation_costs(self, x, old_path):
        """Remove the driver, then insert them into each candidate path."""
        return self.P @ (self.a + self.b*(x-self.P[old_path]+1))

    def best_response(self, initial, order='cyclic', seed=0):
        rng = random.Random(seed)
        s = list(initial)
        x = self.flows(s)
        hist = [dict(move=0, sweep=0, driver=-1, old_path=-1, new_path=-1,
                     private_gain=0, potential=self.potential(x), total_cost=self.total_cost(x))]
        checks = 0
        for sweep in range(1, 100001):
            indices = list(range(len(s)))
            if order == 'random':
                rng.shuffle(indices)
            elif order != 'cyclic':
                raise ValueError(order)
            moved = False
            for i in indices:
                checks += 1
                old = s[i]
                costs = self.deviation_costs(x, old)
                cur = int(costs[old])
                new = int(np.argmin(costs))
                gain = cur-int(costs[new])
                if gain > 0:  # Equal-cost alternatives never trigger a move.
                    before = self.potential(x)
                    x = x-self.P[old]+self.P[new]
                    s[i] = new
                    after = self.potential(x)
                    assert after-before == -gain
                    hist.append(dict(move=len(hist), sweep=sweep, driver=i,
                                     old_path=old+1, new_path=new+1, private_gain=gain,
                                     potential=after, total_cost=self.total_cost(x)))
                    moved = True
            if not moved:  # Complete no-change sweep is required.
                return dict(profile=s, path_counts=np.bincount(s, minlength=len(self.paths)).tolist(),
                            edge_flows=x.tolist(), potential=self.potential(x),
                            total_cost=self.total_cost(x), mean_cost=self.total_cost(x)/len(s),
                            moves=len(hist)-1, sweeps=sweep, checks=checks, history=hist)
        raise RuntimeError('Safety cap hit; convergence has NOT been certified.')

    def independent_verify(self, profile):
        """Brute-force deviations using edge dictionaries, independent of P/formula."""
        path_edges = [list(zip(p, p[1:])) for p in self.paths]
        pars = {(u, v): (a, b) for u, v, a, b in self.edges}
        def driver_cost(state, i):
            counts = Counter(e for p in state for e in path_edges[p])
            return sum(pars[e][0]+pars[e][1]*counts[e] for e in path_edges[state[i]])
        max_gain = 0
        rows = []
        for i, old in enumerate(profile):
            current = driver_cost(profile, i)
            deviations = []
            for p in range(len(self.paths)):
                changed = list(profile)
                changed[i] = p
                deviations.append(driver_cost(changed, i))
            gain = current-min(deviations)
            max_gain = max(max_gain, gain)
            rows.append(dict(driver=i+1, path=old+1, current_cost=current,
                             best_deviation_cost=min(deviations), max_gain=gain,
                             alternative_costs=deviations))
        return dict(is_pne=max_gain == 0, max_gain=max_gain, drivers=rows)

    def enumerate_counts(self, n):
        F = np.array(list(compositions(n, len(self.paths))), dtype=np.int64)
        X = F @ self.P
        cost = self.a + X*self.b
        totals = np.sum(X*cost, axis=1)
        path_costs = cost @ self.P.T
        # Entering q after leaving p costs c_q + sum_e b_e P_qe(1-P_pe).
        correction = (1-self.P)*self.b @ self.P.T
        pne = np.ones(len(F), dtype=bool)
        for p in range(len(self.paths)):
            best = np.min(path_costs+correction[p], axis=1)
            pne &= (F[:, p] == 0) | (path_costs[:, p] <= best)
        opt = int(totals.min())
        records = [dict(path_counts=f.tolist(), total_cost=int(c),
                        potential=self.potential(x), edge_flows=x.tolist())
                   for f, c, x in zip(F[pne], totals[pne], X[pne])]
        assert records
        return dict(n=n, paths=len(self.paths), states=len(F), equilibrium_count=len(records),
                    equilibria=records, optimum_cost=opt,
                    optimum_counts=F[totals == opt].tolist(),
                    best_pne_cost=min(r['total_cost'] for r in records),
                    worst_pne_cost=max(r['total_cost'] for r in records),
                    poa=max(r['total_cost'] for r in records)/opt,
                    pos=min(r['total_cost'] for r in records)/opt)


def self_checks():
    """Exhaustive labeled-profile check and potential identity on N=4."""
    game = Game(BASE_EDGES)
    equilibrium_counts = set()
    identity_checks = 0
    for s in product(range(len(game.paths)), repeat=4):
        x = game.flows(s)
        verified = game.independent_verify(s)
        if verified['is_pne']:
            equilibrium_counts.add(tuple(np.bincount(s, minlength=len(game.paths))))
        for i, old in enumerate(s):
            costs = game.deviation_costs(x, old)
            for new in range(len(game.paths)):
                x2 = x-game.P[old]+game.P[new]
                assert game.potential(x2)-game.potential(x) == int(costs[new]-costs[old])
                identity_checks += 1
    exact = game.enumerate_counts(4)
    assert equilibrium_counts == {tuple(x['path_counts']) for x in exact['equilibria']}
    # Constant equal-cost paths create a genuine alternative tie. In particular,
    # driver 0 starts on path 1 although argmin returns path 0.
    tie = Game([('O','A',1,0),('A','T',0,0),('O','B',1,0),('B','T',0,0)])
    two = Game([('O','A',0,1),('A','T',0,0),('O','B',0,1),('B','T',0,0)])
    tied = tie.best_response([1,0])
    assert tied['moves'] == 0 and tied['profile'] == [1,0]
    assert two.best_response([0,0])['moves'] == 1
    return dict(labeled_profiles=5**4, unilateral_identity_checks=identity_checks,
                aggregate_pne_match=True, tie_stays=True, self_removal_check=True)
