"""Independent scalar checks of stored N12 states and every phase equilibrium."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json
from traffic_game import Game,BASE_EDGES
from advanced_game import expand_counts,classify_interval


def scalar_state(edges,paths,counts):
    routes=[list(zip(p,p[1:])) for p in paths]
    parameters={(u,v):(a,b) for u,v,a,b in edges}
    load=Counter()
    for route,n in zip(routes,counts):
        for e in route:load[e]+=n
    total=sum(n*(parameters[e][0]+parameters[e][1]*n) for e,n in load.items())
    potential=sum(parameters[e][0]*n+parameters[e][1]*n*(n+1)//2 for e,n in load.items())
    ok=True
    for r,n in enumerate(counts):
        if not n:continue
        current=sum(parameters[e][0]+parameters[e][1]*load[e] for e in routes[r])
        removed=load.copy()
        for e in routes[r]:removed[e]-=1
        for route in routes:
            dev=sum(parameters[e][0]+parameters[e][1]*(removed[e]+1) for e in route)
            if dev<current:ok=False
    return total,potential,ok


def main(root,compare=None):
    r=root/'results';game=Game(BASE_EDGES)
    full=json.loads((r/'exact_N12_all_states.json').read_text(encoding='utf-8'))
    for state in full['all_states']:
        total,potential,isne=scalar_state(BASE_EDGES,game.paths,state['counts'])
        assert (total,potential,isne)==(state['social_cost'],state['potential'],state['is_pne'])
    phase=json.loads((r/'phase_all_equilibria.json').read_text(encoding='utf-8'))
    phasechecked=0
    for cell in phase:
        es=[(u,v,cell['mu'] if (u,v)==('C','D') else a,b) for u,v,a,b in BASE_EDGES]
        for edges,key in [(es,'with_edge'),([e for e in es if e[:2]!=('C','D')],'without_edge')]:
            g=Game(edges)
            for eq in cell[key]['equilibria']:
                total,potential,isne=scalar_state(edges,g.paths,eq['counts'])
                assert isne and total==eq['social_cost'] and potential==eq['potential'];phasechecked+=1
    report=dict(n12_complete_states_checked=len(full['all_states']),phase_equilibria_checked=phasechecked,
                n12_social_optimum=full['social_optimum'],n12_phi_min=full['phi_min'])
    if compare:
        rows=[]
        for p in sorted(r.rglob('*')):
            if not p.is_file():continue
            q=compare/p.relative_to(r)
            first=hashlib.sha256(p.read_bytes()).hexdigest();second=hashlib.sha256(q.read_bytes()).hexdigest()
            rows.append(dict(file=str(p.relative_to(r)),sha256=first,replay_sha256=second,identical=first==second))
        assert all(p['identical'] for p in rows)
        report['replay_files_checked']=len(rows);report['all_byte_identical']=True
        (root/'reproducibility_check.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    (root/'verification_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--compare',type=Path)
    main(Path(__file__).parent,p.parse_args().compare)
