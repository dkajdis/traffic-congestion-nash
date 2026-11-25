"""Reproducible extended study, with no timing-based performance claims."""
from pathlib import Path
import argparse,csv,json,random,sys,platform,hashlib,math
from collections import Counter
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from traffic_game import Game,BASE_EDGES,compositions,self_checks
from advanced_game import exact,expand_counts,mincost_flow,br,classify_interval


def js(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def csvout(path,rows):
    if not rows:return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def compare_flow_exact(game,n,enumerated):
    p=mincost_flow(game,n,'potential');l=mincost_flow(game,n,'lex');s=mincost_flow(game,n,'social')
    assert p['potential']==enumerated['phi_min']
    assert (l['potential'],l['social_cost'])==(enumerated['phi_min'],enumerated['lex_phi_social_cost'])
    assert s['social_cost']==enumerated['social_optimum']
    return dict(potential=p,lex=l,social=s)


def edges_cd(mu):
    return [(u,v,mu if (u,v)==('C','D') else a,b) for u,v,a,b in BASE_EDGES]


def continuous_intervals(game,n=12):
    # At mu=0, path intercept alpha and overlap slope matrix H are integers.
    a0=game.a.copy();a0[[i for i,e in enumerate(game.edges) if e[:2]==('C','D')][0]]=0
    alpha=game.P@a0
    H=(game.P*game.b)@game.P.T
    cd_id=[i for i,e in enumerate(game.edges) if e[:2]==('C','D')][0]
    z=game.P[:,cd_id]
    rows=[]
    for f_tuple in compositions(n,len(game.paths)):
        f=np.array(f_tuple,dtype=np.int64);t0=alpha+H@f;low=0;high=12;feasible=True
        for r in np.flatnonzero(f):
            for q in range(len(f)):
                coeff=int(z[r]-z[q]);rhs=int(t0[q]+H[q,q]-H[r,q]-t0[r])
                if coeff==1:high=min(high,rhs)
                elif coeff==-1:low=max(low,-rhs)
                elif rhs<0:feasible=False
        if feasible and low<=high:
            x=f@game.P
            rows.append(dict(counts=list(f_tuple),mu_min=low,mu_max=high,
                social_intercept=int(x@(a0+game.b*x)),social_slope=int(x[cd_id])))
    # Check the complete output against fresh enumerations at all half-integers.
    # Scale every edge cost by two to avoid floating arithmetic at half-points.
    for numerator in range(25):
        scaled=[(u,v,numerator if (u,v)==('C','D') else 2*a,2*b) for u,v,a,b in BASE_EDGES]
        ex=exact(Game(scaled),n)
        predicted={tuple(r['counts']) for r in rows if 2*r['mu_min']<=numerator<=2*r['mu_max']}
        assert predicted=={tuple(r['counts']) for r in ex['equilibria']}
    return rows,dict(alpha_at_mu_zero=alpha.tolist(),H=H.tolist(),mu_coefficient=z.tolist(),
                     intervals=len(rows),half_integer_crosschecks=25)


def main(out):
    out.mkdir(parents=True,exist_ok=True);fig=out/'figures';fig.mkdir(exist_ok=True)
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
       'axes.unicode_minus':False,'font.size':10,'savefig.dpi':220,
       'axes.spines.top':False,'axes.spines.right':False})
    game=Game(BASE_EDGES)
    js(out/'network.json',dict(edges=BASE_EDGES,paths=game.paths,nodes=list(game.graph)))
    checks=self_checks();js(out/'self_checks.json',checks)
    print('Stage 1: exact flows and algorithm comparison',flush=True)
    ex=exact(game,12,include_states=True);js(out/'exact_N12_all_states.json',ex)
    flows=compare_flow_exact(game,12,ex);js(out/'flow_N12.json',flows)
    road_edges={'base':BASE_EDGES,'add_BC':BASE_EDGES+[('B','C',2,0)],
        'delete_CD':[e for e in BASE_EDGES if e[:2]!=('C','D')],
        'improve_DT':[(u,v,a,1 if (u,v)==('D','T') else b) for u,v,a,b in BASE_EDGES]}
    road_N12={}
    for name,es in road_edges.items():
        g=Game(es);rex=exact(g,12)
        road_N12[name]=dict(edges=es,paths=g.paths,exact=rex,flows=compare_flow_exact(g,12,rex))
    js(out/'road_N12.json',road_N12)
    baseline=br(game,[0]*12);js(out/'baseline.json',baseline)
    alg=[]
    for seed in range(100):
        rng=random.Random(20261123+seed);initial=[rng.randrange(5) for _ in range(12)]
        for rule in ['cyclic','random','max_gain']:
            r=br(game,initial,rule,seed)
            alg.append(dict(seed=seed,rule=rule,initial=initial,counts=r['path_counts'],
                potential=r['potential'],social_cost=r['social_cost'],moves=r['moves'],
                driver_queries=r['driver_queries'],path_candidate_evaluations=r['path_candidate_evaluations'],
                oracle_matches=r['dijkstra_matches'],equilibrium_gap=r['equilibrium_gap']))
    csvout(out/'algorithm_runs.csv',alg)
    algo_summary=[]
    for rule in ['cyclic','random','max_gain']:
        rs=[r for r in alg if r['rule']==rule]
        algo_summary.append(dict(rule=rule,runs=len(rs),cost_min=min(r['social_cost'] for r in rs),
          cost_max=max(r['social_cost'] for r in rs),cost_mean=float(np.mean([r['social_cost'] for r in rs])),
          moves_median=float(np.median([r['moves'] for r in rs])),queries_median=float(np.median([r['driver_queries'] for r in rs])),
          queries_min=min(r['driver_queries'] for r in rs),queries_max=max(r['driver_queries'] for r in rs),
          candidate_median=float(np.median([r['path_candidate_evaluations'] for r in rs])),
          outcomes=dict(sorted(Counter(r['social_cost'] for r in rs).items()))))
    csvout(out/'algorithm_summary.csv',algo_summary)
    scale=[];scaleflows=[]
    for n in [12,40,80,160,400]:
        for seed in range(10):
            rng=random.Random(90210+seed);initial=[rng.randrange(5) for _ in range(n)]
            for rule in ['cyclic','random','max_gain']:
                r=br(game,initial,rule,seed)
                scale.append(dict(n=n,seed=seed,rule=rule,cost=r['social_cost'],moves=r['moves'],
                    driver_queries=r['driver_queries'],path_candidate_evaluations=r['path_candidate_evaluations'],
                    dijkstra_matches=r['dijkstra_matches'],equilibrium_gap=r['equilibrium_gap']))
        for objective in ['potential','lex','social']:
            r=mincost_flow(game,n,objective);scaleflows.append(r)
        print(f'  scaling N={n} completed',flush=True)
    csvout(out/'scaling_br.csv',scale);js(out/'scaling_flows.json',scaleflows)
    # Two drivers, parallel routes, c(n)=n: synchronous BR creates a cycle.
    syncgame=Game([('O','A',0,1),('A','T',0,0),('O','B',0,1),('B','T',0,0)])
    state=[0,0];sync=[]
    for t in range(5):
        x=syncgame.flows(state)
        sync.append(dict(t=t,profile=state,potential=syncgame.potential(x),social_cost=syncgame.total_cost(x)))
        changed=[]
        for old in state:
            c=syncgame.deviation_costs(x,old);best=int(np.argmin(c))
            changed.append(best if c[best]<c[old] else old)
        state=changed
    assert sync[0]['profile']==sync[2]['profile'] and sync[0]['profile']!=sync[1]['profile']
    js(out/'synchronous_counterexample.json',dict(latency='n on first edge, zero on second',n=2,trace=sync,
        sequential=br(syncgame,[0,0],check_oracles=True)))
    print('Stage 2: complete demand-cost phase diagram',flush=True)
    phase=[];phase_exact=[]
    without_edges=[e for e in BASE_EDGES if e[:2]!=('C','D')]
    for n in range(2,25):
        without=exact(Game(without_edges),n)
        for mu in range(13):
            with_edge=exact(Game(edges_cd(mu)),n)
            category=classify_interval(with_edge,without)
            phase.append(dict(n=n,mu=mu,category=category,with_ne_count=with_edge['ne_count'],
                with_best=with_edge['ne_best'],with_worst=with_edge['ne_worst'],
                without_ne_count=without['ne_count'],without_best=without['ne_best'],without_worst=without['ne_worst'],
                with_social_optimum=with_edge['social_optimum'],without_social_optimum=without['social_optimum']))
            phase_exact.append(dict(n=n,mu=mu,with_edge=with_edge,without_edge=without))
        print(f'  phase N={n} completed',flush=True)
    csvout(out/'phase.csv',phase);js(out/'phase_all_equilibria.json',phase_exact)
    intervals,matrices=continuous_intervals(game);js(out/'continuous_NE_intervals.json',intervals)
    csvout(out/'continuous_NE_intervals.csv',[dict(counts=str(r['counts']),**{k:v for k,v in r.items() if k!='counts'}) for r in intervals])
    js(out/'path_cost_matrices.json',matrices)
    threshold=[]
    for n in range(2,25):
        rs=[r for r in phase if r['n']==n]
        threshold.append(dict(n=n,**{cat:','.join(str(r['mu']) for r in rs if r['category']==cat)
            for cat in ['robust_benefit','weak_benefit','equal','overlap','weak_harm','robust_harm']}))
    csvout(out/'phase_thresholds.csv',threshold)
    print('Stage 3: fixed-topology parameter ensemble',flush=True)
    rng=random.Random(20261124);parameters=[];param_attempts=[]
    for sample in range(100):
        es=[(u,v,rng.randint(0,15),rng.randint(0,3)) for u,v,a,b in BASE_EDGES]
        # Explicit cost-only rejection; do not filter by intervention outcome.
        attempt=dict(attempt=sample+1,edges=es,accepted=any(a or b for u,v,a,b in es))
        param_attempts.append(attempt)
        if not attempt['accepted']:raise RuntimeError('All-zero draw: rerun requires explicit resampling rule')
        g=Game(es);original=exact(g,8);flow=compare_flow_exact(g,8,original)
        interventions={
            'delete_CD':[e for e in es if e[:2]!=('C','D')],
            'add_BC':es+[('B','C',2,0)],
            'improve_DT':[(u,v,a,max(0,b-1) if (u,v)==('D','T') else b) for u,v,a,b in es],
        }
        changed={}
        for name,ies in interventions.items():
            rex=exact(Game(ies),8)
            changed[name]=dict(exact=rex,category=classify_interval(rex,original))
        parameters.append(dict(sample=sample,edges=es,base=original,flows=flow,interventions=changed,
            lex_equals_best_ne=flow['lex']['social_cost']==original['ne_best']))
    js(out/'parameter_ensemble.json',parameters);js(out/'parameter_attempts.json',param_attempts)
    print('Stage 4: random topology ensemble',flush=True)
    rng=random.Random(20261125);nodes=['O','A','B','C','D','E','T'];chain=list(zip(nodes,nodes[1:]))
    possible=[(u,v) for i,u in enumerate(nodes) for j,v in enumerate(nodes) if i<j and (u,v) not in chain]
    topologies=[];attempts=[]
    while len(topologies)<40:
        m=rng.randint(8,11);extras=rng.sample(possible,m-6);bare=chain+extras
        G=nx.DiGraph();G.add_edges_from(bare);paths=list(nx.all_simple_paths(G,'O','T'));k=len(paths)
        accepted=2<=k<=8
        attempts.append(dict(attempt=len(attempts)+1,edges=bare,path_count=k,accepted=accepted,
                             reason='accepted' if accepted else 'path_count_outside_2_to_8'))
        if not accepted:continue
        eligible=[]
        for e in bare:
            H=G.copy();H.remove_edge(*e)
            if nx.has_path(H,'O','T'):eligible.append(e)
        chosen=rng.choice(eligible)
        es=[(u,v,rng.randint(0,15),rng.randint(0,3)) for u,v in bare]
        g=Game(es);original=exact(g,6);flow=compare_flow_exact(g,6,original)
        reduced=exact(Game([e for e in es if e[:2]!=chosen]),6)
        topologies.append(dict(sample=len(topologies),attempt=len(attempts),edges=es,paths=g.paths,
            removed_edge=chosen,base=original,changed=reduced,flows=flow,
            category=classify_interval(reduced,original),
            lex_equals_best_ne=flow['lex']['social_cost']==original['ne_best']))
    js(out/'topology_ensemble.json',topologies);js(out/'topology_attempts.json',attempts)
    psummary={name:dict(Counter(p['interventions'][name]['category'] for p in parameters))
              for name in ['delete_CD','add_BC','improve_DT']}
    tsummary=dict(Counter(p['category'] for p in topologies))
    # Explicit independent verification of every stored equilibrium in ensembles.
    checked=0
    def verify_all(g,ex):
        nonlocal checked
        for eq in ex['equilibria']:
            assert g.independent_verify(expand_counts(eq['counts']))['is_pne'];checked+=1
    for p in parameters:
        verify_all(Game(p['edges']),p['base'])
        es=p['edges']
        variants={'delete_CD':[e for e in es if e[:2]!=('C','D')],
          'add_BC':es+[('B','C',2,0)],
          'improve_DT':[(u,v,a,max(0,b-1) if (u,v)==('D','T') else b) for u,v,a,b in es]}
        for name,edges in variants.items():verify_all(Game(edges),p['interventions'][name]['exact'])
    for p in topologies:
        verify_all(Game(p['edges']),p['base'])
        verify_all(Game([e for e in p['edges'] if e[:2]!=tuple(p['removed_edge'])]),p['changed'])
    scaling_summary=[]
    for n in [12,40,80,160,400]:
        for rule in ['cyclic','random','max_gain']:
            rows=[r for r in scale if r['n']==n and r['rule']==rule]
            scaling_summary.append(dict(n=n,rule=rule,runs=len(rows),
                moves_median=float(np.median([r['moves'] for r in rows])),
                queries_median=float(np.median([r['driver_queries'] for r in rows])),
                candidates_median=float(np.median([r['path_candidate_evaluations'] for r in rows])),
                queries_min=min(r['driver_queries'] for r in rows),queries_max=max(r['driver_queries'] for r in rows),
                cost_min=min(r['cost'] for r in rows),cost_max=max(r['cost'] for r in rows)))
    csvout(out/'scaling_summary.csv',scaling_summary)
    summary=dict(algorithm_summary=algo_summary,flow_N12=flows,road_N12=road_N12,scaling_summary=scaling_summary,
        scaling_flows=scaleflows,synchronous_counterexample=sync,
        comparison_direction=dict(phase='treatment=with CD; reference=without CD',
            parameter_ensemble='treatment=modified network; reference=original parameter sample',
            topology_ensemble='treatment=edge removed; reference=original topology sample'),
        category_definitions=dict(robust_benefit='treatment_max < reference_min',
            robust_harm='treatment_min > reference_max',
            weak_benefit='treatment_max = reference_min, excluding equal singletons',
            weak_harm='treatment_min = reference_max, excluding equal singletons',
            equal='both intervals are the same singleton',
            overlap='remaining overlapping intervals; includes equal non-singleton intervals'),
        phase_categories=dict(Counter(r['category'] for r in phase)),phase_cells=len(phase),
        continuous_intervals=len(intervals),parameter_ensemble=psummary,topology_ensemble=tsummary,
        parameter_samples=len(parameters),topology_samples=len(topologies),topology_attempts=len(attempts),
        lex_not_best_ne_parameter_samples=sum(not p['lex_equals_best_ne'] for p in parameters),
        lex_not_best_ne_topology_samples=sum(not p['lex_equals_best_ne'] for p in topologies),
        independently_verified_ensemble_equilibria=checked,
        algorithm_driver_queries=sum(r['driver_queries'] for r in alg),
        scaling_driver_queries=sum(r['driver_queries'] for r in scale),
        oracle_matches=sum(r['oracle_matches'] for r in alg)+sum(r['dijkstra_matches'] for r in scale)+baseline['dijkstra_matches'],
        research_br_runs=1+len(alg)+len(scale),all_br_gaps_zero=True)
    js(out/'summary.json',summary)
    js(out/'environment.json',dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,
       networkx=nx.__version__,matplotlib=matplotlib.__version__,
       seeds=dict(algorithm_initial=20261123,scaling_initial=90210,parameters=20261124,topologies=20261125)))
    make_figures(fig,game,baseline,alg,scale,flows,phase,parameters,topologies,summary,sync)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def make_figures(fig,game,baseline,alg,scale,flows,phase,parameters,topologies,summary,sync):
    # 1: retain the original network and deterministic baseline trajectory.
    pos={'O':(0,0),'A':(1,1.2),'B':(1,-1.2),'C':(2.4,1.2),'D':(3.0,0),'E':(2.8,-1.2),'T':(4.5,0)}
    f,ax=plt.subplots(figsize=(8,4.3));nx.draw_networkx(game.graph,pos,ax=ax,node_color='#e2edf6',node_size=1250,
       edgecolors='#32516a',font_size=12,width=1.4,arrowsize=20)
    nx.draw_networkx_edge_labels(game.graph,pos,{(u,v):f'{a}+{b}n' if b else str(a) for u,v,a,b in BASE_EDGES},
       ax=ax,font_size=10,label_pos=.55,bbox=dict(facecolor='white',edgecolor='none',pad=.3))
    ax.set_axis_off();f.tight_layout();f.savefig(fig/'network.png');plt.close(f)
    # 2: convergence + synchronous counterexample, no implied smooth dynamics.
    f,axs=plt.subplots(1,3,figsize=(10.5,3.2));h=baseline['trace']
    axs[0].plot([x['move'] for x in h],[x['potential'] for x in h],'o-',ms=3);axs[0].set_ylabel('势函数Φ')
    axs[1].plot([x['move'] for x in h],[x['social_cost'] for x in h],'o-',ms=3,color='#bc733f');axs[1].set_ylabel('社会成本S')
    for ax in axs[:2]:ax.set_xlabel('串行严格改进次数');ax.grid(alpha=.2)
    axs[2].step([r['t'] for r in sync],[r['profile'][0] for r in sync],where='post',color='#9f5267')
    axs[2].set_yticks([0,1],['均走上路','均走下路']);axs[2].set_xlabel('同步更新轮次');axs[2].set_title('双司机同步更新：2周期')
    f.tight_layout();f.savefig(fig/'convergence_sync.png');plt.close(f)
    # 3: queries count for all drivers, including maximal-gain searches.
    f,axs=plt.subplots(1,2,figsize=(8.5,3.5));rules=['cyclic','random','max_gain'];labels=['循环BR','随机BR','最大收益BR']
    colors=['#356b8e','#6a9b83','#b98444']
    for rule,label,color in zip(rules,labels,colors):
        ns=[12,40,80,160,400]
        meds=[np.median([r['driver_queries'] for r in scale if r['n']==n and r['rule']==rule]) for n in ns]
        axs[0].plot(ns,meds,'o-',label=label,color=color,ms=4)
    axs[0].set_xscale('log');axs[0].set_yscale('log');axs[0].set_xlabel('司机数N（对数轴）');axs[0].set_ylabel('司机oracle查询数中位数（对数轴）')
    axs[0].legend(fontsize=8);axs[0].grid(alpha=.2)
    values=[[r['social_cost'] for r in alg if r['rule']==rule] for rule in rules]
    axs[1].boxplot(values,tick_labels=labels,widths=.5,showmeans=True)
    for y,color,label in [(flows['potential']['social_cost'],'#a15e56','势流401'),(flows['lex']['social_cost'],'#51745a','lex势流397'),(389,'#4b5460','社会最优389')]:
        axs[1].axhline(y,ls='--',color=color,label=label,lw=1)
    axs[1].set_ylabel('N=12终点社会成本');axs[1].legend(fontsize=7,loc='upper right');axs[1].set_ylim(386,423)
    f.tight_layout();f.savefig(fig/'algorithm_comparison.png');plt.close(f)
    # 4: categorical sign, never confuse overlap with zero effect.
    cats=['robust_benefit','weak_benefit','equal','overlap','weak_harm','robust_harm']
    names=['稳健改善','弱改善（含平局）','完全相等','区间重叠','弱恶化（含平局）','稳健恶化']
    colors=['#2c7c59','#a2c8b5','#eeeeee','#b5b7cb','#e5bb93','#b65a45']
    from matplotlib.colors import ListedColormap,BoundaryNorm
    matrix=np.array([[cats.index(next(r['category'] for r in phase if r['n']==n and r['mu']==mu)) for mu in range(13)] for n in range(2,25)])
    f,ax=plt.subplots(figsize=(8,5));im=ax.imshow(matrix,origin='lower',aspect='auto',interpolation='nearest',cmap=ListedColormap(colors),norm=BoundaryNorm(np.arange(-.5,6.5),6))
    ax.set_xticks(range(13),range(13));ax.set_yticks(range(0,23,2),range(2,25,2));ax.set_xlabel('C→D基础成本μ');ax.set_ylabel('司机数N')
    cbar=f.colorbar(im,ax=ax,ticks=range(6),fraction=.04,pad=.04);cbar.ax.set_yticklabels(names,fontsize=9)
    ax.set_title('保留C→D相对于删除C→D的均衡社会成本区间')
    f.tight_layout();f.savefig(fig/'phase_diagram.png');plt.close(f)
    # 5: ensemble results for every accepted sample, categories defined in README.
    f,axs=plt.subplots(1,2,figsize=(9,3.8))
    rows=[summary['parameter_ensemble'][k] for k in ['delete_CD','add_BC','improve_DT']]+[summary['topology_ensemble']]
    labels=['参数组：删C→D','参数组：加B→C','参数组：降D→T斜率','随机DAG：随机删可删边']
    bottom=np.zeros(4)
    for cat,label,color in zip(cats,names,colors):
        vals=np.array([r.get(cat,0) for r in rows])/np.array([100,100,100,40])*100
        axs[0].barh(range(4),vals,left=bottom,label=label,color=color,edgecolor='white');bottom+=vals
    axs[0].set_yticks(range(4),labels,fontsize=8);axs[0].set_xlabel('干预后相对于干预前：样本占比（%）',fontsize=9);axs[0].set_xlim(0,100)
    axs[0].legend(fontsize=7,ncol=2,loc='upper center',bbox_to_anchor=(.5,1.24),frameon=False)
    base=[p['base']['ne_worst'] for p in parameters];change=[p['interventions']['delete_CD']['exact']['ne_worst'] for p in parameters]
    axs[1].scatter(base,change,s=16,alpha=.65,color='#416f8c');mx=max(base+change)
    axs[1].plot([0,mx],[0,mx],'--',color='#999999');axs[1].set_xlabel('删路前最差均衡成本');axs[1].set_ylabel('删路后最差均衡成本')
    f.tight_layout();f.savefig(fig/'ensemble.png');plt.close(f)
    # 6: compare all four N12 equilibria, exact potential versus social objective.
    f,axs=plt.subplots(1,2,figsize=(8.5,3.5))
    counts=np.array([[1,4,0,2,5],[2,3,0,2,5],[2,3,0,3,4],[3,2,0,3,4]])
    bottom=np.zeros(4)
    for p,color in enumerate(['#386b91','#69a5bd','#acc9ad','#d5a552','#b57579']):
        axs[0].bar(range(4),counts[:,p],bottom=bottom,label=f'P{p+1}',color=color);bottom+=counts[:,p]
    axs[0].set_xticks(range(4),['NE1','NE2','NE3','NE4']);axs[0].set_ylabel('各路径司机数');axs[0].legend(ncol=5,fontsize=7,loc='upper center',bbox_to_anchor=(.5,1.15),frameon=False)
    values=[411,401,407,397];axs[1].bar(range(4),values,color='#698ea7')
    for i,v in enumerate(values):axs[1].text(i,v+4,str(v),ha='center',fontsize=9)
    axs[1].axhline(389,ls='--',color='#a66745',label='社会最优389');axs[1].set_xticks(range(4),['NE1','NE2','NE3','NE4']);axs[1].set_ylim(0,500);axs[1].set_ylabel('社会总成本（四者Φ均为308）');axs[1].legend(fontsize=8)
    f.tight_layout();f.savefig(fig/'all_equilibria.png');plt.close(f)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=Path(__file__).parent/'results')
    main(parser.parse_args().out)
