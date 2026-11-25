"""Independent scalar review: imports no project experiment module."""
from pathlib import Path
from fractions import Fraction
from collections import Counter,defaultdict
import csv,json,itertools,random,math,time
import networkx as nx
root=Path(__file__).resolve().parent/'results'
def read(n):return json.loads((root/n).read_text(encoding='utf-8'))
def comps(n,k):
 if k==1:yield (n,);return
 for z in range(n+1):
  for t in comps(n-z,k-1):yield (z,)+t
class Model:
 def __init__(self,es):
  self.es=[tuple(e) for e in es];g=nx.DiGraph();g.add_edges_from((u,v) for u,v,a,b in self.es)
  self.paths=list(nx.all_simple_paths(g,'O','T'));ids={(u,v):j for j,(u,v,a,b) in enumerate(self.es)}
  self.routes=[[ids[x] for x in zip(p,p[1:])] for p in self.paths]
  self.a=[e[2] for e in self.es];self.b=[e[3] for e in self.es];self.k=len(self.routes)
 def loads(self,f):
  x=[0]*len(self.es)
  for r,n in zip(self.routes,f):
   for e in r:x[e]+=n
  return x
 def phi(self,x):return sum(a*n+b*n*(n+1)//2 for a,b,n in zip(self.a,self.b,x))
 def social(self,x):return sum(n*(a+b*n) for a,b,n in zip(self.a,self.b,x))
 def alternatives(self,x,r):
  z=x.copy()
  for e in self.routes[r]:z[e]-=1
  return [sum(self.a[e]+self.b[e]*(z[e]+1) for e in q) for q in self.routes]
 def exact(self,n):
  opt=10**30;pmin=10**30;lex=10**30;nes=[];states=0
  for f in comps(n,self.k):
   states+=1;x=self.loads(f);s=self.social(x);p=self.phi(x);opt=min(opt,s)
   if p<pmin:pmin=p;lex=s
   elif p==pmin:lex=min(lex,s)
   ok=True
   for r,users in enumerate(f):
    if users:
     cc=sum(self.a[e]+self.b[e]*x[e] for e in self.routes[r])
     if min(self.alternatives(x,r))<cc:ok=False;break
   if ok:nes.append((f,s,p))
  return {'states':states,'social_optimum':opt,'phi_min':pmin,'lex_phi_social_cost':lex,'ne_count':len(nes),'ne_best':min(z[1] for z in nes),'ne_worst':max(z[1] for z in nes),'eq':nes}
def compare(ex,stored):
 for k in ['states','social_optimum','phi_min','lex_phi_social_cost','ne_count','ne_best','ne_worst']:assert ex[k]==stored[k],(k,ex[k],stored[k])
 assert sorted(ex['eq'])==sorted((tuple(z['counts']),z['social_cost'],z['potential']) for z in stored['equilibria'])
def category(changed,base):
 lo,hi=changed['ne_best'],changed['ne_worst'];bl,bh=base['ne_best'],base['ne_worst']
 if hi<bl:return 'robust_benefit'
 if lo>bh:return 'robust_harm'
 if lo==hi==bl==bh:return 'equal'
 if hi==bl:return 'weak_benefit'
 if lo==bh:return 'weak_harm'
 return 'overlap'
es=read('network.json')['edges'];base=Model(es);report={};states_total=0
z=base.exact(12);compare(z,read('exact_N12_all_states.json'));states_total+=z['states'];report['baseline']=dict(states=z['states'],social_optimum=z['social_optimum'],phi_min=z['phi_min'],lex_cost=z['lex_phi_social_cost'],ne_count=z['ne_count'])
# Independent rational interval derivation, using direct remove/enter costs at mu=0 and mu=1.
zid=[j for j,e in enumerate(base.es) if e[:2]==('C','D')][0]
e0=[(u,v,0 if (u,v)==('C','D') else a,b) for u,v,a,b in es];e1=[(u,v,1 if (u,v)==('C','D') else a,b) for u,v,a,b in es];m0=Model(e0);m1=Model(e1)
intervals=[]
for f in comps(12,5):
 x=m0.loads(f);low=Fraction(0);high=Fraction(12);ok=True
 for r,users in enumerate(f):
  if not users:continue
  c0=m0.alternatives(x,r);c1=m1.alternatives(x,r)
  for q in range(5):
   intercept=c0[r]-c0[q];slope=(c1[r]-c0[r])-(c1[q]-c0[q])
   if slope>0:high=min(high,Fraction(-intercept,slope))
   elif slope<0:low=max(low,Fraction(-intercept,slope))
   elif intercept>0:ok=False
 if ok and low<=high:intervals.append((tuple(f),low,high,m0.social(x),x[zid]))
saved=read('continuous_NE_intervals.json')
assert sorted(intervals)==sorted((tuple(r['counts']),Fraction(r['mu_min']),Fraction(r['mu_max']),r['social_intercept'],r['social_slope']) for r in saved)
report['continuous_intervals_verified']=len(intervals)
print('baseline and intervals verified',flush=True)
# Regenerate all 300 three-rule BR runs using direct scalar single-driver costs.
def own_br(m,initial,rule,seed):
 s=list(initial);rng=random.Random(seed);moves=queries=0
 while True:
  changed=False;choices=[]
  order=list(range(len(s)))
  if rule=='random':rng.shuffle(order)
  f=[s.count(p) for p in range(m.k)];x=m.loads(f)
  for i in order:
   old=s[i];cc=m.alternatives(x,old);queries+=1;best=cc.index(min(cc));gain=cc[old]-cc[best]
   if rule=='max_gain':choices.append((gain,i,best))
   elif gain>0:
    s[i]=best;moves+=1;changed=True
    for e in m.routes[old]:x[e]-=1
    for e in m.routes[best]:x[e]+=1
  if rule=='max_gain':
   top=max(c[0] for c in choices)
   if top==0:break
   _,i,best=next(c for c in choices if c[0]==top);s[i]=best;moves+=1
  elif not changed:break
 return dict(counts=[s.count(p) for p in range(m.k)],moves=moves,driver_queries=queries,social_cost=m.social(m.loads([s.count(p) for p in range(m.k)])))
rows=list(csv.DictReader((root/'algorithm_runs.csv').open(encoding='utf-8-sig')))
for r in rows:
 rr=own_br(base,json.loads(r['initial']),r['rule'],int(r['seed']))
 assert rr['counts']==json.loads(r['counts'])
 for key in ['moves','driver_queries','social_cost']:assert rr[key]==int(r[key]),(key,rr,r)
report['independent_br_runs']=len(rows)
report['algorithms']={rule:dict(Counter(int(r['social_cost']) for r in rows if r['rule']==rule)) for rule in ['cyclic','random','max_gain']}
print('300 scalar BR runs verified',flush=True)
# Verify every parameter-ensemble state and every topology-ensemble state independently.
pars=read('parameter_ensemble.json');counter=Counter();lexfails=[];ensemble_states=0
for p in pars:
 mm=Model(p['edges']);e=mm.exact(8);compare(e,p['base']);ensemble_states+=e['states']
 variants={'delete_CD':[q for q in p['edges'] if tuple(q[:2])!=('C','D')],'add_BC':p['edges']+[['B','C',2,0]],'improve_DT':[(u,v,a,max(0,b-1) if (u,v)==('D','T') else b) for u,v,a,b in p['edges']]}
 for name,edges in variants.items():
  ee=Model(edges).exact(8);compare(ee,p['interventions'][name]['exact']);ensemble_states+=ee['states'];cat=category(ee,e);assert cat==p['interventions'][name]['category'];counter[(name,cat)]+=1
 assert (e['lex_phi_social_cost']==e['ne_best'])==p['lex_equals_best_ne']
 if not p['lex_equals_best_ne']:lexfails.append(p['sample'])
report['parameter_ensemble']={name:{cat:count for (nn,cat),count in counter.items() if nn==name} for name in variants};report['lex_not_best_parameter']=lexfails
print('100 parameter ensembles verified',flush=True)
tops=read('topology_ensemble.json');tc=Counter();lexfailt=[]
for p in tops:
 e=Model(p['edges']).exact(6);compare(e,p['base']);ensemble_states+=e['states'];reduced=[q for q in p['edges'] if q[:2]!=p['removed_edge']];ee=Model(reduced).exact(6);compare(ee,p['changed']);ensemble_states+=ee['states'];cat=category(ee,e);assert cat==p['category'];tc[cat]+=1
 assert (e['lex_phi_social_cost']==e['ne_best'])==p['lex_equals_best_ne']
 if not p['lex_equals_best_ne']:lexfailt.append(p['sample'])
report['topology_ensemble']=dict(tc);report['lex_not_best_topology']=lexfailt;report['ensemble_states_independently_enumerated']=ensemble_states
print('40 topology ensembles verified',flush=True)
# Check complete phase grid using direct scalar enumeration.
phase=read('phase_all_equilibria.json');pc=Counter();phase_states=0;without_cache={}
for r in phase:
 n=r['n'];mu=r['mu'];m=Model([(u,v,mu if (u,v)==('C','D') else a,b) for u,v,a,b in es]);ex=m.exact(n);compare(ex,r['with_edge']);phase_states+=ex['states']
 if n not in without_cache:
  without_cache[n]=Model([q for q in es if tuple(q[:2])!=('C','D')]).exact(n);phase_states+=without_cache[n]['states']
 wo=without_cache[n];compare(wo,r['without_edge']);pc[category(ex,wo)]+=1
report['phase_cells_verified']=len(phase);report['phase_categories']=dict(pc);report['phase_states_independently_enumerated']=phase_states
# Use own full enumeration as the exact independent reference for all three project flow objectives.
flow=read('flow_N12.json')
for name,r in flow.items():
 x=r['edge_flows'];phi=base.phi(x);social=base.social(x)
 assert phi==r['potential'] and social==r['social_cost']
 assert name!='potential' or phi==z['phi_min']
 assert name!='lex' or (phi,social)==(z['phi_min'],z['lex_phi_social_cost'])
 assert name!='social' or social==z['social_optimum']
report['flows_checked_by_full_enumeration']=3
(root.parent/'independent_review_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
