
import numpy as np
from pathlib import Path

def build_policy(cfg=None):
    p = dict(capture=11., capture_extra=3., save=7., danger=10.,
             eye=22., isolation=3., isolation_threshold=2.5,
             base=1.3, pass_bias=-1., attack=1.3,
             own_adj=.65, opp_adj=.3, opening=2.0,
             save_danger=2., safe2=1., safe3=1., cap_safety=1., crowd=0., temperature_scale=1., defend=0., first_center=0., triangle=0.)
    if cfg:
        p.update(cfg)
    shapes = {'L1.W':(243,512), 'out.W':(512,82), 'out.b':(82,)}
    for k in range(1,7):
        shapes[f'L{k}.b']=(512,)
        if k>1: shapes[f'L{k}.W']=(512,512)
    a = {n:np.zeros(s,np.float32) for n,s in shapes.items()}
    def u(ch,q): return ch*81+q
    xy=[divmod(q,9) for q in range(81)]
    adj=[]
    for r,c in xy:
        adj.append([rr*9+cc for rr,cc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)] if 0<=rr<9 and 0<=cc<9])
    def w(k,i,j,v): a[f'L{k}.W'][i,j]+=v
    def b(k,j,v): a[f'L{k}.b'][j]+=v
    def cp(k,ch):
        for q in range(81): w(k,u(ch,q),u(ch,q),1.)
    for q in range(81):
        w(1,3*q,u(0,q),1)
        w(1,3*q+1,u(1,q),1)
        for color, ch in [(0,2),(1,4)]:
            for off in [0,1]:
                j=u(ch+off,q)
                b(1,j,len(adj[q])-4-off)
                w(1,3*q+color,j,4)
                for t in adj[q]:
                    w(1,3*t,j,-1)
                    w(1,3*t+1,j,-1)
    # Global, phase-dependent positional prior; no clock or history is assumed.
    b(1,486,1.)
    for q in range(81):
        w(1,3*q,486,-1/30)
        w(1,3*q+1,486,-1/30)
    b(1,487,1)
    for q in range(81):
        w(1,3*q,487,-1)
        w(1,3*q+1,487,-1)
    for k in range(2,5):
        cp(k,0); cp(k,1)
        w(k,486,486,1); w(k,487,487,1)

    # Safe-group seeds: two liberties on one stone, or liberties on
    # two adjacent stones (which cannot be the same empty intersection).
    for q in range(81):
        for color,ch in [(0,2),(1,4)]:
            ja,jb=u(ch,q),u(ch+1,q)
            w(2,u(color,q),ja,1)
            for t in [q]+adj[q]: w(2,u(ch+1,t),ja,-1)
            b(2,jb,-4)
            w(2,u(ch,q),jb,4)
            w(2,u(ch+1,q),jb,-4)
            for t in adj[q]:
                w(2,u(ch,t),jb,1)
                w(2,u(ch+1,t),jb,-1)

    for q in range(81):
        for color,prev,new,seed in [(0,2,2,4),(1,4,3,5)]:
            j=u(new,q)
            w(3,u(prev,q),j,1)
            w(3,u(prev+1,q),j,-1)
            for t in adj[q]:
                w(3,u(color,t),j,-1)
                w(3,u(prev,t),j,1)
                w(3,u(prev+1,t),j,-1)
            j=u(seed,q)
            b(3,j,len(adj[q])-6)
            w(3,u(color,q),j,4)
            for t in adj[q]:
                w(3,u(0,t),j,-1); w(3,u(1,t),j,-1)
    for q in range(81):
        for color,unsafe,seed in [(0,2,4),(1,3,5)]:
            j=u(unsafe,q)
            w(4,u(unsafe,q),j,1)
            for t in adj[q]:
                w(4,u(color,t),j,-1)
                w(4,u(unsafe,t),j,1)
            j=u(seed,q)
            w(4,u(color,q),j,1)
            for t in [q]+adj[q]: w(4,u(seed,t),j,-1)

    central_points=[q for q,(r,c) in enumerate(xy) if 2<=r<=6 and 2<=c<=6 and q!=40]
    shape_units={q:488+i for i,q in enumerate(central_points)}
    for q,j in shape_units.items():
        r,c=xy[q]
        b(4,j,-2)
        for t in adj[q]: w(4,u(0,t),j,1)
        for dr,dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
            w(4,u(0,(r+dr)*9+c+dc),j,1)
    line=[-1.2,0.,.6,.8,.9,.8,.6,0.,-1.2]
    for q,(r,c) in enumerate(xy):
        C,S,D,E,I,P=[u(ch,q) for ch in range(6)]
        b(5,D,2-len(adj[q]))
        b(5,E,2-len(adj[q]))
        b(5,I,-p['isolation_threshold'])
        b(5,P,40+p['base']+line[r]+line[c])
        if q in shape_units: w(5,shape_units[q],P,-p['triangle'])
        central=max(0.,1.-((r-4)**2+(c-4)**2)/24)
        w(5,486,P,p['opening']*central)
        if q==40: w(5,487,P,p['first_center'])
        for t in adj[q]:
            w(5,u(3,t),C,1)
            w(5,u(2,t),S,1)
            w(5,u(0,t),D,1-p['safe2']-p['safe3'])
            w(5,u(1,t),D,1)
            w(5,u(2,t),D,p['safe2'])
            w(5,u(4,t),D,p['safe3'])
            w(5,u(3,t),D,-p['cap_safety'])
            w(5,u(0,t),E,1)
            w(5,u(5,t),P,p['attack'])
            w(5,u(3,t),P,-p['attack'])
            w(5,u(4,t),P,p['defend'])
            w(5,u(2,t),P,-p['defend'])
        for t,(rr,cc) in enumerate(xy):
            dr,dc=abs(rr-r),abs(cc-c)
            d=dr+dc
            if d==1:
                wm,wo=p['own_adj'],p['opp_adj']
            elif d==2:
                wm,wo=(-.10,.15) if dr==dc==1 else (.15,.35)
            elif d==3:
                wm,wo=-.12,.10
            elif d==4:
                wm,wo=-.08,0.
            else:
                wm=wo=0.
            if wm: w(5,u(0,t),P,wm)
            if wo: w(5,u(1,t),P,wo)
            iw={1:1.,2:.6,3:.25}.get(d,0.)
            if iw:
                w(5,u(1,t),I,iw)
                w(5,u(0,t),I,-1.25*iw)
        w(6,C,u(0,q),1)
        w(6,C,u(1,q),1); b(6,u(1,q),-1)
        w(6,S,u(2,q),1); w(6,D,u(2,q),-p['save_danger'])
        w(6,D,u(3,q),1)
        w(6,E,u(4,q),1); b(6,u(4,q),-1)
        w(6,P,u(5,q),1)
        b(6,u(5,q),100)
        w(6,E,u(5,q),-p['crowd'])
        w(6,I,u(5,q),-p['isolation'])
        for ch,v in enumerate([p['capture'],p['capture_extra']-p['capture'],
                               p['save'],-p['danger'],-(p['eye']-2*p['crowd']),1.]):
            a['out.W'][u(ch,q),q]=v
        a['out.b'][q]=-140.
    a['out.b'][81]=p['pass_bias']
    a['out.W'] *= p['temperature_scale']
    a['out.b'] *= p['temperature_scale']
    return a,p

def install(cfg=None):
    a,p=build_policy(cfg)
    for n,v in a.items(): save(n,v)
    return p
