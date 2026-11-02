
import numpy as np

DEFAULTS = dict(
    capture=24.0, rescue=8.0, attack=1.15,
    self_atari=7.5, eye=16.0, connection=0.9, contact=1.4,
    own_territory=1.6, territory_threshold=3.0,
    pass_base=-9.0, pass_per_stone=0.10, pass_white=0.35,
    own_adj=0.25, own_diag=0.65, own_two=0.40, own_knight=0.12,
    enemy_adj=0.75, enemy_diag=0.25, enemy_two=0.12,
    temperature_scale=1.8,
)
def build(save, overrides=None):
    c=dict(DEFAULTS)
    if overrides: c.update(overrides)
    W=[np.zeros((243 if k==0 else 512,512),np.float32) for k in range(6)]
    B=[np.zeros(512,np.float32) for k in range(6)]
    OW=np.zeros((512,82),np.float32); OB=np.zeros(82,np.float32)
    def u(ch,p): return 81*ch+p
    adj=[]; diag=[]; rings=[]
    for p in range(81):
        r,s=divmod(p,9)
        adj.append([a*9+b for a,b in [(r-1,s),(r+1,s),(r,s-1),(r,s+1)]
                    if 0<=a<9 and 0<=b<9])
        diag.append([a*9+b for a,b in [(r-1,s-1),(r-1,s+1),(r+1,s-1),(r+1,s+1)]
                     if 0<=a<9 and 0<=b<9])
        rings.append([(q,abs(r-q//9),abs(s-q%9)) for q in range(81) if q!=p])
    for p in range(81):
        W[0][3*p,u(0,p)]=1; W[0][3*p+1,u(1,p)]=1
        for colour in [0,1]:
            j=u(2+colour,p)
            B[0][j]=len(adj[p])-4
            W[0][3*p+colour,j]=4
            for q in adj[p]:
                W[0][3*q,j]-=1; W[0][3*q+1,j]-=1
        # Contact and connection features, directly from the board.
        B[0][u(4,p)]=-1; B[0][u(5,p)]=-1
        for q in adj[p]:
            W[0][3*q+1,u(4,p)]+=1
            W[0][3*q,u(4,p)]-=0.75
            W[0][3*q,u(5,p)]+=1
        for q in diag[p]:
            W[0][3*q,u(4,p)]-=0.20
            W[0][3*q,u(5,p)]-=0.65
        W[0][3*p,486]=1; W[0][3*p+1,487]=1
    W[0][2,488]=1
    for k in range(1,6):
        for g in [486,487,488]: W[k][g,g]=1
    def board(k):
        for col in [0,1]:
            for p in range(81): W[k][u(col,p),u(col,p)]=1

    board(1)
    line=[-1.5,0.25,1.2,0.9,1.0]
    for p in range(81):
        for colour in [0,1]:
            j=u(2+colour,p)
            B[1][j]=-21; W[1][u(colour,p),j]=20
            for q in [p]+adj[p]: W[1][u(2+colour,q),j]+=1
        j=u(4,p); B[1][j]=-c["territory_threshold"]
        for q,dr,dc in rings[p]:
            if dr+dc==1: a=1.0
            elif dr==dc==1: a=0.65
            elif dr+dc==2: a=0.45
            elif sorted([dr,dc])==[1,2]: a=0.20
            else: continue
            W[1][u(0,q),j]+=a; W[1][u(1,q),j]-=1.2*a
        j=u(5,p); r,s=divmod(p,9)
        B[1][j]=100+line[min(r,8-r)]+line[min(s,8-s)]
        W[1][u(4,p),j]=-c["contact"]
        W[1][u(5,p),j]=c["connection"]
        for q,dr,dc in rings[p]:
            if dr+dc==1: a,b=c["own_adj"],c["enemy_adj"]
            elif dr==dc==1: a,b=c["own_diag"],c["enemy_diag"]
            elif dr+dc==2: a,b=c["own_two"],c["enemy_two"]
            elif sorted([dr,dc])==[1,2]: a,b=c["own_knight"],0
            else: continue
            W[1][u(0,q),j]+=a; W[1][u(1,q),j]+=b

    # Propagate liberty evidence for three more layers.
    # Conservative analytic upper bounds make every colour gate exact.
    for k,gate in [(2,100),(3,500)]:
        board(k)
        for p in range(81):
            for colour in [0,1]:
                j=u(2+colour,p)
                B[k][j]=-gate; W[k][u(colour,p),j]=gate
                for q in [p]+adj[p]: W[k][u(2+colour,q),j]+=1
            if k==2:
                j=u(4,p); B[k][j]=2-len(adj[p])
                for q in adj[p]:
                    W[k][u(0,q),j]+=1
                    W[k][u(1,q),j]+=1
                    W[k][u(2,q),j]-=1
                W[k][u(5,p),u(5,p)]=1
                W[k][u(4,p),u(5,p)]=-c["own_territory"]
            else:
                W[k][u(4,p),u(4,p)]=1
                W[k][u(5,p),u(5,p)]=1

    # Friendly danger now reaches four links; enemy safety reaches four.
    board(4)
    for p in range(81):
        j=u(2,p); W[4][u(0,p),j]=1
        for q in [p]+adj[p]: W[4][u(2,q),j]-=1
        j=u(3,p); B[4][j]=-2500; W[4][u(1,p),j]=2500
        for q in [p]+adj[p]: W[4][u(3,q),j]+=1
        W[4][u(4,p),u(4,p)]=1
        W[4][u(5,p),u(5,p)]=1

    for p in range(81):
        W[5][u(5,p),u(0,p)]=1
        j=u(1,p)
        W[5][u(4,p),j]=-4
        for q in adj[p]: W[5][u(2,q),j]+=1
        j=u(2,p); W[5][u(1,p),j]=1
        for q in [p]+adj[p]: W[5][u(3,q),j]-=1
        W[5][u(4,p),u(3,p)]=1
        j=u(4,p); B[5][j]=1-len(adj[p])
        for q in adj[p]: W[5][u(0,q),j]+=1
        j=u(5,p); W[5][u(1,p),j]=3
        W[5][u(3,p),j]=-1
        for q in adj[p]: W[5][u(3,q),j]-=0.5

        OW[u(0,p),p]=1
        OW[u(1,p),p]=c["rescue"]
        OW[u(3,p),p]=-c["self_atari"]
        OW[u(4,p),p]=-c["eye"]
        for q in adj[p]:
            OW[u(2,q),p]+=c["capture"]
            OW[u(5,q),p]+=c["attack"]
        OB[p]=-100

    OB[81]=c["pass_base"]
    OW[486,81]=OW[487,81]=c["pass_per_stone"]
    OW[488,81]=c["pass_white"]
    OW*=c["temperature_scale"]; OB*=c["temperature_scale"]
    ts={}
    for k in range(6):
        ts["L%d.W"%(k+1)]=W[k]; ts["L%d.b"%(k+1)]=B[k]
    ts["out.W"]=OW; ts["out.b"]=OB
    for n,a in ts.items():
        assert np.isfinite(a).all()
        save(n,a)
    return c,ts
