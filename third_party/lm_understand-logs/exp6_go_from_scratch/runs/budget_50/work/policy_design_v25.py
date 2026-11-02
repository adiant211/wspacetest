
import numpy as np

DEFAULTS = dict(
    capture=9.0,
    rescue=6.0,
    attack=1.15,
    self_atari=7.5,
    eye=11.0,
    connection=0.9,
    contact=1.4,
    own_territory=1.6,
    enemy_territory=1.0,
    territory_threshold=3.0,
    pass_base=-9.0,
    pass_per_stone=0.10,
    pass_white=0.35,
    own_adj=0.25,
    own_diag=0.65,
    own_two=0.40,
    own_knight=0.12,
    enemy_adj=0.75,
    enemy_diag=0.25,
    enemy_two=0.12,
    temperature_scale=1.8,
    surrounded_penalty=10.0,
)

def build(save, overrides=None):
    cfg = dict(DEFAULTS)
    if overrides:
        cfg.update(overrides)

    W = [np.zeros((243 if k == 0 else 512, 512), np.float32)
         for k in range(6)]
    B = [np.zeros(512, np.float32) for _ in range(6)]
    OW = np.zeros((512, 82), np.float32)
    OB = np.zeros(82, np.float32)

    def u(ch, p):
        return 81*ch+p

    adj = []
    diag = []
    rings = []
    for p in range(81):
        r, c = divmod(p, 9)
        adj.append([rr*9+cc for rr, cc in
                    [(r-1,c), (r+1,c), (r,c-1), (r,c+1)]
                    if 0 <= rr < 9 and 0 <= cc < 9])
        diag.append([rr*9+cc for rr, cc in
                     [(r-1,c-1), (r-1,c+1), (r+1,c-1), (r+1,c+1)]
                     if 0 <= rr < 9 and 0 <= cc < 9])
        rings.append([(q, abs(r-q//9), abs(c-q%9))
                      for q in range(81) if q != p])

    # Layer 1: board, exact local liberties, and excess liberties.
    for p in range(81):
        W[0][3*p, u(0,p)] = 1
        W[0][3*p+1, u(1,p)] = 1
        for colour in range(2):
            for ch, excess in [(2+colour, 0), (4+colour, 1)]:
                j = u(ch,p)
                B[0][j] = len(adj[p])-4-excess
                W[0][3*p+colour,j] += 4
                for q in adj[p]:
                    W[0][3*q,j] -= 1
                    W[0][3*q+1,j] -= 1

    # A zero-liberty indicator lets the next layer distinguish a
    # duplicated single liberty around a bent chain from independent ones.
    for p in range(81):
        for colour in range(2):
            j=u(4+colour,p)
            W[0][:,j]=0
            B[0][j]=-len(adj[p])
            W[0][3*p+colour,j]=1
            for q in adj[p]:
                W[0][3*q,j]+=1
                W[0][3*q+1,j]+=1

    # Global counts and colour are carried separately.
    for p in range(81):
        W[0][3*p,486] = 1
        W[0][3*p+1,487] = 1
    W[0][2,488] = 1

    for k in range(1,6):
        for g in [486,487,488]:
            W[k][g,g] = 1

    def carry_board(k):
        for ch in [0,1]:
            for p in range(81):
                W[k][u(ch,p),u(ch,p)] = 1

    # Layer 2: liberty evidence along monochromatic connections.
    # Orthogonally adjacent stones cannot share an immediate liberty.
    carry_board(1)
    for p in range(81):
        for colour in range(2):
            j = u(2+colour,p)
            B[1][j] = -21
            W[1][u(colour,p),j] = 20
            for q in [p]+adj[p]:
                W[1][u(2+colour,q),j] += 1
            W[1][u(4+colour,p),j] -= 1

            # Independent supplementary safety certificate. A shared
            # liberty of two neighbours must lie diagonally from p.
            j=u(4+colour,p)
            B[1][j]=-21-len(diag[p])
            W[1][u(colour,p),j]=20
            for q in [p]+adj[p]:
                W[1][u(2+colour,q),j]+=1
            for q in diag[p]:
                W[1][u(0,q),j]+=1
                W[1][u(1,q),j]+=1

    # Layer 3: spread evidence of safety through connected stones.
    # The other channels estimate liberties available after extension.
    carry_board(2)
    for p in range(81):
        for colour in range(2):
            j = u(2+colour,p)
            B[2][j] = -100
            W[2][u(colour,p),j] = 100
            for q in [p]+adj[p]:
                W[2][u(2+colour,q),j] += 1
                W[2][u(4+colour,q),j] += 1

            j = u(4+colour,p)
            B[2][j] = len(adj[p])
            for q in adj[p]:
                W[2][u(0,q),j] -= 1
                W[2][u(1,q),j] -= 1
                W[2][u(2+colour,q),j] += 1
                W[2][u(4+colour,q),j] += 1

    # Exact one-stone capture detector in a previously unused channel.
    for p in range(81):
        j=u(5,p)
        W[2][:,j]=0
        B[2][j]=-len(adj[p])
        W[2][u(1,p),j]=2
        for q in adj[p]:
            W[2][u(0,q),j]+=1
            W[2][u(1,q),j]-=1

    # Layer 4: approximate whole-group atari, self-atari, weak enemies.
    carry_board(3)
    for p in range(81):
        for colour in range(2):
            j = u(2+colour,p)
            W[3][u(colour,p),j] = 1
            for q in [p]+adj[p]:
                W[3][u(2+colour,q),j] -= 1

        B[3][u(4,p)] = 2
        W[3][u(4,p),u(4,p)] = -1

        j = u(5,p)
        W[3][u(1,p),j] = 3
        W[3][u(3,p),j] = -1
        for q in adj[p]:
            W[3][u(3,q),j] -= 0.5

    # Layer 5: spatial prior, captures, attack, safe rescue, and
    # capture-aware self-atari suppression.
    carry_board(4)
    line_prior = [-1.5, 0.25, 1.2, 0.9, 1.0]
    for p in range(81):
        r,c = divmod(p,9)
        j = u(2,p)
        B[4][j] = (100 + line_prior[min(r,8-r)] +
                   line_prior[min(c,8-c)])
        for q,dr,dc in rings[p]:
            if dr+dc == 1:
                a,b = cfg["own_adj"],cfg["enemy_adj"]
            elif dr == 1 and dc == 1:
                a,b = cfg["own_diag"],cfg["enemy_diag"]
            elif dr+dc == 2:
                a,b = cfg["own_two"],cfg["enemy_two"]
            elif sorted([dr,dc]) == [1,2]:
                a,b = cfg["own_knight"],0.0
            else:
                continue
            W[4][u(0,q),j] += a
            W[4][u(1,q),j] += b
        for q in adj[p]:
            W[4][u(3,q),j] += cfg["capture"]
            W[4][u(5,q),j] += cfg["attack"]

        j = u(3,p)
        W[4][u(4,p),j] = 1
        for q in adj[p]:
            W[4][u(3,q),j] -= 2

        j = u(4,p)
        W[4][u(4,p),j] = -4
        for q in adj[p]:
            W[4][u(2,q),j] += 1

    # Empty points surrounded by friendly stones are normally already
    # owned. Preserve them even when diagonal enemies make them false eyes.
    for p in range(81):
        j = u(5,p)
        B[4][j] = 1-len(adj[p])
        for q in adj[p]:
            W[4][u(0,q),j] = 1

    # Layer 6: combine tactics with connection, contact, territory,
    # and true-eye features.
    for p in range(81):
        j = u(0,p)
        W[5][u(2,p),j] = 1
        W[5][u(3,p),j] = -cfg["self_atari"]
        W[5][u(4,p),j] = cfg["rescue"]
        W[5][u(5,p),j] = -cfg["surrounded_penalty"]

        # Connect stones without repeatedly rewarding filled triangles.
        j = u(1,p)
        B[5][j] = -1
        for q in adj[p]:
            W[5][u(0,q),j] = 1
        for q in diag[p]:
            W[5][u(0,q),j] = -0.65

        # Avoid unsupported contact with multiple enemy stones.
        j = u(2,p)
        B[5][j] = -1
        for q in adj[p]:
            W[5][u(1,q),j] = 1
            W[5][u(0,q),j] = -0.75
        for q in diag[p]:
            W[5][u(0,q),j] -= 0.20

        # Discourage both filling established territory and hopeless
        # invasions; urgent tactical bonuses can override either.
        for colour,ch in [(0,3),(1,4)]:
            j = u(ch,p)
            B[5][j] = -cfg["territory_threshold"]
            for q,dr,dc in rings[p]:
                if dr+dc == 1:
                    weight = 1.0
                elif dr == 1 and dc == 1:
                    weight = 0.65
                elif dr+dc == 2:
                    weight = 0.45
                elif sorted([dr,dc]) == [1,2]:
                    weight = 0.20
                else:
                    continue
                W[5][u(colour,q),j] += weight
                W[5][u(1-colour,q),j] -= 1.2*weight

        # A fully friendly orthogonal surround, adjusted for false eyes.
        j = u(5,p)
        B[5][j] = 1-len(adj[p])
        for q in adj[p]:
            W[5][u(0,q),j] += 1
        diagonal_penalty = 1.0 if len(adj[p]) < 4 else 0.5
        for q in diag[p]:
            W[5][u(1,q),j] -= diagonal_penalty

        for ch,coefficient in [
            (0,1.0),
            (1,cfg["connection"]),
            (2,-cfg["contact"]),
            (3,-cfg["own_territory"]),
            (4,-cfg["enemy_territory"]),
            (5,-cfg["eye"]),
        ]:
            OW[u(ch,p),p] = coefficient
        OB[p] = -100

    OB[81] = cfg["pass_base"]
    OW[486,81] = cfg["pass_per_stone"]
    OW[487,81] = cfg["pass_per_stone"]
    OW[488,81] = cfg["pass_white"]


    # At layer 4, encode own/empty/enemy as 1, 1/2, 0. This frees
    # a full channel for repetition-risk detection without losing attack.
    for p in range(81):
        old_weak=W[3][:,u(5,p)].copy()
        old_bias=B[3][u(5,p)]
        W[3][:,u(1,p)]=old_weak
        B[3][u(1,p)]=old_bias
        W[3][:,u(0,p)]=0
        B[3][u(0,p)]=0.5
        W[3][u(0,p),u(0,p)]=0.5
        W[3][u(1,p),u(0,p)]=-0.5

        j=u(5,p)
        W[3][:,j]=0
        B[3][j]=-len(adj[p])
        for q in adj[p]:
            W[3][u(5,q),j]+=2
            W[3][u(0,q),j]-=3
            W[3][u(1,q),j]+=1

        # Decode the board for the final shape layer.
        for ch,sign,bias in [(0,2,-1),(1,-2,1)]:
            j=u(ch,p)
            W[4][:,j]=0
            B[4][j]=bias
            W[4][u(0,p),j]=sign

        # Tactical score only here; move the spatial prior to layer 6.
        j=u(2,p)
        W[4][:,j]=0
        B[4][j]=100
        for q in adj[p]:
            W[4][u(3,q),j]+=cfg["capture"]
            W[4][u(1,q),j]+=cfg["attack"]
        W[4][u(5,p),j]-=cfg.get("repeat_risk",18.0)

        j=u(5,p)
        W[4][:,j]=0
        B[4][j]=1-2*len(adj[p])
        for q in adj[p]:
            W[4][u(0,q),j]=2

        j=u(0,p)
        r,c=divmod(p,9)
        B[5][j]=line_prior[min(r,8-r)]+line_prior[min(c,8-c)]
        for q,dr,dc in rings[p]:
            if dr+dc==1:
                a,b=cfg["own_adj"],cfg["enemy_adj"]
            elif dr==dc==1:
                a,b=cfg["own_diag"],cfg["enemy_diag"]
            elif dr+dc==2:
                a,b=cfg["own_two"],cfg["enemy_two"]
            elif sorted([dr,dc])==[1,2]:
                a,b=cfg["own_knight"],0.0
            else:
                continue
            W[5][u(0,q),j]+=a
            W[5][u(1,q),j]+=b


    # An endangered stone with several friendly connections is more
    # likely to represent a substantial group than an isolated stone.
    for p in range(81):
        j=u(2,p)
        W[3][:,j]=0
        B[3][j]=-4
        W[3][u(0,p),j]=5
        for q in adj[p]:
            W[3][u(0,q),j]+=cfg.get("rescue_size",0.4)
        for q in [p]+adj[p]:
            W[3][u(2,q),j]-=10
        W[4][u(4,p),u(4,p)]=-12


    # Pack colour one layer earlier, freeing a channel for short-range
    # weakness. Capture certainty still uses the longer safety field.
    for p in range(81):
        j=u(0,p)
        W[2][:,j]=0
        B[2][j]=0.5
        W[2][u(0,p),j]=0.5
        W[2][u(1,p),j]=-0.5

        j=u(1,p)
        W[2][:,j]=0
        B[2][j]=0
        W[2][u(3,p),j]=1

        j=u(0,p)
        W[3][:,j]=0
        B[3][j]=0
        W[3][u(0,p),j]=1

        j=u(1,p)
        W[3][:,j]=0
        B[3][j]=3
        W[3][u(0,p),j]=-20
        W[3][u(1,p),j]=-1
        for q in adj[p]:
            W[3][u(1,q),j]-=0.5

        j=u(2,p)
        W[3][:,j]=0
        size=cfg.get("rescue_size",0.4)
        B[3][j]=1-0.5*size-20
        W[3][u(0,p),j]=20
        for q in adj[p]:
            W[3][u(0,q),j]+=size
        for q in [p]+adj[p]:
            W[3][u(2,q),j]-=10

        j=u(3,p)
        W[3][:,j]=0
        B[3][j]=1
        W[3][u(0,p),j]=-20
        for q in [p]+adj[p]:
            W[3][u(3,q),j]-=1

        j=u(5,p)
        W[3][:,j]=0
        B[3][j]=0
        for q in adj[p]:
            W[3][u(5,q),j]+=2
            W[3][u(0,q),j]-=2


    # Value additional territory at the contested boundary more than
    # safe consolidation inside an existing sphere of influence.
    for p in range(81):
        B[5][u(3,p)]=-cfg.get("own_territory_threshold",0.5)


    # A third-line framework is economical on 9x9: it can limit a
    # central group while retaining room for eyes on the outside.
    for p in range(81):
        r,c=divmod(p,9)
        if max(abs(r-4),abs(c-4))==2:
            OB[p]+=cfg.get("framework_bonus",2.0)


    # Compress the two material counts into one total, leaving 24
    # auxiliary units for stronger edge self-atari certification.
    for k in range(6):
        W[k][:,486:]=0
        B[k][486:]=0
        if k:
            W[k][486:,:]=0
    OW[486:,:]=0
    for p in range(81):
        W[0][3*p,486]=1
        W[0][3*p+1,486]=1
    W[0][2,487]=1
    for k in range(1,6):
        W[k][486,486]=1
        W[k][487,487]=1
    OW[486,81]=cfg["pass_per_stone"]
    OW[487,81]=cfg["pass_white"]

    edge_points=[]
    for p in range(81):
        r,c=divmod(p,9)
        if r in [0,8] and c in [0,1,2,4,6,7,8]:
            edge_points.append(p)
        elif c in [0,8] and r in [1,2,4,6,7]:
            edge_points.append(p)
    assert len(edge_points)==24
    for i,p in enumerate(edge_points):
        g=488+i
        # Exact singleton captures, rather than approximate group danger.
        for q in adj[p]:
            W[3][u(5,q),g]+=1

        # Multiple threatened contacts can justify a larger capture.
        B[4][g]=-1
        for q in adj[p]:
            W[4][u(3,q),g]+=1

        j=u(3,p)
        W[4][:,j]=0
        B[4][j]=0
        W[4][u(4,p),j]=1
        W[4][g,j]=-2

        W[5][u(3,p),g]=1
        W[5][g,g]=-2
        W[5][u(3,p),u(0,p)]=0
        OW[g,p]=-cfg.get("edge_self_atari",32.0)

    OW *= cfg["temperature_scale"]
    OB *= cfg["temperature_scale"]

    tensors = {}
    for k in range(6):
        tensors["L%d.W"%(k+1)] = W[k]
        tensors["L%d.b"%(k+1)] = B[k]
    tensors["out.W"] = OW
    tensors["out.b"] = OB
    for name,array in tensors.items():
        assert np.isfinite(array).all()
        save(name,array)
    return cfg, tensors
