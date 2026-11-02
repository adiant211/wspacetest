
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
            W[1][u(4+colour,p),u(4+colour,p)] = 1

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

            j = u(4+colour,p)
            B[2][j] = len(adj[p])
            for q in adj[p]:
                W[2][u(0,q),j] -= 1
                W[2][u(1,q),j] -= 1
                W[2][u(2+colour,q),j] += 1

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
