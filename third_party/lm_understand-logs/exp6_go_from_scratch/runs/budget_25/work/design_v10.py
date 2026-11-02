
import numpy as np

P = dict(
    capture=7.0, rescue=4.5, self_atari=10.0,
    eye=11.0, crowd=2.5, pass_bias=-2.0,
    friend_adj=0.15, enemy_adj=0.35,
    friend_diag=0.35, enemy_diag=0.10,
    friend_jump=0.65, enemy_jump=0.15,
    friend_knight=0.25, enemy_knight=0.05,
    seed_shared_liberty=1.0, positional_scale=2.0, low_liberty=1.5, weak_friend=1.0, weak_enemy=2.0, decisiveness=1.6,
)
P.update(overrides)

W = [np.zeros((243 if k == 0 else 512, 512), np.float32)
     for k in range(6)]
B = [np.zeros(512, np.float32) for _ in range(6)]
OW = np.zeros((512, 82), np.float32)
OB = np.zeros(82, np.float32)

def unit(ch, p):
    return ch * 81 + p

def points(p, offsets):
    r, c = divmod(p, 9)
    return [(r+dr)*9+c+dc for dr,dc in offsets
            if 0 <= r+dr < 9 and 0 <= c+dc < 9]

AD = [(1,0),(-1,0),(0,1),(0,-1)]
DI = [(1,1),(1,-1),(-1,1),(-1,-1)]
JU = [(2,0),(-2,0),(0,2),(0,-2)]
KN = [(a,b) for a,b in
      [(1,2),(1,-2),(-1,2),(-1,-2),(2,1),(2,-1),(-2,1),(-2,-1)]]
adj = [points(p, AD) for p in range(81)]
diag = [points(p, DI) for p in range(81)]

def add(k, dest, src, value):
    W[k][src, dest] += value

def carry(k, ch, p):
    add(k, unit(ch,p), unit(ch,p), 1)

# First layer:
# own, enemy, and two ReLUs per colour encoding 0/1 local liberties.
# D-Z is exactly 1 for a stone with at most one local liberty.
for p in range(81):
    add(0, unit(0,p), 3*p, 1)
    add(0, unit(1,p), 3*p+1, 1)
    for color, dch, zch in [(0,2,3),(1,4,5)]:
        for ch, coefficient in [(dch,2),(zch,1)]:
            dest = unit(ch,p)
            add(0, dest, 3*p+color, coefficient)
            B[0][dest] = -len(adj[p])
            for q in adj[p]:
                add(0, dest, 3*q, 1)
                add(0, dest, 3*q+1, 1)


# Conservative safety seeds: never mark a genuinely atari group safe.
# Channels 2/4 are binary danger; 3/5 detect adjacent liberty-bearing
# stones, whose liberties must be distinct on the square grid.
for p in range(81):
    carry(1,0,p)
    carry(1,1,p)
    for color,dch,zch in [(0,2,3),(1,4,5)]:
        dest=unit(dch,p)
        add(1,dest,unit(dch,p),1)
        add(1,dest,unit(zch,p),-1)
        for q in adj[p]:
            add(1,dest,unit(color,q),-1)
            add(1,dest,unit(dch,q),1)
            add(1,dest,unit(zch,q),-1)
        dest=unit(zch,p)
        B[1][dest]=-4
        add(1,dest,unit(dch,p),4)
        add(1,dest,unit(zch,p),-8)
        for q in adj[p]:
            add(1,dest,unit(dch,q),1)
            add(1,dest,unit(zch,q),-2)

# Safety propagates through each connected string.
for p in range(81):
    carry(2,0,p)
    carry(2,1,p)
    for color,dch,zch,outch in [(0,2,3,2),(1,4,5,3)]:
        dest=unit(outch,p)
        add(2,dest,unit(dch,p),1)
        add(2,dest,unit(zch,p),-1)
        for q in adj[p]:
            add(2,dest,unit(color,q),-1)
            add(2,dest,unit(dch,q),1)
            add(2,dest,unit(zch,q),-1)

    # Local liberty pressure, reduced for already connected stones.
    for color,ch in [(0,4),(1,5)]:
        dest=unit(ch,p)
        add(2,dest,unit(color,p),3)
        B[2][dest]=-len(adj[p])
        for q in adj[p]:
            add(2,dest,unit(0,q),1)
            add(2,dest,unit(1,q),1)
            add(2,dest,unit(color,q),-0.5)

# Fourth layer: more group-safety propagation and true-eye recognition.
for p in range(81):
    for ch in (0,1):
        carry(3,ch,p)
    for color,dch in [(0,2),(1,3)]:
        dest = unit(dch,p)
        add(3,dest,unit(dch,p),1)
        for q in adj[p]:
            add(3,dest,unit(color,q),-1)
            add(3,dest,unit(dch,q),1)

    # Urgency to support pressured stones and reduce enemy liberties.
    for q in adj[p]:
        add(3,unit(4,p),unit(4,q),P["weak_friend"])
        add(3,unit(4,p),unit(5,q),P["weak_enemy"])

    # Binary access to a friendly string with spare liberties.
    # Count the string once even if it touches a move on several sides.
    dest=unit(5,p)
    B[3][dest]=1
    for q in adj[p]:
        add(3,dest,unit(0,q),-1)
        add(3,dest,unit(2,q),1)

# Fifth layer: final safety propagation, preserving eye features.
for p in range(81):
    for ch in (0,1,4,5):
        carry(4,ch,p)
    for color,dch in [(0,2),(1,3)]:
        dest = unit(dch,p)
        add(4,dest,unit(dch,p),1)
        for q in adj[p]:
            add(4,dest,unit(color,q),-1)
            add(4,dest,unit(dch,q),1)

# Sixth layer computes action features.
for p in range(81):
    # Saturated tactical bonuses avoid counting the same string repeatedly.
    B[5][unit(0,p)]=1
    B[5][unit(1,p)]=1
    for q in adj[p]:
        add(5,unit(0,p),unit(3,q),-1)
        add(5,unit(1,p),unit(2,q),-1)


    # Conservative self-atari prevention with saturated friendly access.
    dest=unit(2,p)
    B[5][dest]=0.5-len(adj[p])
    add(5,dest,unit(5,p),1.5)
    for q in adj[p]:
        add(5,dest,unit(0,q),1)
        add(5,dest,unit(1,q),1)
        add(5,dest,unit(3,q),-2)

    dest=unit(3,p)
    B[5][dest]=1-len(adj[p])
    for q in adj[p]:
        add(5,dest,unit(0,q),1)
    for q in diag[p]:
        add(5,dest,unit(1,q),-0.5 if len(adj[p])==4 else -1)


    # Discourage dense, redundant own-stone shapes.
    dest = unit(4,p)
    B[5][dest] = -2
    for q in adj[p]:
        add(5,dest,unit(0,q),1)
    for q in adj[p]:
        add(5,dest,unit(1,q),-0.75)
    for q in diag[p]:
        add(5,dest,unit(0,q),0.5)

    # A positive offset permits a signed linear positional feature.
    dest = unit(5,p)
    B[5][dest] = 20
    for offsets, ownkey, enemykey in [
        (AD,"friend_adj","enemy_adj"),
        (DI,"friend_diag","enemy_diag"),
        (JU,"friend_jump","enemy_jump"),
        (KN,"friend_knight","enemy_knight"),
    ]:
        for q in points(p,offsets):
            add(5,dest,unit(0,q),P[ownkey])
            add(5,dest,unit(1,q),P[enemykey])

    # Extend from healthy strings rather than from stones in trouble.
    for offsets, strength in [(DI,0.3),(JU,0.5),(KN,0.2)]:
        for q in points(p,offsets):
            add(5,dest,unit(0,q),strength)
            add(5,dest,unit(2,q),-strength)
    add(5,dest,unit(5,p),-P["low_liberty"])
    add(5,dest,unit(4,p),1)

    for ch,coef in enumerate([
        -P["capture"], -P["rescue"], -P["self_atari"],
        -P["eye"], -P["crowd"], 1.0
    ]):
        OW[unit(ch,p),p] = coef

    r,c = divmod(p,9)
    edge = min(r,c,8-r,8-c)
    positional = [-0.9,0.1,0.75,0.65,0.65][edge]
    if (r,c) in [(2,2),(2,6),(6,2),(6,6),(4,4)]:
        positional += 0.35
    OB[p] = -20 + P["capture"] + P["rescue"] + P["positional_scale"]*positional


# Tengen is a useful universal first move on the small board.
# This gate is active only on the completely empty board.
B[0][486]=1
for p in range(81):
    W[0][3*p,486]=-1
    W[0][3*p+1,486]=-1
for k in range(1,6):
    W[k][486,486]=1
OW[486,40]=8

OB[81] = P["pass_bias"]

OW *= P["decisiveness"]
OB *= P["decisiveness"]

for k in range(6):
    save("L%d.W" % (k+1), W[k])
    save("L%d.b" % (k+1), B[k])
save("out.W", OW)
save("out.b", OB)
