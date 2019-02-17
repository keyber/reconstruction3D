
def chamfer(F, x, P, S):
    """return chamfer loss between
    the generated set G = {f(x, p)  for each f, x, p}
    and the real set S"""
    
    loss = 0
    #pour chaque p et f la distance au s le plus proche
    #les MLP doivent bien atteindre un Q
    for p in P:
        for f in F:
            loss += min( (f(x,p) - s) ** 2 for s in S)
    
    #pour chaque s la distance au p, MLP le plus proche
    #les Q doivent bien être atteint par un MLP
    for s in S:
        loss += min(min( (f(x,p) - s) ** 2
                for p in P) for f in F)
    
