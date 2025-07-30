import numpy as np

def kelly(p, b=1):
    # p = win probability, b = odds (assume 1:1)
    return max(0, p - (1-p)/b)

def position_size(equity, p):
    f = kelly(p)
    return equity * f
