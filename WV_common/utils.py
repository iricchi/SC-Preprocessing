import numpy as np


def flatten_list(l): 
    if type(l[0]) == list: 
    	l2 = [item for sublist in l for item in sublist]  
    else: 
    	l2 = l 
    return l2