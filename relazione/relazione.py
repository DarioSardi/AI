def function(x):
    m=-1/3
    c=0.5
    return m*x+c

def genFunction(x,y):
    if y>function(x): return 1
    else: return -1


class point:
    def __init__(self,x,y,b):
        self.pos=[x,y,b]
