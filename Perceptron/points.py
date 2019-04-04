import numpy as np
class point:
    def __init__(self,x,y,b):
        self.pos=[x,y,b]
        if self.pos[1]>self.pos[0]: self.group=1
        else: self.group=-1

    

