import numpy as np

class StateHistory(object):
    '''
    container to hold history states and be able to dump them
    '''


    def __init__(self):
        joints = ['elbowLeft', 'elbowRight',
                  'wristLeft', 'wristRight']
        self.states = dict((el,[]) for el in joints)
        
    def dump(self):
        keys = self.states.keys()
        l = len(self.states[keys[0]])
        line = ""
        for i in range(l):
            if self.states[keys[0]][i] is None:
                break
            for x in keys:
                st = self.states[x]
                line += " ".join(np.char.mod("%f", st[i].pt))+" "
                line += " ".join(np.char.mod("%f", st[i].angle))+" "
                line += " ".join(np.char.mod("%f", st[i].v)) + " "
            line += "\n"
        return line
        
class State(object):
    ''' 
    container to hold states:
    pt: (x,y,z)
    angle: (alpha, beta)
    v
    '''
    
    
    def __init__(self, pt, angle, v):
        self.pt = pt
        self.angle = angle
        self.v = v
    
    def __repr__(self, *args, **kwargs):
        return str(self.pt) + " " + str(self.angle) + " " + str(self.v)