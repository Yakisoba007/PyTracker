import numpy as np

class MotionFilter3D(object):
    '''
    class to model motion of a joint
    
    - using angles alpha and beta to describe motion
    - linear change of these angles assumed
    '''
    
    
    def __init__(self, initState, initAnchor, length, length2=None):
        self.vMax = np.asarray([80.0, 80.0])*np.pi/180.0
        vec = np.asarray(initState) - np.asarray(initAnchor)
        self.state  = np.asarray(self.getAngles(vec))
        self.v = np.array([0,0])
        self.r = length
    
    def update(self, point, anchor):
        ''' update current state (alpha, beta, v) by new observation 'point' '''
        nowVec = np.array(point).reshape(-1)-np.array(anchor).reshape(-1)
        nowState = np.asarray(self.getAngles(nowVec))
        
        nowV = nowState-self.state
        for i in range(len(nowV)):
            if nowV[i] > np.pi:
                nowV[i] -= 2*np.pi
            elif nowV[i] < -np.pi:
                nowV[i] += 2*np.pi
        
        for i in range(len(nowV)):
            if np.abs(nowV[i]) >= self.vMax[i]:
                print "too fast", i
                print "before", nowV[i]*180.0/np.pi
                nowV[i] = self.vMax[i] * np.abs(nowV[i])/nowV[i]
                print "after", nowV[i]*180.0/np.pi
        self.v = (nowV+ self.v)/2.0
        self.state = self.state + self.v
        
        for i in range(len(self.state)):
            if self.state[i] > np.pi:
                self.state[i] -= 2*np.pi
            elif self.state[i] < -np.pi:
                self.state[i] += 2*np.pi
        
        ret =  np.asarray( [self.r*np.cos(self.state[0])*np.cos(self.state[1]),
                            self.r*np.sin(self.state[0])*np.cos(self.state[1]),
                            self.r*np.sin(self.state[1])] )
        return anchor + ret
    
    def predict(self, anchor):
        ''' with the help of current state v, the position of next frame joint
        is returned '''
        
        state = self.state + self.v
        ret =  np.asarray( [self.r*np.cos(state[0])*np.cos(state[1]),
                self.r*np.sin(state[0])*np.cos(state[1]),
                self.r*np.sin(state[1])] )
        return anchor + ret
    
    def project(self, point, anchor):
        ''' project point to new position if anchor
        has changed previously
        '''
        vec = point-anchor
        r = np.linalg.norm(vec)
        a = np.asarray(self.getAngles(vec)) +self.v
        ret =  np.asarray( [r*np.cos(a[0])*np.cos(a[1]),
                            r*np.sin(a[0])*np.cos(a[1]),
                            r*np.sin(a[1])] )
        return anchor + ret
    
    def getAngles(self, vec):
        ''' calculate angles of vec (alpha, beta) '''
        alpha = np.arctan2(vec[1], vec[0])
        norm = np.linalg.norm(vec[:2])
        beta = np.arctan2(vec[2], 42) if norm==0 else np.arctan2(vec[2], norm) 
        return alpha, beta