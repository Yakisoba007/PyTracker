import numpy as np
import registration
from motion import MotionFilter3D
from process.history import State

class Skeleton(object):
    '''
    class to find joint position of human skeleton
    
    - an initial skeleton has to be given
    - in order to fit a new position, frame has to be transformed to a binary
      distance transformation (skeleton of frame)
    - occlusions only partly supported
    '''
    
    
    def __init__(self):
        self.joints = ['head', 'shoulderLeft', 'shoulderRight', 
                       'elbowLeft', 'elbowRight',
                       'wristLeft', 'wristRight'][::-1]
        self.states = dict((el,(0,0,0)) for el in self.joints)
        
        self.initState = len(self.states)-1 #countdown for each part until zero
        self.fullStates = {}
        self.depth = None
        
        self.lengthUp = 0
        self.lengthBot = 0
        self.mf = {}
    
    def setPoint(self, pt):
        ''' interface function: set a point for class (x,y,z)
        
        set initial positions for every joint, left and right,
        occlusions are not supported
        '''
        self.states[self.joints[self.initState]] = pt
        
        if self.initState == 0:
            lengthUp = [0]*2
            lengthBot = [0]*2
            for i, side in enumerate(['Left', 'Right']):
                lengthUp[i] = np.sqrt(
                        np.sum( (
                            np.asarray(registration.point2world(self.states['shoulder'+side]), dtype=float)
                          - np.asarray(registration.point2world(self.states['elbow'+side]), dtype=float) 
                        )**2 ) )
                lengthBot[i] = np.sqrt(
                        np.sum( (
                            np.asarray(registration.point2world(self.states['elbow'+side]), dtype=float)
                          - np.asarray(registration.point2world(self.states['wrist'+side]), dtype=float) 
                        )**2 ) )
            self.lengthBot = np.mean(lengthBot)
            self.lengthUp = np.mean(lengthUp)
            
            for side in ['Left', 'Right']:
                self.mf['shoulder'+side+"-elbow"+side] = MotionFilter3D(registration.point2world(self.states['elbow'+side]),
                                                                        registration.point2world(self.states['shoulder'+side]), 
                                                                        self.lengthUp, self.lengthBot)
                self.mf['elbow'+side+"-wrist"+side] = MotionFilter3D(registration.point2world(self.states['wrist'+side]),
                                                                     registration.point2world(self.states['elbow'+side]),
                                                                     self.lengthBot)
                self.fullStates['elbow'+side] = State(registration.point2world(self.states['elbow'+side]),
                                      self.mf['shoulder'+side+'-elbow'+side].state,
                                      self.mf['shoulder'+side+'-elbow'+side].v)
                self.fullStates['wrist'+side] = State(registration.point2world(self.states['wrist'+side]),
                                      self.mf['elbow'+side+'-wrist'+side].state,
                                      self.mf['elbow'+side+'-wrist'+side].v)
                
            
        self.initState -= 1
    
    def setState(self, joint, state):
        ''' set state (x,y,z) and fullstate (State) '''
        self.fullStates[joint] = state
        pt = registration.world2point(state.pt)
        self.states[joint] = (pt[0], pt[1], state.pt[2])
        idx = [x[-len(joint):] for x in self.mf.keys()].index(joint)
        mfKey = self.mf.keys()[idx]
        self.mf[mfKey].state = state.angle
        self.mf[mfKey].v = state.v
        
    def fit(self, skeleton):
        ''' with the help of adjusted frame 'skeleton' new joint position
        are obtained
        '''
        self.fitPart(skeleton, 'shoulderLeft', 'elbowLeft', 'wristLeft')
        self.fitPart(skeleton, 'elbowLeft', 'wristLeft')
        self.fitPart(skeleton, 'shoulderRight', 'elbowRight', 'wristRight')
        self.fitPart(skeleton, 'elbowRight', 'wristRight')
        
#        self.fitPart('elbowLeft', 'wristLeft', skeleton)
        
    def fitPart(self, skeleton, fixPart, loosePart, nextPart = None):
        ''' fit one point, if this has successor, it is also adjusted '''
        localSkeleton = np.copy(skeleton)
        ###### predeccesor ######
        fix = self.states[fixPart]
        fixWorld  = np.asarray(registration.point2world(fix))
        ###### successor ######
        nextPoint = None
        if nextPart is not None:
            nextPoint = registration.point2world(self.states[nextPart])
            
        ###### interpolated new point ######
        futurePoint = self.mf[fixPart+"-"+loosePart].predict(fixWorld)
        ###### observed new point ######
        nowPoint = registration.point2world(self.states[loosePart])
        
        localSkeleton[np.abs(self.depth-nowPoint[2])>100] = 0 # limit search space: only points +- 10cm
        world = registration.get3Dworld(self.depth)[0]
        world = world.T.reshape((480,640,3))
        
        ###### window parameters ######
        # new point has to be in a 
        # window (+-10px) as interpolated
        # new point
        x = int(np.round( futurePoint[0]*registration.depthFX/futurePoint[2] + registration.depthCX))
        y = int(np.round( futurePoint[1]*registration.depthFY/futurePoint[2] + registration.depthCY))
        z = futurePoint[2]
        x1=np.max((int(np.round(x)), fix[0])) + 10
        x2=np.min((int(np.round(x)), fix[0])) - 10
        y1=np.max((int(np.round(y)), fix[1])) + 10
        y2=np.min((int(np.round(y)), fix[1])) - 10
        
        ###### find best candidate ######
        # by using fitting linear function
        # and minimizing the distance
        candidates = world[y-10:y+10, x-10:x+10][np.where(localSkeleton[y-10:y+10, x-10:x+10])]
        if len(candidates) != 0:
            what2Fit= world[y2:y1, x2:x1][np.where(localSkeleton[y2:y1, x2:x1])]
            fitFaktor = np.zeros(candidates.shape[0])
            for i, c in enumerate(candidates):
                u = c - fixWorld
                d = np.cross((what2Fit - fixWorld), u)
                d = np.sqrt(np.sum(d**2, axis=1))/np.linalg.norm(u)
                fitFaktor[i] = np.sqrt(np.sum(d**2))
            
            bestFit = candidates[np.argmin(fitFaktor)]
            bestFit = self.mf[fixPart+"-"+loosePart].update(bestFit, fixWorld)
            x,y = registration.world2point(bestFit)
            z = bestFit[2]
        
        ###### adjust successor point ######
        if nextPoint is not None:
            nextPoint = self.mf[fixPart+"-"+loosePart].project(nextPoint, fixWorld)
            a, b = registration.world2point(nextPoint)
            self.states[nextPart] = (a, b, nextPoint[2])
            
        self.states[loosePart] = (x, y, z)
        self.fullStates[loosePart] = State(registration.point2world(self.states[loosePart]),
                                           self.mf[fixPart+"-"+loosePart].state,
                                           self.mf[fixPart+"-"+loosePart].v)
              
    def reset(self):
        ''' reset states '''
        self.initState = len(self.states)-1