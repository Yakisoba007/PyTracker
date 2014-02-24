import registration
from process.history import State
import numpy as np

class Adjustment(object):
    '''
    class in order to change calculated joint positions
    
    position is not arbitrary: with the help of hovered depth value,
    a 3D position is obtained. this vector's length is set to lower resp.
    upper arm length. following joints of the kinematic chains are adjusted
    accordingly (their state to the actually adjusted joint is preserved)
    '''


    def __init__(self, frame, skeleton, joint):
        # structure of skeleton
        self.kinematics = ['shoulder', 'elbow', 'wrist']
        self.frame = frame
        self.skeleton = skeleton
        
        side = ""
        idx = 0
        if joint.find('Left') >= 0:
            side = 'Left'
            idx = joint.find('Left')
        else:
            side = 'Right'
            idx = joint.find('Right')
            
        self.side = side
        # for what to adjust
        self.joint = self.kinematics.index(joint[:idx])
        
        self.valid = True
        
    def updatePosition(self, x, y, z):
        ''' calculate position accoring to mouse position and depth value (x,y,z)
        '''
        mfKey = self.kinematics[self.joint-1]+self.side + "-" + self.kinematics[self.joint]+self.side
        fixPt = self.skeleton.states[self.kinematics[self.joint-1]+self.side]
        fixPt = registration.point2world(fixPt)
        
        pt = registration.point2world((x,y,z))
        pt = self.skeleton.mf[mfKey].update(pt, fixPt)
        x,y = registration.world2point(pt)
        z = pt[2]
        
        self.skeleton.states[self.kinematics[self.joint]+self.side] = (x,y,z)
        self.skeleton.fullStates[self.kinematics[self.joint]+self.side] = State(pt, 
                                                                                self.skeleton.mf[mfKey].state,
                                                                                self.skeleton.mf[mfKey].v)
        # update consecutive joints        
        if self.joint == 1:
            depPt = self.skeleton.states[self.kinematics[self.joint+1]+self.side]
            depPt = registration.point2world(depPt)
            depPt = self.skeleton.mf[mfKey].project(np.asarray(depPt), np.asarray(fixPt))
#            mfKeyDep = self.kinematics[self.joint]+self.side + "-" + self.kinematics[self.joint+1]+self.side
#            depPt = self.skeleton.mf[mfKeyDep].update(depPt, pt)
            dx,dy = registration.world2point(depPt)
            dz = depPt[2]
            
            self.skeleton.states[self.kinematics[self.joint+1]+self.side] = (dx,dy,dz)
            self.skeleton.fullStates[self.kinematics[self.joint+1]+self.side].pt = depPt
        