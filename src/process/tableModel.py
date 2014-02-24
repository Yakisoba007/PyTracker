import numpy as np
import cv2

class Table(object):
    '''
    class to model table patients sitting in front of
    
    - 3 points on the table are necessary
    - table assumed to be planar
    - to filter table, modeled table is subtracted from fram 
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.initState = 3
        self.depth = None
        self.table = None
    
    def setPoint(self, pt):
        ''' interface function: set a point for class (x,y,z)
        
        set three points of table: front right, depth right, depth left
        '''        
        if self.initState ==3:
            self.frontRight = np.asarray([pt[1], pt[0], pt[2]])
        elif self.initState ==2:
            self.depthRight = np.asarray([pt[1], pt[0], pt[2]])
        elif self.initState == 1:
            self.depthLeft = np.asarray([pt[1], pt[0], pt[2]])
        
            self.depthVec = self.depthRight - self.frontRight
            self.widthVec = self.depthLeft - self.depthRight
            
            self.table = None
            self.calcTable()
        self.initState -= 1
    
    def view(self, event, y, x, flag, param):
        ''' helper function '''
        if event !=  cv2.EVENT_LBUTTONDOWN:
            return
        print "y: ", y
        print "x: ", x
        print "table: ", self.table[x, y]
        
    def calcTable(self):
        ''' model table according to the three sampled points
        
        explanation see thesis
        '''
        self.table = np.zeros((480,640))
        mask= np.zeros((480, 640), dtype=np.int8)
        pts = np.asarray([self.frontRight[:2][::-1], 
                          self.depthRight[:2][::-1],
                          self.depthLeft[:2][::-1], 
                          self.frontRight[:2][::-1]+self.widthVec[:2][::-1]
                        ])
        cv2.fillConvexPoly(mask, pts.astype(int), 1)
        
        z = np.asarray(np.unravel_index(np.arange(640*480), (480, 640)))
        z = z - np.tile(self.frontRight[:2], (z.shape[1], 1)).T
        
        A = np.asarray([self.depthVec[:2]/np.linalg.norm(self.depthVec[:2]),
                        self.widthVec[:2]/np.linalg.norm(self.widthVec[:2])]).T
        self.table = np.dot(np.linalg.inv(A), z)
        self.table = self.table*np.tile([self.depthVec[2]/np.linalg.norm(self.depthVec[:2]), 
                                         self.widthVec[2]/np.linalg.norm(self.widthVec[:2])],
                                        (self.table.shape[1], 1)).T
        self.table = (np.sum(self.table, axis=0).reshape((480,640)) + self.frontRight[2] )*mask
    
    def filter(self):
        ''' substract table from frame with a variance of +8 and -3cm 
        
        return: frame with filtered table
        '''
        newDepth = np.zeros_like(self.depth)
        newDepth[self.table - self.depth > 80] = self.depth[self.table - self.depth > 80]
        newDepth[self.table - self.depth < -30] = self.depth[self.table - self.depth < -30]
        return newDepth