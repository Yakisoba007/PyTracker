import cv2
import numpy as np
from process.tableModel import Table
from process.joint import Skeleton

class Prepare(object):
    '''
    class organizing the preparation process:
    - table model
    - initial skeleton joints
    - thresholding values
    '''
    
    
    def __init__(self, app):
        self.app = app
        self.table = Table()
        self.skeleton = Skeleton()
        self.threshold = None
        self.showImage = None
        self.mask = None
        
    def setApoint(self, event, x, y, flag, param):
        ''' interface for function setPoint '''
        if event !=  cv2.EVENT_LBUTTONDOWN:
            return
        pt = (x, y, self.frame[y, x])
        param.setPoint(pt)
        cv2.circle(self.showImage, (x,y), 5, (255,255,0))
        
    def tabling(self):
        ''' window to choose points on the table '''
        frame = self.app.getFrame(0)
        self.frame = frame
        self.showImage = cv2.cvtColor(self.app.normFrame(frame), cv2.COLOR_GRAY2BGR)
        cv2.namedWindow("click 3 points on the table")
        cv2.setMouseCallback("click 3 points on the table", self.setApoint, self.table);
        while self.table.initState > 0:
            cv2.imshow("click 3 points on the table", self.showImage)
            cv2.waitKey(5)
        cv2.destroyWindow("click 3 points on the table")
            
    
    def posture(self):
        ''' window to set intital skeleton joints '''
        frame = self.app.getFrame(0)
        self.showImage = cv2.cvtColor(self.app.normFrame(frame), cv2.COLOR_GRAY2BGR)
        cv2.namedWindow("posture")
        cv2.setMouseCallback("posture", self.setApoint, self.skeleton);#
        while self.skeleton.initState >= 0:
            cv2.imshow("posture", self.showImage)
            cv2.waitKey(5)
        cv2.destroyWindow("posture")
        return self.skeleton
    
    def thresholding(self):
        ''' window with slider bars to set thresholds for
        far and narrow values 
        '''
        threshKLo = 1;
        threshZLo = 200;
        threshK = 2;
        threshZ = 400;
    
        cv2.namedWindow("depth image", 1)
        cv2.createTrackbar("Threshold k", "depth image", threshK, 65, lambda x: None)
        cv2.createTrackbar("Threshold z", "depth image", threshZ, 1000, lambda x: None)
        cv2.createTrackbar("Threshold k Low", "depth image", threshKLo, 65, lambda x: None)
        cv2.createTrackbar("Threshold z Low", "depth image", threshZLo, 1000, lambda x: None)
        incr = 0
        
        cannyTop = 30;
        cannyBot = 15
        while True:
            if incr >= self.app.player.get_number_of_frames(self.app.depth_stream):
                incr = 0
            frame = self.app.getFrame(incr)
            
            threshK= cv2.getTrackbarPos("Threshold k", "depth image")
            threshZ= cv2.getTrackbarPos("Threshold z", "depth image")
            threshKLo= cv2.getTrackbarPos("Threshold k Low", "depth image")
            threshZLo= cv2.getTrackbarPos("Threshold z Low", "depth image")
            threshold = (threshKLo*1000 + threshZLo, threshK*1000 + threshZ)
            
            maskd = self.masking(frame, threshold)
            
            masked = np.copy(maskd*frame)
            masked = masked/(np.max(masked)-np.min(masked))-np.min(masked)
            masked *= 255.
            masked = np.rint(masked).astype(np.uint8)
            canny = cv2.Canny(masked, cannyBot, cannyTop)
            cv2.imshow("depth image", self.app.normFrame(frame*maskd))
            cv2.imshow("canny", canny)
            
            incr += 1
            key = cv2.waitKey(5)
            if key == 113:
                break
        self.threshold = (threshKLo*1000 + threshZLo, threshK*1000 + threshZ)

        cv2.destroyWindow("depth image")
        cv2.destroyWindow("canny")
        
    def masking(self, frame, threshold):
        ''' accepts tuple of thresholds for far and near,
        set frame to zero where values are outside these limits
        (>threshold[0], <threshold[1])
        
        returns mask
        '''
        res = np.zeros_like(frame)
        res2 = np.zeros_like(frame)
        cv2.threshold(frame,threshold[0], 1, cv2.THRESH_BINARY, res)
        cv2.threshold(frame, threshold[1], 1, cv2.THRESH_BINARY_INV, res2)
    
        maskd = cv2.bitwise_and(res, res2)
        el = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7), (3,3))
    
        maskd= cv2.erode(maskd, el)
        maskd = cv2.dilate(maskd, el)
        self.mask = maskd
        return maskd
        
    def apply(self):
        ''' filter table and apply threhsold mask '''
        depth = self.table.filter()
        depth *= self.masking(depth, self.threshold)
        return depth
