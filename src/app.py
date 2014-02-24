import primesense.openni2 as oni
import cv2
import cv2.cv as cv
import numpy as np
from wrapPare.prepare import Prepare
from process.history import StateHistory
from wrapPare.postpare import saveHistory
from process.adjust import Adjustment

class PyTrackerApp(object):
    '''
    class containing main loop
    
    application divided into preparation, main loop, post things
    
    PyTracker tracks joints of a human (occlusions partly supported), saves
    the history to be able to adjust every frame afterwards (consecutive frames
    has to be recalculated) and then saves them in a file
    '''

    def __init__(self, filename):
        oni.initialize()
        self.filename = filename
        self.dev = oni.Device.open_file(filename)
        
        self.player = oni.PlaybackSupport(self.dev)
        self.player.set_speed(-1)
        self.depth_stream = self.dev.create_depth_stream()
        self.clr_stream = self.dev.create_color_stream()
        self.depth_stream.start()
        self.clr_stream.start()
        
        self.prep = Prepare(self)
        self.skeleton = None
        
        self.pause = 5
        self.history = StateHistory()
        self.currentFrameIdx = 0
        self.drag = False
        
        self.adjustment = None
        for x in self.history.states.keys():
            self.history.states[x] = [None]*self.player.get_number_of_frames(self.depth_stream)
        
    def prepare(self):
        ''' do preparations '''
        self.prep.thresholding()
        self.prep.tabling()
        self.skeleton = self.prep.posture()
        
    def mainLoop(self):
        ''' main loop consisting of distance transformation of frame and then fitting the
        skeleton, finally saving the state into history, a slider bar allows to navigate
        through already calculated frames and adjust joints 
        '''
        cannyTop = 30;
        cannyBot = 15
        key = -1
        
        cv2.namedWindow("main", 1)
        cv2.createTrackbar("Frame id", "main", self.currentFrameIdx,
                           self.player.get_number_of_frames(self.depth_stream),
                           self.onTrackbar)
        cv2.setMouseCallback("main", self.onClick)
        
        while True:
            incr = self.currentFrameIdx
            frame = self.getFrame(incr)
            
            if incr < self.player.get_number_of_frames(self.depth_stream):
                keys = self.skeleton.fullStates.keys()
                ###############################
                # check history if this frame #
                # is already calculated       #
                ###############################
                if self.history.states[keys[0]][incr] is not None:
                    for x in keys:
                        self.skeleton.setState(x, self.history.states[x][incr])
                else:
                    for x in keys:
                        self.history.states[x][incr] = self.skeleton.fullStates[x]
        
                    ###############################
                    # global adjustment to obtain #
                    # skeleton like frame         #
                    ###############################
                    depth = self.prep.apply()
                    masked = np.copy(depth)
                    masked = masked/(np.max(masked)-np.min(masked))-np.min(masked)
                    masked *= 255.
                    masked = np.rint(masked).astype(np.uint8)
#                    outA = masked
                    
                    depthEdge = cv2.Canny(masked, cannyBot, cannyTop)
                    depthEdge[self.prep.mask==0] = 255
#                    outB = depthEdge
                    
                    dist = cv2.distanceTransform(~depthEdge, cv.CV_DIST_L2, 3)
                    dist = self.normFrame(dist)
#                    outC = dist
                    skeleton = cv2.adaptiveThreshold(dist, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 0)
#                    outD = skeleton
                    
                    self.skeleton.fit(skeleton)
                    
            ######################
            # some nice outputs  #
            ######################
            out2 = cv2.cvtColor(self.normFrame(depthEdge), cv2.COLOR_GRAY2BGR)    
            out = cv2.cvtColor(self.normFrame(frame), cv2.COLOR_GRAY2BGR)
            cv2.putText(out, str(self.currentFrameIdx-1), (600, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0))
            for i in self.skeleton.joints:
                cv2.circle(out2, self.skeleton.states[i][:2], 5, (255,255,0), -1)
                cv2.circle(out, self.skeleton.states[i][:2], 5, (255,255,0), -1)
                
#            cv2.imshow("A", outA)
#            cv2.imshow("B", outB)
#            cv2.imshow("C", outC)
#            cv2.imshow("D", outD)

            cv2.imshow("main", out)
            cv2.imshow("skeleton", out2)

            key = cv.WaitKey(self.pause)
            if incr < self.player.get_number_of_frames(self.depth_stream):
                self.currentFrameIdx += 1 # only increase index, if further frames can be read
            if key == 113: # q
                break
            if key == 112: # p
                self.pause = 5 if self.pause == 0 else 0
        saveHistory(self.filename + ".label", self.history)
                
        
    def getFrame(self, frameIdx=-1):
        ''' read a frame at index frameIdx '''
        if frameIdx >= 0:
            self.player.seek(self.depth_stream, frameIdx)
        depth = self.depth_stream.read_frame()
        depth= np.ctypeslib.as_array(depth.get_buffer_as_uint16())
        depth = depth.reshape((480,640))
        depth = np.float32(depth)
        self.prep.table.depth = depth
        self.prep.skeleton.depth = depth
        return depth

    def onTrackbar(self, param):
        ''' pause tracking and set currentFrameIdx to slider bar index'''
        self.pause = 0
        currentFrameIdx = cv2.getTrackbarPos("Frame id", "main")
        key = self.history.states.keys()[0]
        idx =  next(i for i, j in enumerate(self.history.states[key]) if j is None)
        if currentFrameIdx > idx:
            currentFrameIdx = idx
        self.currentFrameIdx = currentFrameIdx
        
    def onClick(self, event, x, y, flag, param):
        ''' in pause modus be able to adjust joints manually with left click
        l
        '''
        
        if self.pause != 0:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            keys = ['wristRight', 'wristLeft', 'elbowRight', 'elbowLeft']
            for key in keys:
                ####################
                # determine joint  #
                ####################
                sx = self.skeleton.states[key][0]
                sy = self.skeleton.states[key][1]
                
                if np.abs(sx-x)<= 3 and np.abs(sy-y) <= 3:
                    ###########################
                    # load state from history #
                    # all consecutive states  #
                    # are deleted             #
                    ###########################
                    for k in keys:
                        l = len(self.history.states[key])
                        self.history.states[k][self.currentFrameIdx+1:] = [None]*(l-self.currentFrameIdx)
                        self.skeleton.setState(k, self.history.states[k][self.currentFrameIdx])
                    ###########################################
                    # state for adjusted joint is set by user #
                    # therefore, load previous state          #
                    ###########################################
                    self.skeleton.setState(key, self.history.states[k][self.currentFrameIdx-1])
                    self.adjustment = Adjustment(self.getFrame(self.currentFrameIdx), self.skeleton, key)
                    self.drag = True
                    break
        elif event == cv2.EVENT_MOUSEMOVE and self.drag:
            #############################
            # update joint position     #
            # as mouse position changes #
            #############################
            frame = self.getFrame(self.currentFrameIdx)
            self.adjustment.updatePosition(x, y, frame[y,x])
            out = cv2.cvtColor(self.normFrame(frame), cv2.COLOR_GRAY2BGR)
            cv2.putText(out, str(self.currentFrameIdx-1), (600, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0))
            for i in self.skeleton.joints:
                cv2.circle(out, self.skeleton.states[i][:2], 5, (255,255,0), -1)
            cv2.imshow("main", out)
        elif event == cv2.EVENT_LBUTTONUP and self.drag:
            ###############################
            # set joint to mouse position #
            ###############################
            print "lost at", x, y
            keys = self.skeleton.fullStates.keys()
            for key in keys:
                self.history.states[key][self.currentFrameIdx] = self.skeleton.fullStates[key]
            self.drag = False
             
    def normFrame(self, frame):
        return np.rint((1.0*frame)/np.max(frame)*255.0).astype(np.uint8)