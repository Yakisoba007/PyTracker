import numpy as np
import cv2

#################################
# camera calibration parameters #
# of microsoft kinect           #
#################################
depthCX = 319.554313;
depthCY = 237.375891;
depthFX = 544.166699;
depthFY = 550.666193;
depthIntrinsic = np.array([[depthFX, 0, depthCX],
        [0, depthFY, depthCY],
        [0, 0, 1]])

depthDistortion = np.array([8.279470e-03, -2.103757e-01, -6.704529e-04, 0.000000e+00, 3.943653e-01])

rgbCX = 317.909506;
rgbCY = 246.662717;
rgbFX = 490.577541;
rgbFY = 497.235159;
rgbIntrinsic = np.array([[rgbFX, 0, rgbCX],
        [0, rgbFY, rgbCY],
        [0, 0, 1]])
rgbDistortion = np.array([1.284672e-01, -1.447001e-01, -1.255696e-04, 0.000000e+00, -2.283035e-01]);

extTranslation = np.array([2.744582e-02, -1.935876e-03,3.180614e-03]);
extRotation = np.array([[9.9998677518893e-01, -3.7700884771569e-03, -3.4979822889508e-03],
        [3.7402011469390e-03, 9.9995677981541e-01, -8.5117211272654e-03],
        [3.5299210472534e-03, 8.4985254039926e-03, 9.9995765646519e-01]] );

def world2point(coord):
    ''' transforms a 3d-point to x-y coordinates
    of depth image
    '''
    x = int(np.round( coord[0]*depthFX/coord[2] + depthCX))
    y = int(np.round( coord[1]*depthFY/coord[2] + depthCY))
    return x,y

def point2world(coord):
    ''' transforms a x-y coorfinate of a depth
    image to 3d-point 
    '''
    x = coord[0]
    y = coord[1]
    z = coord[2]
    x = (x - depthCX)*z/depthFX
    y = (y - depthCY)*z/depthFY
    return (x,y, z)

def get3Dworld(depth):
    ''' transforms whole depth frame to
    a point cloud in 3d space
    
    aggregate numpy array operations for efficiency
    '''
    gridX = np.arange(depth.shape[1], dtype=float)
    gridX = np.tile(gridX, (depth.shape[0], 1))

    gridY = np.arange(depth.shape[0], dtype=float)
    gridY = np.tile(gridY, (depth.shape[1], 1)).T
    
    worldX= ((gridX-depthCX)* depth/depthFX).flatten()
    worldY = ((gridY-depthCY)* depth/depthFY).flatten()
    
    world = np.vstack((worldX, worldY, depth.flatten()))
    return world, gridX, gridY

def register(rgb, depth):
    ''' register rgb and depth frames with each other:
    the coordinates of a point in the depth frame corrsponds
    to the point in rgb frame
    '''
    out= np.zeros_like(rgb)
    rgb = cv2.undistort(rgb, rgbIntrinsic, rgbDistortion)
    depth = cv2.undistort(depth, depthIntrinsic, depthDistortion)
    
    world, gridX, gridY = get3Dworld(depth)
    p3d = np.dot(extRotation.T, world) + np.tile(extTranslation, (world.shape[1], 1)).T
    
    rgbX = np.rint( (p3d[0]*rgbFX/ p3d[2]) + rgbCX).reshape(480,640)
    rgbY = np.rint( (p3d[1]*rgbFY/ p3d[2]) + rgbCY).reshape(480,640)

    rgbX[depth>=4095] = 0
    rgbX[depth<=0] = 0
    
    
    out[gridY.astype(int), gridX.astype(int)] = rgb[rgbY.astype(int), rgbX.astype(int)]
    return out
    