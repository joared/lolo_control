import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as R
from lolo_perception.perception_utils import projectPoints, plotPoints

def interactionMatrix(x, y, Z):
    """
    IBVS interaction matrix
    """
    return [[-1/Z, 0, x/Z, x*y, -(1+x*x), y],
                [0, -1/Z, y/Z, 1+y*y, -x*y, -x]]


def calcIBVS(targetTransl, 
             targetRotVec, 
             detectedTransl, 
             detectedRotVec, 
             camera, 
             featureModel,
             controlScheme, 
             drawImg=None):
    """
    Calculates IBVS control velocity based on target pose and detected pose
    controlScheme - IBVS1, IBVS2 or IBVS3 (from chaumette)
    """
    targetFeatures = projectPoints(targetTransl, targetRotVec, camera, featureModel.features)
    detectedFeatures = projectPoints(detectedTransl, detectedRotVec, camera, featureModel.features)

    if drawImg is not None:
        plotPoints(drawImg, targetFeatures, color=(0,255,255), radius=3)
        plotPoints(drawImg, detectedFeatures, color=(0,0,255), radius=3)

    cx = camera.cameraMatrix[0, 2]
    cy = camera.cameraMatrix[1, 2]

    fx = camera.cameraMatrix[0, 0]
    fy = camera.cameraMatrix[1, 1]

    targetFeatures = [((x-cx)/fx, (y-cy)/fy) for x,y in targetFeatures]
    detectedFeatures = [((x-cx)/fx, (y-cy)/fy) for x,y in detectedFeatures]

    zDetected = detectedTransl[2] # TODO: calculated for each detected feature point
    zTarget = targetTransl[2]

    Lx = []
    for target, feat  in zip(targetFeatures, detectedFeatures):
        x = feat[0]
        y = feat[1]
        
        Le = interactionMatrix(x, y, zDetected)

        x = target[0]
        y = target[1]
        
        if drawImg is not None:
            cv.line(drawImg, 
                    (int(round(feat[0]*fx+cx)), int(round(feat[1]*fy+cy))), 
                    (int(round(target[0]*fx+cx)), int(round(target[1]*fy+cy))),
                    thickness=1, 
                    color=(0,0,255))

        LeStar = interactionMatrix(x, y, zTarget)

        LeLeStar = (np.array(Le) + np.array(LeStar))/2

        if controlScheme == "IBVS1":
            # IBVS version 1 in chaumette
            L = Le
        elif controlScheme == "IBVS2":
            # IBVS version 2 in chaumette
            L = LeStar
        elif controlScheme == "IBVS3":
            # IBVS version 3 in chaumette
            L = LeLeStar
        else:
            raise Exception("Invalid controller '{}'".format(controlScheme))

        Lx.append(L)

    Lx = np.row_stack(Lx)
    LxPinv = np.linalg.pinv(Lx)
    err = np.array([v for f in detectedFeatures for v in f]) - np.array([ v for t in targetFeatures for v in t])
    v = -np.matmul(LxPinv, err)
    return v

def calcPBVS(targetTransl, 
             targetRotVec, 
             detectedTransl, 
             detectedRotVec, 
             controlScheme, 
             drawImg=None):
    """
    targetTranlation and targetRotation expressed in feature frame
    """
    #targetRot = R.from_rotvec(targetRotVec)
    #dsToTargetTransl = targetRot.inv().apply(targetTransl) - targetRot.inv().apply(detectedTransl)
    #camToTargetRotVec = targetRot.inv().as_rotvec()
    #camToTargetTransl = targetRot.inv().apply(targetTransl)

    detectedRot = R.from_rotvec(detectedRotVec)    
    camToDSTransl = -detectedRot.inv().apply(detectedTransl)
    targetToDSTransl = detectedRot.inv().apply(targetTransl) - detectedRot.inv().apply(detectedTransl)
    camToTargetRotVec = R.from_rotvec(targetRotVec).inv().as_rotvec()
    targetToCamRotVec = targetRotVec
    camToTargetTransl = -R.from_rotvec(targetRotVec).inv().apply(targetTransl)

    if controlScheme == "PBVS1":
        # PBVS version 1 in chaumette
        def skew(m):
            return [[   0, -m[2],  m[1]], 
                    [ m[2],    0, -m[0]], 
                    [-m[1], m[0],     0]]

        Lx = [] # TODO
        v = -(targetToDSTransl-camToDSTransl + np.matmul(np.linalg.matrix_power(skew(camToDSTransl), 1), camToTargetRotVec))
        w = -camToTargetRotVec
    
    elif controlScheme == "PBVS2":
        # PBVS version 2 in chaumette
        Lx = [] # TODO
        v = -np.matmul(R.from_rotvec(camToTargetRotVec).as_dcm().transpose(), camToTargetTransl)
        w = -camToTargetRotVec

    else:
        raise Exception("Invalid controller '{}'".format(controlScheme))

    velocity = np.concatenate((v, w))
    err = np.concatenate((camToTargetTransl, camToTargetRotVec))

    return velocity

def virtualDetection(dsToCamTransl, 
                     dsToCamRotVec):
    """
    Calculated virtual pose detection based on detection
    """



    # ds to virtual cam
    dsToAUVNEDTransl = self.camToAUVNEDTransl + R.from_rotvec(self.camToAUVNEDRotVec).apply(dsToCamTransl)
    dsToAUVNEDRot = R.from_rotvec(self.camToAUVNEDRotVec)*R.from_rotvec(dsToCamRotVec)
    dsToAUVNEDRotVec = dsToAUVNEDRot.as_rotvec()

    dsToVirtualCamTransl = R.from_rotvec(self.camToAUVNEDRotVec).inv().apply(dsToAUVNEDTransl)
    dsToVirtualCamRotVec = dsToCamRotVec
    
    return dsToVirtualCamTransl, dsToVirtualCamRotVec
    # target to cam
    targetToCamTransl = targetToVirtualCamTransl - R.from_rotvec(self.camToAUVNEDRotVec).inv().apply(self.camToAUVNEDTransl)
    targetToCamRotVec = targetToVirtualCamRotVec.copy()