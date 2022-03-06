#!/usr/bin/env python
import time
import numpy as np
import rospy
import tf
import tf.msg
import cv2 as cv

from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from sensor_msgs.msg import CameraInfo, Image
from smarc_msgs.msg import ThrusterRPM

from cv_bridge import CvBridge, CvBridgeError

from lolo_perception.perception_ros_utils import vectorToTransform, poseToVector
from lolo_perception.perception_utils import plotPosePoints, projectPoints, plotPoints
from lolo_control.control_ros_utils import velToTwist, odometryToState


class ControlNode:
    def __init__(self, featureModel, hz):

        self.dockingStationName = "docking_station"
        self.auvName = "lolo"

        self.featureModel = featureModel

        self.listener = tf.TransformListener()

        # First get the transform from camera to AUV base link 
        while not rospy.is_shutdown():
            try:
                camToAUVNEDTransl, camToAUVNEDQuat = self.listener.lookupTransform(self.auvName + "/base_link_ned",
                                                                             "lolo_camera_link",
                                                                             rospy.Time(0))
            except:
                rospy.loginfo_throttle(2, "Waiting for transform from {} to {}".format("lolo_camera_link", self.auvName + "/base_link_ned"))
            else:
                self.camToAUVNEDTransl = np.array(camToAUVNEDTransl)
                self.camToAUVNEDRotVec = R.from_quat(camToAUVNEDQuat).as_rotvec()
                break
                
        self.camera = None
        self.cameraInfoSub = rospy.Subscriber("lolo_camera/camera_info", CameraInfo, self._getCameraCallback)
        while not rospy.is_shutdown() and self.camera is None:
            print("Waiting for camera info to be published")
            rospy.sleep(1)

        self.twistControlPublisher = rospy.Publisher("lolo/twist_command", TwistStamped, queue_size=1)

        self._estDSPoseMsg = None
        self.dsPoseSubscriber = rospy.Subscriber('docking_station/feature_model/estimated_pose', PoseWithCovarianceStamped, self._dsPoseCallback) 

        self.bridge = CvBridge()
        self.controlImg = None
        self.virtualControlImg = None
        self.controlImgPublisher = rospy.Publisher('control/image', Image, queue_size=1)
        self.virtualControlImgPublisher = rospy.Publisher('control/image_virtual', Image, queue_size=1)

        self.hz = hz
        self.dt = 1. / self.hz

        self.velAUV = np.array([0., 0., 0., 0., 0., 0.]) # [vx, vy, vz, wx, wy, wz]

    def _getCameraCallback(self, msg):
        """
        Use either K and D or just P
        https://answers.ros.org/question/119506/what-does-projection-matrix-provided-by-the-calibration-represent/
        https://github.com/dimatura/ros_vimdoc/blob/master/doc/ros-camera-info.txt
        """
        from lolo_perception.camera_model import Camera
        # Using only P (D=0), we should subscribe to the rectified image topic
        camera = Camera(cameraMatrix=np.array(msg.P, dtype=np.float32).reshape((3,4))[:, :3], 
                        distCoeffs=np.zeros((4,1), dtype=np.float32),
                        resolution=(msg.height, msg.width))
        # Using K and D, we should subscribe to the raw image topic
        #_camera = Camera(cameraMatrix=np.array(msg.K, dtype=np.float32).reshape((3,3)), 
        #                distCoeffs=np.array(msg.D, dtype=np.float32),
        #                resolution=(msg.height, msg.width))
        self.camera = camera

        # We only want one message
        self.cameraInfoSub.unregister()

    def _dsPoseCallback(self, msg):
        self._estDSPoseMsg = msg

    def publishControlImg(self):
        if self.controlImg is not None:
            self.controlImgPublisher.publish(self.bridge.cv2_to_imgmsg(self.controlImg))
            self.virtualControlImgPublisher.publish(self.bridge.cv2_to_imgmsg(self.virtualControlImg))

    def interactionMatrix(self, x, y, Z):
        return [[-1/Z, 0, x/Z, x*y, -(1+x*x), y],
                [0, -1/Z, y/Z, 1+y*y, -x*y, -x]]

    def calcIBVSVelocity(self, targetTransl, targetRotVec, detectedTransl, detectedRotVec, controller, drawImg=None):
        targetFeatures = projectPoints(targetTransl, targetRotVec, self.camera, self.featureModel.features)
        detectedFeatures = projectPoints(detectedTransl, detectedRotVec, self.camera, self.featureModel.features)

        if drawImg is not None:
            plotPoints(drawImg, targetFeatures, color=(0,255,255), radius=3)
            plotPoints(drawImg, detectedFeatures, color=(0,0,255), radius=3)

        cx = self.camera.cameraMatrix[0, 2]
        cy = self.camera.cameraMatrix[1, 2]

        fx = self.camera.cameraMatrix[0, 0]
        fy = self.camera.cameraMatrix[1, 1]

        targetFeatures = [((x-cx)/fx, (y-cy)/fy) for x,y in targetFeatures]
        detectedFeatures = [((x-cx)/fx, (y-cy)/fy) for x,y in detectedFeatures]

        zDetected = detectedTransl[2] # TODO: calculated for each detected feature point
        zTarget = targetTransl[2]

        Lx = []
        for target, feat  in zip(targetFeatures, detectedFeatures):
            x = feat[0]
            y = feat[1]
          
            Le = self.interactionMatrix(x, y, zDetected)

            x = target[0]
            y = target[1]
            
            if drawImg is not None:
                cv.line(drawImg, 
                        (int(round(feat[0]*fx+cx)), int(round(feat[1]*fy+cy))), 
                        (int(round(target[0]*fx+cx)), int(round(target[1]*fy+cy))),
                        thickness=1, 
                        color=(0,0,255))

            LeStar = self.interactionMatrix(x, y, zTarget)

            LeLeStar = (np.array(Le) + np.array(LeStar))/2

            if controller == "IBVS1":
                # IBVS version 1 in chaumette
                L = Le
            elif controller == "IBVS2":
                # IBVS version 2 in chaumette
                L = LeStar
            elif controller == "IBVS3":
                # IBVS version 3 in chaumette
                L = LeLeStar
            else:
                raise Exception("Invalid controller '{}'".format(controller))

            #Lx.append(np.row_stack(L))
            Lx.append(L)

        Lx = np.row_stack(Lx)
        LxPinv = np.linalg.pinv(Lx)
        err = np.array([v for f in detectedFeatures for v in f]) - np.array([ v for t in targetFeatures for v in t])
        v = -np.matmul(LxPinv, err)
        return v

    def _calcVirtualTarget(self):
        
        if self._estDSPoseMsg:

            # ds to cam
            dsToCamTransl, dsToCamRotVec = poseToVector(self._estDSPoseMsg)
            self._estDSPoseMsg = None 

            self.controlImg = np.zeros((self.camera.resolution[0], self.camera.resolution[1], 3), dtype=np.uint8)
            self.virtualControlImg = np.zeros((self.camera.resolution[0], self.camera.resolution[1], 3), dtype=np.uint8)

            # target to virtual cam
            targetToVirtualCamTransl = np.array([0., 0., 7.])
            targetToVirtualCamRotVec = np.array([0., 0., 0.])

            # ds to virtual cam
            dsToAUVNEDTransl = self.camToAUVNEDTransl + R.from_rotvec(self.camToAUVNEDRotVec).apply(dsToCamTransl)
            dsToAUVNEDRot = R.from_rotvec(self.camToAUVNEDRotVec)*R.from_rotvec(dsToCamRotVec)
            dsToAUVNEDRotVec = dsToAUVNEDRot.as_rotvec()

            dsToVirtualCamTransl = R.from_rotvec(self.camToAUVNEDRotVec).inv().apply(dsToAUVNEDTransl)
            dsToVirtualCamRotVec = dsToCamRotVec
            
            # target to cam
            targetToCamTransl = targetToVirtualCamTransl - R.from_rotvec(self.camToAUVNEDRotVec).inv().apply(self.camToAUVNEDTransl)
            targetToCamRotVec = targetToVirtualCamRotVec.copy()

            # plot ds to cam
            plotPosePoints(self.controlImg, 
                           dsToCamTransl, 
                           dsToCamRotVec, 
                           self.camera, 
                           self.featureModel.features, 
                           color=(0, 0, 255))
            
            # plot target to cam
            plotPosePoints(self.controlImg, 
                            targetToCamTransl, 
                            targetToCamRotVec, 
                            self.camera, 
                            self.featureModel.features, 
                            color=(0, 255, 255))

            velVirtualCamera = self.calcIBVSVelocity(targetToVirtualCamTransl, 
                                                     targetToVirtualCamRotVec,
                                                     dsToVirtualCamTransl,
                                                     dsToVirtualCamRotVec,
                                                     controller="IBVS1",
                                                     drawImg=self.virtualControlImg)

            velLinear = R.from_rotvec(self.camToAUVNEDRotVec).apply(velVirtualCamera[:3])
            velAngular = R.from_rotvec(self.camToAUVNEDRotVec).apply(velVirtualCamera[3:])

            velLinear *= 0.8
            velAngular *= 0.8

            #self.velAUV = np.array(list(velLinear) + list(velAngular))
            wz, wy, wx = R.from_rotvec(velAngular).as_euler("ZYX")
            self.velAUV = np.array(list(velLinear) + [wx, wy, wz])

            self.publishControlCommand()
            self.publishControlImg()
            
    def update(self, dt):
        self._calcVirtualTarget()

    def publishControlCommand(self):
        self.twistControlPublisher.publish(velToTwist(self.auvName + "/base_link_ned", self.velAUV))

    def run(self):
        rate = rospy.Rate(self.hz)
        while not rospy.is_shutdown():
            self.update(1./self.hz)
            rate.sleep()


if __name__ == "__main__":
    import os
    import rospkg
    from lolo_perception.feature_model import FeatureModel

    rospy.init_node("control_node")

    featureModelYaml = rospy.get_param("~feature_model_yaml")
    hz = rospy.get_param("~hz")
    featureModelYamlPath = os.path.join(rospkg.RosPack().get_path("lolo_perception"), "feature_models/{}".format(featureModelYaml))
    featureModel = FeatureModel.fromYaml(featureModelYamlPath)

    controlNode = ControlNode(featureModel, hz)
    controlNode.run()