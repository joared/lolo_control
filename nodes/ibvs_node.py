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
from lolo_control.control_utils import calcIBVS, calcPBVS
from lolo_perception.camera_model import Camera

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

        camera = Camera(cameraMatrix=np.array(msg.P, dtype=np.float32).reshape((3,4))[:, :3], 
                        distCoeffs=np.zeros((4,1), dtype=np.float32),
                        resolution=(msg.height, msg.width))

        self.camera = camera

        # We only want one message
        self.cameraInfoSub.unregister()

    def _dsPoseCallback(self, msg):
        self._estDSPoseMsg = msg

    def publishControlImg(self):
        if self.controlImg is not None:
            self.controlImgPublisher.publish(self.bridge.cv2_to_imgmsg(self.controlImg))
            self.virtualControlImgPublisher.publish(self.bridge.cv2_to_imgmsg(self.virtualControlImg))

    def _calcVirtualTarget(self):
        
        if self._estDSPoseMsg:

            # ds to cam
            dsToCamTransl, dsToCamRotVec = poseToVector(self._estDSPoseMsg)
            self._estDSPoseMsg = None 

            self.controlImg = np.zeros((self.camera.resolution[0], self.camera.resolution[1], 3), dtype=np.uint8)
            self.virtualControlImg = np.zeros((self.camera.resolution[0], self.camera.resolution[1], 3), dtype=np.uint8)

            # target to virtual cam
            targetToVirtualCamTransl = np.array([0., -.33, 10.])
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

            controlScheme = ""
            if controlScheme == "IBVS":
            
                velVirtualCamera = calcIBVS(targetToVirtualCamTransl, 
                                            targetToVirtualCamRotVec,
                                            dsToVirtualCamTransl,
                                            dsToVirtualCamRotVec,
                                            self.camera,
                                            self.featureModel,
                                            controlScheme="IBVS1",
                                            drawImg=self.virtualControlImg)
            else:
                velVirtualCamera = calcPBVS(targetToVirtualCamTransl, 
                                            targetToVirtualCamRotVec, 
                                            dsToVirtualCamTransl, 
                                            dsToVirtualCamRotVec, 
                                            controlScheme="PBVS2", 
                                            drawImg=None)
            
            velLinear = R.from_rotvec(self.camToAUVNEDRotVec).apply(velVirtualCamera[:3])
            velAngular = R.from_rotvec(self.camToAUVNEDRotVec).apply(velVirtualCamera[3:])

            velLinear *= 0.5
            velAngular *= 0.5

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