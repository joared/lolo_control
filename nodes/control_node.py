#!/usr/bin/env python
import time
import numpy as np
import rospy
import tf
import tf.msg

from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from smarc_msgs.msg import ThrusterRPM

from lolo_perception.perception_ros_utils import vectorToTransform, poseToVector
from lolo_control.control_utils import velToTwist, odometryToState

from lolo_simulation.coordinate_system import CoordinateSystemArtist
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# TODO: put these in a config file and load as paramter
C1 = 1.768*10e-2
C2 = -0.6
C3 = 2.974*10e-7
C4 = -0.4
C5 = -4.1
C6 = -0.19
C7 = 0.22
C8 = -1.51
C9 = -1.3
C10 = 0.21
C11 = -0.4
C12 = 2.0
C13 = 0.4
C14 = 1.0

class ControlNode:
    def __init__(self):

        self.dockingStationName = "docking_station"
        self.auvName = "lolo"

        self.listener = tf.TransformListener()

        # First get the transform from camera to AUV base link 
        while not rospy.is_shutdown():
            try:
                camToAUVTransl, camToAUVQuat = self.listener.lookupTransform(self.auvName + "/base_link",
                                                                             "lolo_camera_link",
                                                                             rospy.Time(0))
            except:
                rospy.loginfo_throttle(2, "Waiting for transform from {} to {}".format("lolo_camera_link", self.auvName + "/base_link"))
            else:
                self.camToAUVTransl = np.array(camToAUVTransl)
                self.camToAUVRotVec = R.from_quat(camToAUVQuat).as_rotvec()
                break
                
        #self.twistPublisher = rospy.Publisher("lolo/twist_command", TwistStamped, queue_size=1)

        self._estDSPoseMsg = None
        self.dsPoseSubscriber = rospy.Subscriber('docking_station/feature_model/estimated_pose', PoseWithCovarianceStamped, self._dsPoseCallback) 

        self.rudderPub = rospy.Publisher('core/rudder_cmd', Float32, queue_size=1)
        self.elevatorPub = rospy.Publisher('core/elevator_cmd', Float32, queue_size=1)
        #self.elevon_stbd_angle = rospy.Publisher('core/elevon_strb_cmd', Float32, queue_size=1)
        #self.elevon_port_angle = rospy.Publisher('core/elevon_port_cmd', Float32, queue_size=1)
        self.thrusterPub = rospy.Publisher('core/thruster_cmd', ThrusterRPM, queue_size=1)

        self._odometryMsg = None
        self._auvState = None
        self.odometrySub = rospy.Subscriber("core/odometry", Odometry, self._odometryCallback)

        self.transformPublisher = rospy.Publisher("/tf", tf.msg.tfMessage, queue_size=1)
        
        self.hz = 30
        self.dt = 1. / self.hz

        self.velAUV = np.array([0., 0., 0., 0., 0., 0.]) # [vx, vy, vz, wx, wy, wz]
        self.controlCamera = False

        self.vxI = 0
        self.vxErrPrev = 0

        self.wzI = 0
        self.vyErrPrev = 0

        self.wyI = 0
        self.vzErrPrev = 0

        fig = plt.figure()
        self.ax = fig.add_subplot(111, projection='3d')

    def _dsPoseCallback(self, msg):
        self._estDSPoseMsg = msg

    def _odometryCallback(self, msg):
        self._odometryMsg = msg

    def publishControlFrames(self):
        timeStamp = rospy.Time.now()

        targetTrans = np.array([0., 0., -9.])
        targetRot = R.from_euler("XYZ", (np.pi/2, 0, -np.pi/2)).as_rotvec()
        targetTransform = vectorToTransform(self.dockingStationName + "/feature_model_estimated_link",
                                            "target_link",
                                            targetTrans,
                                            targetRot,
                                            timeStamp=timeStamp)
        

        targetTrans[2] -= 3
        losTransform = vectorToTransform(self.dockingStationName + "/feature_model_estimated_link",
                                    "los_link",
                                    targetTrans,
                                    targetRot,
                                    timeStamp=timeStamp)

        self.transformPublisher.publish(tf.msg.tfMessage([targetTransform, losTransform]))

    def _calcThruster(self, uRef, pTime=1):
        """
        pTime - time to achieve uRef in seconds
        """
        P = self.dt/pTime
        u = self._auvState[6]
        pitch = self._auvState[5]
        deltaMG = 0
        X = C1*u + C2*abs(u)*u -deltaMG*np.sin(pitch)

        if P*(uRef-u)/self.dt < X:
            return 0

        return np.sqrt((P*(uRef-u)/self.dt - X)/C3)

    def _calcDeltaR(self, rRef, pTime=1):
        P = self.dt/pTime
        r = self._auvState[11]
        u = self._auvState[6]

        if u == 0:
            return 0

        return (P*(rRef-r)/self.dt - C8*r - C9*abs(r)*r) / (C10*u**2)

    def _calcDeltaE(self, qRef, pTime=1):
        P = self.dt/pTime
        u = self._auvState[6]
        pitch = self._auvState[4]
        q = self._auvState[10]

        if u == 0:
            return 0

        return (P*(qRef-q)/self.dt - C4*q - C5*abs(q)*q - C6*np.sin(pitch)) / (C7*u**2)

    def _calcControlFrames(self):
        if self._estDSPoseMsg:
            dsToCamTransl, dsToCamRotVec = poseToVector(self._estDSPoseMsg)
            self._estDSPoseMsg = None

            dsToCamRot = R.from_rotvec(dsToCamRotVec)
            losToDSRot = R.from_euler("XYZ", (np.pi/2, 0, -np.pi/2))

            dsToAUVTransl = self.camToAUVTransl + R.from_rotvec(self.camToAUVRotVec).apply(dsToCamTransl)
            losToAUVRot = R.from_rotvec(self.camToAUVRotVec)*dsToCamRot*losToDSRot
            #losToAUVRotMat = losToAUVRot.as_dcm()

            targetToLoloTrans = dsToAUVTransl + losToAUVRot.apply([9, 0, 0])
            losToLoloTrans = dsToAUVTransl + losToAUVRot.apply([12, 0, 0])
            return targetToLoloTrans, losToLoloTrans
        
        rospy.loginfo("No estimated docking station pose retrieved, using same control command")

        return None

    def update(self, dt):
        self.dt = dt
        if self._odometryMsg:
            self._auvState = odometryToState(self._odometryMsg)
        else:
            print("No auv state published")
            return

        ret = self._calcControlFrames()
        if ret is None:
            self.velAUV = np.array([1.5, 0, 0, 0, 0, 0])
        else:
            targetToLoloTrans, losToLoloTrans = ret
            # Los controller
            losRotX = losToLoloTrans/np.linalg.norm(losToLoloTrans)
            losRotY = losRotX.copy()
            losRotY[2] = 0 # project to xy-plane
            losRotY = np.array([-losRotY[1], losRotY[0], 0]) # y is x rotated by 90 deg
            losRotY = losRotY/np.linalg.norm(losRotY) # normalize
            losRotZ = np.cross(losRotX, losRotY)
            losRotZ = losRotZ/np.linalg.norm(losRotZ)

            losRotMat = np.stack([losRotX, losRotY, losRotZ], axis=1)

            #roll, pitch, yaw = R.from_dcm(losRotMat).as_euler("XYZ")
            yaw, pitch, roll = R.from_dcm(losRotMat).as_euler("ZYX")

            self.ax.cla()
            csRef = CoordinateSystemArtist()
            cs = CoordinateSystemArtist()
            cs.cs.rotation = losRotMat
            csRef.draw(self.ax)
            cs.draw(self.ax)
            self.ax.plot3D(*zip(*[[0]*3, losRotX]), color="black", linewidth=2)
            s = 1
            self.ax.set_xlim(-s, s)
            self.ax.set_ylim(-s, s)
            self.ax.set_zlim(-s, s)
            plt.pause(0.0001)

            # vx
            vxErr = targetToLoloTrans[0]
            vxP = vxErr*0.9
            vxD = (vxErr-self.vxErrPrev)*0.1
            self.vxI += vxErr*.01
            self.vxErrPrev = vxErr

            # wz
            vyErr = -targetToLoloTrans[1]
            vyD = (vyErr-self.vyErrPrev)*1.0
            wz = -yaw*.1
            if vyErr < 1:
                self.wzI += vyErr*.0005
            self.vyErrPrev = vyErr

            # wy
            vzErr = targetToLoloTrans[2]
            vzD = (vzErr-self.vzErrPrev)*1.0
            wy = -pitch*0.1
            self.vzErrPrev = vzErr

            self.velAUV[0] = vxP + vxD + self.vxI 
            self.velAUV[5] = wz + vyD + self.wzI
            self.velAUV[4] = wy + vzD#+ self.wyI
            
            
            #self.velAUV[0] = min(100., self.velAUV[0])
            #self.velAUV[0] = max(0., self.velAUV[0])
            #print("VX:", self.velAUV[0])
            #self.velAUV[5] = max(-1., self.velAUV[5])
            #self.velAUV[5] = min(1., self.velAUV[5])

        if self.controlCamera:
            #Rotating the camera towards the detected feature model
            try:
                trueTrans, trueRot = self.listener.lookupTransform(self.auvName + "_camera_link", 
                                                                self.dockingStationName + "/feature_model_estimated_link", 
                                                                rospy.Time(0))
            except:
                #self.auv.controlCamera(0, P=0.01)
                return
            

            #deltaYaw = np.arctan(trueTrans[0]/trueTrans[2])
            #self.auv.controlCameraDelta(deltaYaw, P=0.01)

    def _velToControlCommand(self):
        n = self._calcThruster(self.velAUV[0], pTime=1) # Achieve ref velocity in pTime sec
        deltaR = self._calcDeltaR(self.velAUV[5], pTime=1)
        deltaE = self._calcDeltaE(self.velAUV[4], pTime=1)
        return n, deltaR, deltaE

    def publishControlCommand(self):
        if self._auvState is None:
            return
        n, deltaR, deltaE = self._velToControlCommand()
        self.rudderPub.publish(Float32(deltaR))
        self.elevatorPub.publish(Float32(deltaE))
        self.thrusterPub.publish(ThrusterRPM(n))
        #self.twistPublisher.publish(velToTwist(self.auvName + "/base_link", self.velAUV))

    def run(self):
        rate = rospy.Rate(self.hz)
        while not rospy.is_shutdown():
            self.publishControlFrames()
            self.update(1./self.hz)
            self.publishControlCommand()
            rate.sleep()


if __name__ == "__main__":
    import os
    import rospkg

    rospy.init_node("control_node")

    controlNode = ControlNode()
    controlNode.run()