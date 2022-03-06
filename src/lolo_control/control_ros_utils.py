import rospy
import numpy as np
from geometry_msgs.msg import TwistStamped, Quaternion, Vector3
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R

def odometryToState(msg):
    state = np.array([0.]*12)
    state[0] = msg.pose.pose.position.x
    state[1] = msg.pose.pose.position.y
    state[2] = msg.pose.pose.position.z
    r = R.from_quat([msg.pose.pose.orientation.x,
                     msg.pose.pose.orientation.y,
                     msg.pose.pose.orientation.z,
                     msg.pose.pose.orientation.w])
    az, ay, ax = r.as_euler("ZYX") # NED convention, yaw, pitch, roll
    state[3:6] = ax, ay, az
    state[6] = msg.twist.twist.linear.x
    state[7] = msg.twist.twist.linear.y
    state[8] = msg.twist.twist.linear.z
    state[9] = msg.twist.twist.angular.x
    state[10] = msg.twist.twist.angular.y
    state[11] = msg.twist.twist.angular.z

    return state

def stateToOdometry(frameId, childFrameID, state, timeStamp=None):
    msg = Odometry()
    msg.header.stamp = timeStamp if timeStamp else rospy.Time.now()
    msg.header.frame_id = frameId
    msg.child_frame_id = childFrameID
    msg.pose.pose.position.x = state[0]
    msg.pose.pose.position.y = state[1]
    msg.pose.pose.position.z = state[2]

    ax, ay, az = state[3:6]
    q = R.from_euler("ZYX", (az, ay, ax)).as_quat()
    msg.pose.pose.orientation = Quaternion(*q)
    msg.twist.twist.linear = Vector3(*state[6:9])
    msg.twist.twist.angular = Vector3(*state[9:12])

    return msg

def velToTwist(frameID, vel, timeStamp=None):
    msg = TwistStamped()
    msg.header.frame_id = frameID
    msg.header.stamp = timeStamp if timeStamp else rospy.Time.now()
    msg.twist.linear.x = vel[0]
    msg.twist.linear.y = vel[1]
    msg.twist.linear.z = vel[2]
    msg.twist.angular.x = vel[3]
    msg.twist.angular.y = vel[4]
    msg.twist.angular.z = vel[5]

    return msg

def twistToVel(msg):
    vel = np.array([0., 0., 0., 0., 0., 0.])
    vel[0] = msg.twist.linear.x
    vel[1] = msg.twist.linear.y
    vel[2] = msg.twist.linear.z
    vel[3] = msg.twist.angular.x
    vel[4] = msg.twist.angular.y
    vel[5] = msg.twist.angular.z

    return vel