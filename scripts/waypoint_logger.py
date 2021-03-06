#!/usr/bin/env python
import rospy
import numpy as np
import atexit
import tf
from os.path import expanduser
from time import gmtime, strftime
from numpy import linalg as LA
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry

# Credits: https://github.com/f1tenth/f1tenth_labs/blob/main/waypoint_logger/scripts/waypoint_logger.py

home = expanduser('~')
file = open(strftime(home+'/catkin_ws/src/f1tenth_simulator/logs/wp-%Y-%m-%d-%H-%M-%S', gmtime())+'.csv', 'w')


def save_waypoint(data):
    quaternion = np.array([data.pose_current.pose_current.orientation.x,
                           data.pose_current.pose_current.orientation.y,
                           data.pose_current.pose_current.orientation.z,
                           data.pose_current.pose_current.orientation.w])

    euler = tf.transformations.euler_from_quaternion(quaternion)
    speed = LA.norm(np.array([data.twist.twist.linear.x, 
                              data.twist.twist.linear.y, 
                              data.twist.twist.linear.z]), 2)

    if data.twist.twist.linear.x > 0.:
        print(data.twist.twist.linear.x)

    file.write('%f, %f, %f, %f\n' % (data.pose_current.pose_current.position.x,
                                     data.pose_current.pose_current.position.y,
                                     euler[2],
                                     speed))


def shutdown():
    file.close()
    print('Goodbye')


def listener():
    rospy.init_node('waypoints_logger', anonymous=True)
    # rospy.Subscriber('pf/pose/odom', Odometry, save_waypoint)
    rospy.Subscriber('odom', Odometry, save_waypoint)

    rospy.spin()


if __name__ == '__main__':
    exit(999)
    atexit.register(shutdown)
    print('Saving waypoints...')
    try:
        listener()
    except rospy.ROSInterruptException:
        pass
