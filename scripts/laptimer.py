#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
import numpy as np
from tf.transformations import euler_from_quaternion


class LapTimer:
    NUMBER_LAPS = 5

    pose_subscriber = None
    POSE_TOPIC = "gt_pose"

    current_lap = 0
    """Current lap number from 0 to (NUMBER_LAPS-1)"""

    pose_previous = None
    pose_current = None
    header = None

    curr_x = None
    curr_y = None
    curr_angle = None
    """From +X to +Y in radians"""

    def __init__(self):
        self.pose_subscriber = rospy.Subscriber(self.POSE_TOPIC, PoseStamped, self.pose_callback, queue_size=1)

    def pose_callback(self, pose_stamped: PoseStamped):
        self.pose_current = pose_stamped.pose
        self.header = pose_stamped.header

        self.curr_x = self.pose_current.position.x
        self.curr_y = self.pose_current.position.y

        quat = (self.pose_current.orientation.x,
                self.pose_current.orientation.y,
                self.pose_current.orientation.z,
                self.pose_current.orientation.w)
        euler = euler_from_quaternion(quat)
        self.curr_angle = np.double(euler[2])  # From +X towards +Y

        self.pose_previous = self.pose_current

    def is_first_move(self) -> bool:
        """
            Returns true is no previous pose or if no target
        """
        return self.pose_previous is None

    def is_not_moving(self, error: float = 0.00003) -> bool:
        """
            Returns true if current pos is different than last pos

            Use a tight error tolerance as the car's position updates often
        """

        return float_equal_double(self.curr_x, self.pose_previous.position.x, self.curr_y,
                                  self.pose_previous.position.y, error=error)


def float_equal(val1: float, val2: float, error: float = 0.05) -> bool:
    """
        Returns True if val1 and val2 are within error
    """
    return val1 - error <= val2 <= val1 + error


def float_equal_double(x1: float, x2: float, y1: float, y2: float, alt_inputs: bool = False,
                       error: float = 0.05) -> bool:
    """
        Returns True if x1 and x2 are the within error AND y1 and y2 are within error
        If alt_inputs is true compares x1/y1 AND x2/y2 instead
    """
    if alt_inputs:
        return float_equal(x1, y1, error=error) and float_equal(x2, y2, error=error)
    return float_equal(x1, x2, error=error) and float_equal(y1, y2, error=error)


def main() -> None:
    rospy.init_node("laptimer", anonymous=True)

    LapTimer()

    rospy.spin()


if __name__ == '__main__':
    print("LapTimer running")
    try:
        main()
    except rospy.ROSInterruptException:
        pass
