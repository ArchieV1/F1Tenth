from math import isnan, isinf, degrees, radians, atan, cos, sin

import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
import numpy as np


class WallFollowing:
    SCAN_TOPIC = "scan"
    DRIVE_TOPIC = "wf_drive"

    STRAIGHTS_SPEED = 5.0 / 2.5
    CORNERS_SPEED = 3.0 / 5
    STRAIGHTS_STEERING_ANGLE = radians(10)

    TARGET_RIGHT_DIST = 1  # In metres
    LOOKAHEAD = 0.5  # Look 1m ahead

    def __init__(self):
        self.scan_subscriber = rospy.Subscriber(self.SCAN_TOPIC, LaserScan, self.lidar_callback, queue_size=1)
        self.drive_publisher = rospy.Publisher(self.DRIVE_TOPIC, AckermannDriveStamped, queue_size=10)

    def lidar_callback(self, laser_scan: LaserScan) -> None:
        """
            https://f1tenth-coursekit.readthedocs.io/en/latest/assignments/labs/lab3.html
            D_t0 = Distance to right wall at time t0 (Current time)
            alpha = Angle between car's X axis and wall

            theta = Angle from the car's x axis
                0 <= theta <= 70
            a = Distance to the right wall at angle theta (To the  car's X axis)
            b = Distance to the right wall at angle 0 (To the  car's X axis)

            => alpha = atan([a * cos(theta) - b] / [a * sin(theta)])
            => D_t0 = b * cos(alpha)

            target_dist = target distance to left wall
            error_t = Difference between D_t0 and target_dist (Current dist to wall and target dist to wall)

            => error_t0 = target_dist - D_t0

            PREDICTING THE FUTURE
            D_t1 = Distance to right wall at time t1 (Time in future)
            L = Lookahead distance

            => D_t1 = D_t0 + L * sin(alpha)
        """
        # Current info
        theta_d = 50
        theta = radians(theta_d)

        # Local angles
        a = self.get_distance(laser_scan, 90)  # Dist to right wall at x axis
        b = self.get_distance(laser_scan, 90 + theta_d)  # Dist to right wall at theta deg to right axis (Going towards +y)
        # DEFINITELY + or alpha is HUGE

        alpha = atan((a * cos(theta) - b) / (a * sin(theta)))  # Angle away from local +y axis

        # rospy.loginfo_throttle(2, f"Alpha: {degrees(alpha)}. Right dist {a}, b: {b}")

        dist_t0 = b * cos(alpha)  # Distance to wall at y=0 global pos
        dist_t1 = dist_t0 + self.LOOKAHEAD * sin(alpha)  # Distance to wall at y=0 in 1 LOOKAHEAD of distance in future

        # TODO decide  what lookahead. Product of speed * time to execute all of this??
        # Difference from where car will be in 1 LOOKAHEAD to where want car to be
        error_t0 = self.TARGET_RIGHT_DIST - dist_t0
        error_t1 = self.TARGET_RIGHT_DIST - dist_t1

        angle = self.PID_controller(error_t0, error_t1)

        angle = alpha * 1.5
        rospy.loginfo_throttle(2, f"Alpha: {degrees(alpha)}. Angle: {degrees(angle)}")
        self.publish_drive_msg(angle)

    def get_distance(self, data: LaserScan, angle: float) -> float:
        """
            Get distance of object from scan at angle `angle`

            data: LaserScan from /scan
            angle: Angle in degrees from -45 to 225 (0 is directly to the right)

            Returns: Distance to object at angle angle
        """
        angle = int(round(angle))
        if isnan(data.ranges[angle]) or isinf(data.ranges[angle]):
            return 100
        return data.ranges[angle]

    def publish_drive_msg(self, angle: float) -> None:
        """
            Angle in radians
        """
        if abs(angle) > self.STRAIGHTS_STEERING_ANGLE:
            speed = self.CORNERS_SPEED
        else:
            speed = self.STRAIGHTS_SPEED

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = rospy.Time.now()
        drive_msg.header.frame_id = "laser"
        drive_msg.drive.steering_angle = angle
        drive_msg.drive.speed = speed

        rospy.loginfo_throttle(2, f"Angle: {degrees(angle)}")
        self.drive_publisher.publish(drive_msg)

    def PID_controller(self, prev_error: float, curr_error: float) -> float:
        # https://en.wikipedia.org/wiki/Ziegler–Nichols_method
        """
            error_0 = Previous error term
            error_1 = Current error term
        """
        prev_error *= 5
        curr_error *= 5
        # https://youtu.be/qIpiqhO3ITY?t=1718
        # V0 =  Kp * err + Kd * (prev - curr)

        # Tune these
        kp = 0.01
        kd = 2
        ki = 0.0

        e = curr_error

        # Proportional:
        # Kp * e(t)
        proportional = kp * e

        # Differential
        # Kd * (d/dt)[e(t)]
        differential = kd * (prev_error - e)

        # Integral
        # Ki * integral[0, t, e(t')] dt' # Between 0 and t
        integral = 0
        integral = ki * integral

        rospy.loginfo_throttle(2, f"Prop: {proportional}, diff: {differential}")
        return proportional + differential + integral
        # https://github.com/KlrShaK/Wall_following_F1thenth/blob/main/race/scripts/control.py


def main() -> None:
    rospy.init_node("wf", anonymous=True)
    WallFollowing()
    rospy.spin()


if __name__ == '__main__':
    print("WallFollowing running...")
    try:
        main()
    except rospy.ROSInterruptException:
        pass