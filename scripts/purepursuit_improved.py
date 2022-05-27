#!/usr/bin/env python
import math
import sys
import time
from math import sqrt, radians, cos, sin, tan, atan2

import rospy
import numpy as np
import pandas as pd

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion


class PurePursuit:
    POSE_TOPIC = "/gt_pose"
    DRIVE_TOPIC = "/pp_improv_drive"
    SCAN_TOPIC = "scan"

    LOOKAHEAD_DEFAULT = 2

    STRAIGHTS_SPEED = 5.0
    CORNERS_SPEED = 3.0
    STRAIGHTS_STEERING_ANGLE = radians(10)
    STEERING_ANGLE_CONSTANT = 1  # Curvature = K (2|y|)/(L^2)

    WHEELBASE = 0.3302
    """Constant from car size"""

    target_dist_diff = None
    """Distance of target from self.LOOKAHEAD"""

    VIABLE_WAYPOINT_ANGLE = radians(20)
    """Positive number of radians from the +X axis (Straight ahead).
    Only waypoints where (LOCAL) `tan(x/y)<VIABLE_WAYPOINT_ANGLE` will be considered viable waypoints"""

    global_waypoints = None
    """Global coords"""
    lookahead_dist = None
    """Lookahead distance in metres"""

    scan = None
    """The most recent LiDAR scan"""
    radians_per_elem = None
    CAR_DIVIDER = 1.9
    """What do divide car width by to get "half" of the car's width"""

    pose_previous = None
    pose_current = None

    # Current pos global
    global_curr_x = None
    global_curr_y = None
    global_curr_angle = None
    """
        Angle relative to the +X axis going towards +Y (Anti clockwise)
        Between 0 and 360 NOT -180 and +180
    """
    # To speed up global_to_local
    # Both equal to -current_angle
    cos_theta = None
    sin_theta = None

    # Local current everything will always be `0`

    # Current target global and local
    global_tar_x = None
    global_tar_y = None
    local_tar_x = None
    local_tar_y = None

    cycle_times = []

    def __init__(self, waypoints: pd.DataFrame, lookahead: float = LOOKAHEAD_DEFAULT):
        self.global_waypoints = waypoints
        self.lookahead_dist = lookahead

        self.CAR_WIDTH = rospy.get_param("width", 0.2032)

        self.pose_subscriber = rospy.Subscriber(self.POSE_TOPIC, PoseStamped, self.pose_callback, queue_size=1)
        self.scan_subscriber = rospy.Subscriber(self.SCAN_TOPIC, LaserScan, self.scan_callback, queue_size=1)
        self.drive_publisher = rospy.Publisher(self.DRIVE_TOPIC, AckermannDriveStamped, queue_size=10)

    def scan_callback(self, scan: LaserScan):
        self.scan = scan
        self.radians_per_elem = (2 * np.pi) / len(scan.ranges)

    def pose_callback(self, pose_stamped: PoseStamped):
        start_time = time.time_ns() / 10**9

        if self.scan is None:
            return

        self.pose_current = pose_stamped.pose

        self.global_curr_x = self.pose_current.position.x
        self.global_curr_y = self.pose_current.position.y

        quat = (self.pose_current.orientation.x,
                self.pose_current.orientation.y,
                self.pose_current.orientation.z,
                self.pose_current.orientation.w)

        euler = euler_from_quaternion(quat)
        self.global_curr_angle = np.double(euler[2])  # From +X towards +Y

        # Caching for global_to_local and local_to_global
        self.cos_theta = cos(self.global_curr_angle)
        self.sin_theta = sin(self.global_curr_angle)

        # Find taget and assing target values
        self.calc_target_waypoint()
        # rospy.loginfo(f"Viable: {len(self.local_viable_waypoints)}")

        angle = self.calc_drive_angle()
        self.publish_drive_msg(angle)

        self.pose_previous = self.pose_current

        end_time = time.time_ns() / 10**9
        self.cycle_times.append(end_time - start_time)
        # rospy.loginfo_throttle(15, f"{np.mean(self.cycle_times)}")

    def is_first_move(self) -> bool:
        """
            Returns true is no previous pose or if no target
        """
        return self.pose_previous is None or self.global_tar_x is None or self.global_tar_y is None

    def is_not_moving(self, error: float = 0.00003) -> bool:
        """
            Returns true if current pos is different than last pos

            Use a tight error tolerance as the car's position updates often
        """

        return float_equal_double(self.global_curr_x, self.pose_previous.position.x, self.global_curr_y,
                                   self.pose_previous.position.y, error=error)

    def has_reached_target(self, error: float = 0.10) -> bool:
        """
            Returns True if car's current x AND y coords are within error of target coords
        """

        return float_equal_double(self.global_tar_x, self.global_curr_x, self.global_curr_y, self.global_tar_y,
                                   error=error)

    def calc_target_waypoint(self) -> None:
        """
            Calculates target waypoint from self.waypoints
            Assigns:
                self.global_tar_x
                self.global_tar_y
                self.local_tar_x
                self.local_tar_y

            Viable waypoints are:
                In front of the car
                Not behind an object
        """

        # Convert global coords to local coords (of the car)
        # Then filter out unviable ones (Eg behind car)
        # Then find best one
        self.target_dist_diff = float("inf")

        for i, row in self.global_waypoints.iterrows():
            local_x, local_y = self.global_to_local_coords(row[0], row[1])

            # Anywhere in front of the car (But not underneath)
            if local_x > 0 or (local_x >= 0 and local_y != 0):
                # Within the angle range (Angle is from the +X axis in front of the car)
                if local_y == 0.0 or tan(local_x/local_y) < self.VIABLE_WAYPOINT_ANGLE:
                    dist_from_car = self.distance_to_point(local_x, local_y)
                    diff_dist_lookahead = dist_from_car - self.lookahead_dist

                    # If better than previous (Closer to lookahead)
                    if abs(diff_dist_lookahead) < abs(self.target_dist_diff):

                        # If not behind a wall
                        if self.waypoint_not_blocked(local_x, local_y):
                            self.target_dist_diff = diff_dist_lookahead

                            self.global_tar_x = row[0]
                            self.global_tar_y = row[1]

                            self.local_tar_x = local_x
                            self.local_tar_y = local_y

        if self.local_tar_x is None:
            rospy.logerr(f"No Viable Waypoints\n"
                         f"Global Waypoints: {self.global_waypoints}\n\n")
            exit()

    def waypoint_not_blocked(self, x: float, y: float) -> bool:
        """
            x: Local x value of waypoint
            y: Local y value of waypoint

            Returns true if waypoint closer to car than the obstacle in that direction is
        """
        distance_to_waypoint = self.distance_to_point(x, y)
        angle_to_point = self.angle_to_point(x, y)
        index_of_target = math.ceil(angle_to_point)
        distance_to_wall = self.scan.ranges[index_of_target]

        will_hit_wall = True

        # If not directly blocked
        if distance_to_waypoint < distance_to_wall:
            will_hit_wall = False

            # Check if edge of car will clip it
            theta = math.atan((self.CAR_WIDTH / self.CAR_DIVIDER) / distance_to_waypoint)
            bubble_radius = int(math.ceil(theta / self.radians_per_elem))

            min_index = index_of_target - bubble_radius
            max_index = index_of_target + bubble_radius
            if min_index < 0:
                min_index = 0
            if max_index >= len(self.scan.ranges):
                max_index = len(self.scan.ranges) - 1

            for i, dist_wall in enumerate(self.scan.ranges[min_index: max_index]):
                # Using distance_to_waypoint isn't exactly mathematically correct but will do
                if dist_wall < distance_to_waypoint:
                    will_hit_wall = True

        return not will_hit_wall

    def angle_to_point(self, x:float, y: float) -> float:
        angle = atan2(y, x)
        return angle

    def distance_to_point(self, x: float, y: float) -> float:
        """
            Calculates the distance to the given point assuming all coordinates are local
            All returned values will be positive
            Returns sqrt(x ** 2 + y ** 2)
        """
        return sqrt(x ** 2 + y ** 2)

    def calc_drive_angle(self) -> float:
        """
            Calculates the desired drive angle of the car in order to reach the waypoint

            Returns: The angle in radians. In range [-pi, +pi)
        """

        """
        Paper used:
        https://www.ri.cmu.edu/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf

        Curvature = 2x / L^2
        => 2x / (x^2 + y^2)
        Where:
            x = Distance between target's x coords and cars (Car's coord is 0,0 as local coords)
            L = Lookahead distance of the car to find waypoints
                OPTIONAL: Replaced with distance between car and point to increase accuracy
        """

        waypoint_x = self.local_tar_x
        waypoint_y = self.local_tar_y

        # lookahead can be replaced with (target_dist_diff + lookahead) as lookahead will not be exactly equal to the
        # actual distance "L"
        distance = self.distance_to_point(waypoint_x, waypoint_y)

        if distance == 0 or waypoint_x == 0:
            rospy.logerr(f"Distance to target waypoints is 0. This means target waypoint is beneath car. "
                         f"This should not happen")
            return 0

        curvature = 2 * waypoint_y / distance ** 2  # Curvature with +X in straight line and Y left and right
        return curvature

    def publish_drive_msg(self, angle: float) -> None:
        """
            Publish the final steering angle and speed to self.DRIVE_TOPIC
            Speed is determined by:
                self.STRAIGHTS_STEERING_ANGLE
                self.CORNERS_SPEED
                self.STRAIGHTS_SPEED
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

        self.drive_publisher.publish(drive_msg)

    def global_to_local_coords(self, tar_x: float, tar_y: float, log: bool = False) -> (float, float):
        # https://gamedev.stackexchange.com/a/109377
        # Checked this answer and think is correct

        x = tar_x - self.global_curr_x
        y = tar_y - self.global_curr_y

        # Uses constants calculated in pose_callback to speed up calculations
        loc_x = (x * self.cos_theta) + (y * self.sin_theta)
        loc_y = (-x * self.sin_theta) + (y * self.cos_theta)

        return loc_x, loc_y

    def local_to_global_coords(self, x: float, y: float) -> (float, float):
        # https://gamedev.stackexchange.com/a/109377
        # Checked this answer and think is correct

        # Uses constants calculated in pose_callback to speed up calculations
        global_x = (x * self.cos_theta) - (y * self.sin_theta) + self.global_curr_x
        global_y = (x * self.sin_theta) + (y * self.cos_theta) + self.global_curr_y

        return global_x, global_y


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


def main(args: list) -> None:
    # https://vinesmsuic.github.io/2020/09/29/robotics-purepersuit/#importance-of-visualizations
    # Interesting way to smooth waypoints

    rospy.init_node("pure_pursuit_improv", anonymous=True)

    # Load raceline (The path to follow on this map)
    map_uri = args[1]

    # Uncomment to use race line
    # raceline_uri = map_uri.replace("map.yaml", "raceline.csv")
    # waypoints = pd.read_csv(raceline_uri, delimiter=";", dtype=float, header=2)
    # waypoints.rename(columns={" x_m": "x", " y_m": "y"}, inplace=True)
    # waypoints = waypoints[["x", "y"]]

    # Uncomment to use centre line
    raceline_uri = map_uri.replace("map.yaml", "centerline.csv")
    waypoints = pd.read_csv(raceline_uri, delimiter=",", dtype=float, header=0)
    waypoints.rename(columns={"# x_m": "x", " y_m": "y"}, inplace=True)

    PurePursuit(waypoints)

    rospy.spin()


if __name__ == '__main__':
    print("PPi running...")
    try:
        main(sys.argv)
    except rospy.ROSInterruptException:
        pass
