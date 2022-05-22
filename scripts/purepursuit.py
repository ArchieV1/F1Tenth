#!/usr/bin/env python
import io
import sys
import time
from math import sqrt, degrees, radians, cos, acos, sin, tan, atan, atan2
from typing import TextIO, List

import roslaunch
import rospy
import tf
import yaml
import numpy as np
from argparse import Namespace
import pandas as pd

from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped, Pose, Quaternion, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from numba import njit
from pyglet.gl import GL_POINTS
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from scipy.spatial.transform import Rotation


class PurePursuit:
    POSE_TOPIC = "/gt_pose"
    DRIVE_TOPIC = "/pp_drive"

    LOOKAHEAD_DEFAULT = 2
    # __LOOKAHEAD_DIFFERENCE = LOOKAHEAD_DEFAULT / 2
    # MAX_WAYPOINT_DISTANCE = LOOKAHEAD_DEFAULT + __LOOKAHEAD_DIFFERENCE
    # MIN_WAYPOINT_DISTANCE = LOOKAHEAD_DEFAULT - __LOOKAHEAD_DIFFERENCE

    STRAIGHTS_SPEED = 5.0 #/ 5
    CORNERS_SPEED = 3.0 #`/ 5
    STRAIGHTS_STEERING_ANGLE = radians(10)
    STEERING_ANGLE_CONSTANT = 1  # Curvature = K (2|y|)/(L^2)

    WHEELBASE = 0.3302
    """Constant from car size"""
    distance_from_rear_wheel_to_front_wheel = 0.5  # From pmusau17  # TODO not needed

    target_dist_diff = None
    """Distance of target from self.LOOKAHEAD"""

    VIABLE_WAYPOINT_ANGLE = radians(20)
    """Positive number of radians from the +X axis (Straight ahead).
    Only waypoints where (LOCAL) `tan(x/y)<VIABLE_WAYPOINT_ANGLE` will be considered viable waypoints"""

    header = None
    global_waypoints = None
    """Global coords"""
    lookahead_dist = None
    """Lookahead distance in metres"""

    pose_subscriber = None
    drive_publisher = None

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

    def __init__(self, waypoints: pd.DataFrame, lookahead: float = LOOKAHEAD_DEFAULT):
        self.global_waypoints = waypoints
        self.lookahead_dist = lookahead

        self.pose_subscriber = rospy.Subscriber(self.POSE_TOPIC, PoseStamped, self.pose_callback, queue_size=1)
        self.drive_publisher = rospy.Publisher(self.DRIVE_TOPIC, AckermannDriveStamped, queue_size=10)

    def pose_callback(self, pose_stamped: PoseStamped):
        self.pose_current = pose_stamped.pose
        self.header = pose_stamped.header

        self.global_curr_x = self.pose_current.position.x
        self.global_curr_y = self.pose_current.position.y

        quat = (self.pose_current.orientation.x,
                self.pose_current.orientation.y,
                self.pose_current.orientation.z,
                self.pose_current.orientation.w)

        euler = euler_from_quaternion(quat)
        self.global_curr_angle = np.double(euler[2])  # From +X towards +Y
        # if self.global_curr_angle < 0:
        #     self.global_curr_angle += radians(360)  # Translate from -180 -> +180 to 0 -> 360

        # Caching for global_to_local and local_to_global
        self.cos_theta = cos(self.global_curr_angle)
        self.sin_theta = sin(self.global_curr_angle)

        # rospy.loginfo(f"Angle of car: {round(degrees(self.global_curr_angle), 2)}")

        # Find taget and assing target values
        self.calc_target_waypoint()
        # rospy.loginfo(f"Viable: {len(self.local_viable_waypoints)}")

        angle = self.calc_drive_angle()
        self.publish_drive_msg(angle)

        self.pose_previous = self.pose_current

    def is_first_move(self) -> bool:
        """
            Returns true is no previous pose or if no target
        """
        if self.pose_previous is None or self.global_tar_x is None or self.global_tar_y is None:
            # rospy.loginfo(f"FirstMove")
            return True
        return False

    def is_not_moving(self, error: float = 0.00003) -> bool:
        """
            Returns true if current pos is different than last pos

            Use a tight error tolerance as the car's position updates often
        """

        if float_equal_double(self.global_curr_x, self.pose_previous.position.x, self.global_curr_y,
                                   self.pose_previous.position.y, error=error):
            # rospy.loginfo(f"NotMoving")
            return True
        return False

    def has_reached_target(self, error: float = 0.10) -> bool:
        """
            Returns True if car's current x AND y coords are within error of target coords
        """

        if float_equal_double(self.global_tar_x, self.global_curr_x, self.global_curr_y, self.global_tar_y,
                                   error=error):
            # rospy.loginfo("HasReachedTarget")
            return True
        return False

    def calc_target_waypoint(self) -> None:
        """
            Calculates target waypoint from self.waypoints
            Assigns:
                self.global_tar_x
                self.global_tar_y
                self.local_tar_x
                self.local_tar_y

            Viable waypoints are:
                ##Within +-VIABLE_WAYPOINT_ANGLE rad of the direction of the car
                In front of the car
                ##Greater than MIN_WAYPOINT_DISTANCE away
                ##Less than MAX_WAYPOINT_DISTANCE way
        """

        # Convert global coords to local coords (of the car)
        # Then filter out unviable ones (Eg behind car)
        # Then find best one
        self.target_dist_diff = float("inf")

        for i, row in self.global_waypoints.iterrows():
            local_x, local_y = self.global_to_local_coords(row[0], row[1])
            # rospy.loginfo(f"Localxy: {local_x, local_y}")

            # Anywhere in front of the car (But not underneath)
            if local_x > 0 or (local_x >= 0 and local_y != 0):
                # Within the angle range (Angle is from the +X axis in front of the car)
                if local_y == 0.0 or tan(local_x/local_y) < self.VIABLE_WAYPOINT_ANGLE:
                    dist_from_car = self.distance_to_point(local_x, local_y)
                    diff_dist_lookahead = dist_from_car - self.lookahead_dist

                    # If better than previous (Closer to lookahead)
                    if abs(diff_dist_lookahead) < abs(self.target_dist_diff):
                        # Log the info again
                        self.global_to_local_coords(row[0], row[1], log=True)
                        self.target_dist_diff = diff_dist_lookahead

                        self.global_tar_x = row[0]
                        self.global_tar_y = row[1]

                        self.local_tar_x = local_x
                        self.local_tar_y = local_y

        # rospy.loginfo(f"TargetIs: L{np.round([self.local_tar_x, self.local_tar_y], 2)}; G{np.round([self.global_tar_x, self.global_tar_y], 2)}; Angle{np.round(degrees(self.global_curr_angle), 2)}; SIN/COS{np.round([self.sin_theta, self.cos_theta], 2)}")
        if self.local_tar_x is None:
            rospy.logerr(f"No Viable Waypoints\n"
                         f"Global Waypoints: {self.global_waypoints}\n\n")
            exit()

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
        # distance = self.lookahead_dist

        if distance == 0 or waypoint_x == 0:
            rospy.logerr(f"Distance to target waypoints is 0. This means target waypoint is beneath car. "
                         f"This should not happen")
            return 0

        curvature = 2 * waypoint_x / distance ** 2  # Curvature with +y in straight line and x left and right
        # curvature = 2 * waypoint_y / distance ** 2

        radius = 1 / curvature
        """"
            Quoting the paper:
            "The curvature is transformed into steering wheel angle by the vehicle’s on board controller."
            
            The angle is proportional to the curvature and the wheelbase (Because of ackerman steering)
            
            https://www.racecar-engineering.com/articles/tech-explained-ackermann-steering-geometry/
            For a given turn radius R, wheelbase L, and track width T, engineers calculate the required front steering
            angles (δ_(f,in) and δ_(f,out)) with the following expressions:
            δ_(f,in) = atan(L / (R - T/2))
            δ_(f,out) = atan(L / (R - T/2))
            
            => angle = c0 * (wheelbase/radius)
            => curvature = c1/radius
            k1 = c0 * c1
            => radius = k1/curvature
            
            angle = wheelbase/(k1/curvature)
            => (k1 * angle)/curvature = wheelbase
            => k1 * angle = wheelbase * curvature
            => angle = (wheelbase * curvature) / k1
            k1 proportional to (1 / k2) 
            => angle = wheelbase * curvature * k2
        """

        self.STEERING_ANGLE_CONSTANT = 0.75  # y doesnt get small enough
        self.STEERING_ANGLE_CONSTANT = 0.85  # y not low enough
        self.STEERING_ANGLE_CONSTANT = 1  #
        self.STEERING_ANGLE_CONSTANT = 2
        self.STEERING_ANGLE_CONSTANT = 0.55  # y doesnt get small enough
        angle = radians((self.WHEELBASE / curvature) * self.STEERING_ANGLE_CONSTANT)

        # Translate from 0 -> +360 to -180 -> +180 (As that's what car takes)
        self.WHEELBASE = 1
        self.STEERING_ANGLE_CONSTANT = 1
        to_angle = (self.WHEELBASE * curvature * self.STEERING_ANGLE_CONSTANT)

        # # Rate of change of curvature gives m of tangent y=mx + c which gives angle
        # rate_curvature = 2 * (waypoint_x ** 2 - waypoint_y ** 2)/((waypoint_x ** 2 + waypoint_y ** 2) ** 2)  # angle rel +x
        # rate_curvature = 4 * (waypoint_x * waypoint_y) / ((waypoint_x ** 2 + waypoint_y ** 2) ** 2)  # angle rel +y
        """
            y = mx
            m = y/x
            1/m = x/y
            m = 2x / (x^2 + y^2)
            atan(o/a) = atan (x/y)
            atan(1/m) = atan((x^2 + y^2) / 2x)
        """
        to_angle = degrees(atan(self.WHEELBASE / curvature))

        if to_angle >= 180:
            to_angle = 360 - to_angle
        else:
            to_angle *= -1

        # If x is on the right hand side of the car turn right. Otherwise turn left
        # +x coord is left side of car
        # -angle is steer left
        # if waypoint_x > 0:
        #     to_angle = to_angle

        # From Paulius
        # https://github.com/pmusau17/Platooning-F1Tenth/blob/noetic-port/src/pure_pursuit/scripts/pure_pursuit_angle.py
        angle = atan2(self.local_tar_y, self.local_tar_x)
        # Straight line instead of curve though isn't it?

        # region From github
        # # TODO delete From https://github.com/f1tenth-dev/pure_pursuit/blob/master/scripts/vehicle_controller.py
        # WHEELBASE_LEN = self.WHEELBASE
        # ang_goal_x = self.global_tar_x
        # ang_goal_y = self.global_tar_y
        #
        # curr_x = self.global_curr_x
        # curr_y = self.global_curr_y
        # heading = self.global_curr_angle
        #
        # GOAL_RIGHT = "goal_to_right"
        # GOAL_LEFT = "goal_to_left"
        # GOAL_ON_LINE = "goal_on_line"
        #
        # MSG_A = "goal at {}m"
        # MSG_B = "goal at {}m bearing {} {}"
        # MSG_GOAL = "recieved new goal: ({}, {})"
        #
        # ANGLE_RANGE_A = 45.0  # 60.0
        # ANGLE_RANGE_B = 30.0
        # ANGLE_RANGE_C = 15.0
        # ANGLE_RANGE_D = 5.0
        #
        # eucl_d = sqrt(pow(ang_goal_x - curr_x, 2) + pow(ang_goal_y - curr_y, 2))
        # curvature = degrees(2.0 * (abs(ang_goal_x) - abs(curr_x)) / (pow(eucl_d, 2)))
        # steering_angle = atan(curvature * WHEELBASE_LEN)
        # theta = atan2(ang_goal_y - curr_y, ang_goal_x - curr_x)
        #
        # proj_x = eucl_d * cos(heading) + curr_x
        # proj_y = eucl_d * sin(heading) + curr_y
        #
        # proj_eucl_shift = sqrt(pow(proj_x - ang_goal_x, 2) + pow(proj_y - ang_goal_y, 2))
        #
        # angle_error = acos(1 - (pow(proj_eucl_shift, 2) / (2 * pow(eucl_d, 2))))
        # angle_error = degrees(angle_error)
        #
        # goal_sector = (ang_goal_x - curr_x) * (proj_y - curr_y) - (ang_goal_y - curr_y) * (proj_x - curr_x)
        #
        # if goal_sector > 0:
        #     goal_sector = GOAL_RIGHT
        # elif goal_sector < 0:
        #     goal_sector = GOAL_LEFT
        # else:
        #     goal_sector = GOAL_ON_LINE
        #
        # if goal_sector == GOAL_ON_LINE:
        #     msg = MSG_A.format(round(eucl_d, 2))
        # else:
        #     msg = MSG_B.format(round(eucl_d, 2), round(angle_error, 2), goal_sector)
        #
        # if angle_error > ANGLE_RANGE_A:
        #     angle_error = ANGLE_RANGE_A
        # steering_angle = angle_error / ANGLE_RANGE_A
        #
        # if goal_sector == GOAL_RIGHT:
        #     steering_angle = -1.0 * steering_angle
        # angle = steering_angle
        # endregion

        # #https://github.com/Nolancw98/F1Tenth/blob/main/slam_pure_pursuit
        # kp = 0.5
        # integral =
        # error = curvature
        # P = kp * error  # proportional component
        # I = integral + ki * error * TIME_INCREMENT  # integral component
        # D = kd * (error - prev_error) / TIME_INCREMENT  # derivative component
        # angle = P + I + D  # angle output is sum of components

        # rospy.loginfo(f"D: {distance}, x: {waypoint_x}, curv: {curvature}")
        # rospy.loginfo_throttle(1,
        #                        f"Aiming for L({round(waypoint_x, 3)}, {round(waypoint_y, 3)}) "
        #                        f"Gl({round(self.global_tar_x, 3)}, {round(self.global_tar_y, 3)}) at {round(degrees(angle), 3)}deg "
        #                        f"(+{round(self.target_dist_diff, 4)}m from {self.lookahead_dist}) ({round(distance, 4)}m away) "
        #                        f"G->L({np.round(self.global_to_local_coords(self.global_tar_x, self.global_tar_y), 3)} @ {round(degrees(self.global_curr_angle), 3)})")
        return angle

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


def initialise_car_pos(waypoints: pd.DataFrame, init_waypoint: int = 0, target_waypoint: int = 1) -> None:
    """
        Teleport car to initial position/orientation
        Orientation is the angle between the first and second waypoints

        The car will be in the position of the initial waypoint and facing the second
    """
    initialpose_publisher = rospy.Publisher("/initialpose", PoseWithCovarianceStamped, queue_size=1)

    message = PoseWithCovarianceStamped()
    message.header.stamp = rospy.Time.now()
    message.header.frame_id = "map"

    message.pose.pose.position.x = waypoints.iloc[init_waypoint]["x"]
    message.pose.pose.position.y = waypoints.iloc[init_waypoint]["y"]
    d_x = abs(message.pose.pose.position.x - waypoints.iloc[target_waypoint]["x"])
    d_y = abs(message.pose.pose.position.y - waypoints.iloc[target_waypoint]["y"])

    # https://automaticaddison.com/how-to-convert-euler-angles-to-quaternions-using-python/
    # https://answers.ros.org/question/181689/computing-posewithcovariances-6x6-matrix/
    roll = 0
    pitch = 0
    yaw = -atan2(d_y, d_x)

    q = quaternion_from_euler(roll, pitch, yaw)
    message.pose.pose.orientation = Quaternion(*q)

    # Taken from the messages F1TenthSimulator sends when using "2D pose estimate"
    message.pose.covariance = [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.06853892326654787]

    # Check f1tenth subscriber is active
    while initialpose_publisher.get_num_connections() < 1:  # Set to 2 if using `rostopic echo /initialpose`
        time.sleep(0.05)
        rospy.loginfo("Waiting for subscribers before positioning car")

    # rospy.loginfo(f"Moving car to:\n{message}")
    initialpose_publisher.publish(message)


def main(args: list) -> None:
    # https://vinesmsuic.github.io/2020/09/29/robotics-purepersuit/#importance-of-visualizations
    # Interesting way to smooth waypoints

    rospy.init_node("pure_pursuit", anonymous=True)

    # Load raceline (The path to follow on this map)
    map_uri = args[1]

    # raceline_uri = map_uri.replace("map.yaml", "raceline.csv")
    # raceline_uri = map_uri.replace("map.yaml", "DonkeySim_waypoints.txt")

    raceline_uri = map_uri.replace("map.yaml", "centerline.csv")
    waypoints = pd.read_csv(raceline_uri, delimiter=",", dtype=float, header=0)
    waypoints.rename(columns={"# x_m": "x", " y_m": "y"}, inplace=True)

    # # TODO remove this. Selects only 2 rows
    # waypoints = waypoints.iloc[[0, 10]]

    initialise_car_pos(waypoints)

    PurePursuit(waypoints)

    rospy.spin()


if __name__ == '__main__':
    print("PP running...")
    try:
        main(sys.argv)
    except rospy.ROSInterruptException:
        pass
