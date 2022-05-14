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
from tf.transformations import quaternion_from_euler
from scipy.spatial.transform import Rotation

"""
Planner Helpers
"""


@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.
    Same as nearest_point_on_line_segment, but vectorized. This method is quite fast, time constraints should
    not be an issue so long as trajectories are not insanely long.
        Order of magnitude: trajectory length: 1000 --> 0.0002 second computation (5000fps)
    point: size 2 numpy array
    trajectory: Nx2 matrix of (x,y) trajectory waypoints
        - these must be unique. If they are not unique, a divide by 0 error will destroy the world
    """
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    # this is equivalent to the elementwise dot product
    # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
    dots = np.empty((trajectory.shape[0] - 1,))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0
    # t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1, :] + (t * diffs.T).T
    # dists = np.linalg.norm(point - projections, axis=1)
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp * temp))
    min_dist_segment = np.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment


@njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(point, radius, trajectory, t=0.0, wrap=False):
    """
    starts at beginning of trajectory, and find the first point one radius away from the given point along the trajectory.
    Assumes that the first segment passes within a single radius of the point
    http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
    """
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0] - 1):
        start = trajectory[i, :]
        end = trajectory[i + 1, :] + 1e-6
        V = np.ascontiguousarray(end - start)

        a = np.dot(V, V)
        b = 2.0 * np.dot(V, start - point)
        c = np.dot(start, start) + np.dot(point, point) - 2.0 * np.dot(start, point) - radius * radius
        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            continue
        #   print "NO INTERSECTION"
        # else:
        # if discriminant >= 0.0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0 * a)
        t2 = (-b + discriminant) / (2.0 * a)
        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1
            first_i = i
            first_p = start + t1 * V
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2
            first_i = i
            first_p = start + t2 * V
            break
    # wrap around to the beginning of the trajectory if no intersection is found1
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0], :]
            end = trajectory[(i + 1) % trajectory.shape[0], :] + 1e-6
            V = end - start

            a = np.dot(V, V)
            b = 2.0 * np.dot(V, start - point)
            c = np.dot(start, start) + np.dot(point, point) - 2.0 * np.dot(start, point) - radius * radius
            discriminant = b * b - 4 * a * c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0 * a)
            t2 = (-b + discriminant) / (2.0 * a)
            if t1 >= 0.0 and t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    return first_p, first_i, first_t


@njit(fastmath=False, cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    """
    Returns actuation
    """
    # Extract the Waypoint information
    waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2] - position)
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.
    # Define the radius of the arc to follow
    radius = 1 / (2.0 * waypoint_y / lookahead_distance ** 2)

    # Calculate the steering angle based on the curvature of the arc to follow
    steering_angle = np.arctan(wheelbase / radius)

    return speed, steering_angle


class PurePursuitPlanner:
    """
    Example Planner
    """

    def __init__(self, conf, wb):
        self.wheelbase = wb
        self.conf = conf
        self.load_waypoints(conf)
        self.max_reacquire = 20.

        self.drawn_waypoints = []

    def load_waypoints(self, conf):
        """
        loads waypoints
        """
        self.waypoints = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)

    def render_waypoints(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """

        # points = self.waypoints

        points = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T

        scaled_points = 50. * points

        for i in range(points.shape[0]):
            if len(self.drawn_waypoints) < points.shape[0]:
                b = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
                                ('c3B/stream', [183, 193, 222]))
                self.drawn_waypoints.append(b)
            else:
                self.drawn_waypoints[i].vertices = [scaled_points[i, 0], scaled_points[i, 1], 0.]

    def _get_current_waypoint(self, waypoints, lookahead_distance, position, theta):
        """
        gets the current waypoint to follow
        """
        wpts = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T
        nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(position, wpts)
        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(position, lookahead_distance, wpts,
                                                                                    i + t, wrap=True)
            if i2 == None:
                return None
            current_waypoint = np.empty((3,))
            # x, y
            current_waypoint[0:2] = wpts[i2, :]
            # speed
            current_waypoint[2] = waypoints[i, self.conf.wpt_vind]
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return np.append(wpts[i, :], waypoints[i, self.conf.wpt_vind])
        else:
            return None

    def plan(self, pose_x, pose_y, pose_theta, lookahead_distance, vgain):
        """
        gives actuation given observation
        """
        # Get the current Position of the car
        position = np.array([pose_x, pose_y])

        # Search for the next waypoint to track based on lookahead distance parameter
        lookahead_point = self._get_current_waypoint(self.waypoints, lookahead_distance, position, pose_theta)

        if lookahead_point is None:
            return 4.0, 0.0

        # Calculate the Actuation: Steering angle and speed
        speed, steering_angle = get_actuation(pose_theta, lookahead_point, position, lookahead_distance, self.wheelbase)
        speed = vgain * speed

        return speed, steering_angle


def main2():
    """
    main entry point
    """
    return
    work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 0.8, 'vgain': 1.0}

    with open('config_Spielberg_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    planner = PurePursuitPlanner(conf, 0.17145 + 0.15875)

    def render_callback(env_renderer):
        planner.render_waypoints(env_renderer)

    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1)
    env.add_render_callback(render_callback)

    obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
    env.render()

    laptime = 0.0
    start = time.time()

    while not done:
        speed, steer = planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], work['tlad'],
                                    work['vgain'])
        obs, step_reward, done, info = env.step(np.array([[steer, speed]]))
        laptime += step_reward
        env.render(mode='human')

    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)


class PurePursuit:
    POSE_TOPIC = "/gt_pose"
    DRIVE_TOPIC = "/pp_drive"

    LOOKAHEAD_DEFAULT = 7
    __LOOKAHEAD_DIFFERENCE = LOOKAHEAD_DEFAULT / 2
    MAX_WAYPOINT_DISTANCE = LOOKAHEAD_DEFAULT + __LOOKAHEAD_DIFFERENCE
    MIN_WAYPOINT_DISTANCE = LOOKAHEAD_DEFAULT - __LOOKAHEAD_DIFFERENCE

    STRAIGHTS_SPEED = 5.0 / 10
    CORNERS_SPEED = 3.0 / 10
    STRAIGHTS_STEERING_ANGLE = radians(10)
    STEERING_ANGLE_CONSTANT = 1  # Curvature = K (2|y|)/(L^2)

    WHEELBASE = 0.3302
    """Constant from car size"""

    target_dist_diff = None

    local_viable_waypoints = None
    """Waypoints that are ahead of the car and within reasonable MAX_WAYPOINT_LOOKAHEAD > X > MIN_WAYPOINT_LOOKAHEAD"""
    VIABLE_WAYPOINT_ANGLE = radians(70)
    """Positive number in radians. Will look +- this number to find viable waypoints"""

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
        rospy.loginfo("==========PP POSE_CALLBACK==========")
        self.pose_current = pose_stamped.pose
        self.header = pose_stamped.header

        # rospy.loginfo(f"Current pos: {pose_stamped.pose}")
        self.global_curr_x = self.pose_current.position.x
        self.global_curr_y = self.pose_current.position.y

        # self.global_curr_angle = (2 * np.pi) - (2 * acos(self.pose_current.orientation.w))
        # self.global_curr_angle = 2 * acos(self.pose_current.orientation.w)

        quat = [self.pose_current.orientation.x,
                self.pose_current.orientation.y,
                self.pose_current.orientation.z,
                self.pose_current.orientation.w]
        rot = Rotation.from_quat(quat)
        self.global_curr_angle = rot.as_euler('xyz', degrees=False)[2]
        if self.global_curr_angle < 0:
            self.global_curr_angle += radians(360)  # Translate from -180 -> +180 to 0 -> 360

        rospy.loginfo(f"Angle of car: {round(degrees(self.global_curr_angle), 2)}")

        # If it has either reached target or isn't moving (eg Planner was turned off) then calculate new target to go to
        if self.is_first_move() or self.is_not_moving() or self.has_reached_target():
            # Generate df of waypoints to be used
            self.calc_viable_waypoints()
            rospy.loginfo(f"Viable: {len(self.local_viable_waypoints)}")

            self.find_target()

            angle = self.calc_drive_angle()
            self.publish_drive_msg(angle)
        else:
            rospy.loginfo(
                f"Not reached target G({round(self.global_tar_x, 3)}, {round(self.global_tar_y, 3)}) "
                f"L({round(self.local_tar_x, 3)}, {round(self.local_tar_y, 3)}). "
                f"Current: G({round(self.pose_current.position.x, 3)}, {round(self.pose_current.position.y, 3)}) "
                f"(Previous G({round(self.pose_previous.position.x, 3)}, {round(self.pose_previous.position.y, 3)})")

        self.pose_previous = self.pose_current

    def is_first_move(self) -> bool:
        """
            Returns true is no previous pose or if no target
        """
        if self.pose_previous is None or self.global_tar_x is None or self.global_tar_y is None:
            rospy.loginfo(f"FirstMove")
            return True
        return False

    def is_not_moving(self, error: float = 0.0003) -> bool:
        """
            Returns true if current pos is different than last pos

            Use a tight error tolerance as the car's position updates often
        """

        if self.float_equal_double(self.global_curr_x, self.pose_previous.position.x, self.global_curr_y, self.pose_previous.position.y, error=error):
            rospy.loginfo(f"NotMoving")
            return True
        return False

    def has_reached_target(self, error: float = 0.10) -> bool:
        """
            Returns True if car's current x AND y coords are within error of target coords
        """

        if self.float_equal_double(self.global_tar_x, self.global_curr_x, self.global_curr_y, self.global_tar_y, error=error):
            rospy.loginfo("HasReachedTarget")
            return True
        return False

    def calc_viable_waypoints(self) -> None:
        """
            Calculates dataframe of viable waypoints
            Assigns self.viable_waypoints

            Viable waypoints are:
                Within +-VIABLE_WAYPOINT_ANGLE rad of the direction of the car
                In front of the car
                Greater than MIN_WAYPOINT_DISTANCE away
                Less than MAX_WAYPOINT_DISTANCE way
        """

        # Convert global coords to local coords (of the car)
        columns = ["x", "y"]
        local_waypoints = pd.DataFrame(columns=columns)
        for i, row in self.global_waypoints.iterrows():
            x, y = self.global_to_local_coords(row[0], row[1])

            df = pd.DataFrame([[x, y]], columns=columns)
            local_waypoints = pd.concat([local_waypoints, df])

        local_waypoints.reset_index(drop=True, inplace=True)

        # Get all values in front of the car

        # Angle created by the x/y of the coords greater than wanted angle
        # |arctan(y/x)| > theta
        # Constraints:
        # x != 0  # Will break the division
        # y >= 0

        waypoints = pd.DataFrame(columns=columns)
        for i, row in local_waypoints.iterrows():
            x = row[0]
            y = row[1]

            # # rospy.loginfo(f"({x}, {y})")
            # if y != 0 and x != 0:  # Not underneath car
            #     if y >= 0:  # In front of car
            #         # rospy.loginfo(f"{y} >= 0")
            #         if x == 0 or atan(y / x) > self.VIABLE_WAYPOINT_ANGLE:  # Within the viable angle
            #             # rospy.loginfo(f"{x} == 0 OR greater than angle")
            #             distance_from_car = self.distance_to_point(x, y)
            #             if True:  # self.MAX_WAYPOINT_DISTANCE > distance_from_car > self.MIN_WAYPOINT_DISTANCE:
            #                 # rospy.loginfo(f"{distance_from_car} within range {self.MIN_WAYPOINT_DISTANCE} - {self.MAX_WAYPOINT_DISTANCE}")
            #                 df = pd.DataFrame([[x, y]], columns=columns)
            #                 waypoints = pd.concat([waypoints, df])

            # If in front of car
            # if (x == 0 and y > 0) or (y > 0 and abs(atan(y / x)) > self.VIABLE_WAYPOINT_ANGLE):
            if y > 0:
                df = pd.DataFrame([[x, y]], columns=columns)
                waypoints = pd.concat([waypoints, df])

        waypoints.reset_index(drop=True, inplace=True)  # Stop every index being "0"

        self.local_viable_waypoints = waypoints

        if len(self.local_viable_waypoints) == 0:
            rospy.logerr(f"No Viable Waypoints\n"
                         f"Global Waypoints: {self.global_waypoints}\n\n"
                         f"Local Waypoints: {local_waypoints}\n\n"
                         f"Viable Waypoints: {self.local_viable_waypoints}")
            exit()

    def find_target(self) -> None:
        """
            Find the nearest waypoint to LOOKAHEAD
            Sets:
                self.target_dist_diff to the signed difference of the distance of the target from the car to LOOKAHEAD
                self.target_ind to the index of the target waypoint in self.viable_waypoints
        """

        smallest_diff_val = float("inf")
        smallest_diff_ind = 0

        for i, row in self.local_viable_waypoints.iterrows():
            dist_from_car = self.distance_to_point(row[0], row[1])
            diff_dist_lookahead = dist_from_car - self.lookahead_dist

            if abs(diff_dist_lookahead) < abs(smallest_diff_val):
                smallest_diff_ind = i
                smallest_diff_val = diff_dist_lookahead

        self.target_dist_diff = smallest_diff_val

        # Convert local to global coords
        local_x = self.local_viable_waypoints["x"].iloc[smallest_diff_ind]
        local_y = self.local_viable_waypoints["y"].iloc[smallest_diff_ind]

        global_x, global_y = self.local_to_global_coords(local_x, local_y)

        # Set local positioning variables
        self.local_tar_x = local_x
        self.local_tar_y = local_y
        self.global_tar_x = global_x
        self.global_tar_y = global_y

        # rospy.loginfo(f"TargetLocal: {local_x}, {local_y}")
        # rospy.loginfo(f"TargetGlobal: {global_x}, {global_y}")

        if self.target_dist_diff is float("inf"):
            rospy.logerr("No Viable Targets")
            exit()

        rospy.loginfo("Viable waypoints /\\")
        rospy.loginfo(self.local_viable_waypoints)

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

            Returns: The angle (In radians)
        """

        """
        Paper used:
        https://www.ri.cmu.edu/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf

        Curvature = 2x / L^2
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
            return 0

        curvature = 2 * waypoint_x / distance ** 2

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
        self.STEERING_ANGLE_CONSTANT = 0.55 # y doesnt get small enough
        angle = radians((self.WHEELBASE / curvature) * self.STEERING_ANGLE_CONSTANT)

        angle = radians(self.WHEELBASE * curvature)

        rospy.loginfo(f"D: {distance}, x: {waypoint_x}, curv: {curvature}")
        rospy.loginfo(
            f"Aiming for ({round(waypoint_x, 3)}, {round(waypoint_y, 3)}) (Gl: {round(self.global_tar_x, 3)},"
            f"{round(self.global_tar_y, 3)}) at {round(degrees(angle), 3)}deg (+{round(self.target_dist_diff, 4)}m from"
            f" {self.lookahead_dist}) ({round(distance, 4)}m away)")
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

    def global_to_local_coords(self, tar_x: float, tar_y: float) -> (float, float):
        # https://gamedev.stackexchange.com/a/109377
        # Checked this answer and think is correct

        x = tar_x - self.global_curr_x
        y = tar_y - self.global_curr_y

        theta = degrees(self.global_curr_angle)

        loc_x = x * cos(theta) + y * sin(theta)
        loc_y = -x * sin(theta) + y * cos(theta)

        # TODO remove test (Line below)
        # rospy.loginfo(f"GLOBAL/LOCAL->GLOBAL: ({round(degrees(theta), 2)}) {tar_x}, {tar_y} == {self.local_to_global_coords(loc_x, loc_y)}?")

        return loc_x, loc_y

    def local_to_global_coords(self, x: float, y: float) -> (float, float):
        # https://gamedev.stackexchange.com/a/109377
        # Checked this answer and think is correct

        theta = degrees(self.global_curr_angle)

        global_x = x * cos(theta) - y * sin(theta) + self.global_curr_x
        global_y = x * sin(theta) + y * cos(theta) + self.global_curr_y

        return global_x, global_y

    def float_equal(self, val1: float, val2: float, error: float = 0.05) -> bool:
        """
            Returns True if val1 and val2 are within error
        """
        return val1 - error <= val2 <= val1 + error

    def float_equal_double(self, x1: float, x2: float, y1: float, y2: float, error: float = 0.05) -> bool:
        """
            Returns True if x1 and x2 are the within error AND y1 and y2 are within error
        """
        return self.float_equal(x1, x2, error=error) and self.float_equal(y1, y2, error=error)


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

    rospy.loginfo(f"Moving car to:\n{message}")
    initialpose_publisher.publish(message)


def main(args: List[str]) -> None:
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

    initialise_car_pos(waypoints)

    PurePursuit(waypoints)

    rospy.spin()


if __name__ == '__main__':
    print("PP running...")
    try:
        main(sys.argv)
    except rospy.ROSInterruptException:
        pass
