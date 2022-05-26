#!/usr/bin/env python
import math

import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from math import radians, atan, tan
import numpy as np


class FollowTheGap:
    SCAN_TOPIC = "scan"
    DRIVE_TOPIC = "ftg_improv_drive"

    BUBBLE_RADIUS = 160  # 160cm
    PREPROCESS_CONV_SIZE = 3
    TARGET_CONVOLVE_SIZE = 80
    MAX_LIDAR_DIST = 10  # 10m
    MIN_ZEROING_RANGE = 3  # 3m
    MIN_WALL_DIST = 1

    STRAIGHTS_SPEED = 5.0
    CORNERS_SPEED = 3.0
    STRAIGHTS_STEERING_ANGLE = radians(10)

    def __init__(self):
        self.radians_per_elem = None

        self.scan_subscriber = rospy.Subscriber(self.SCAN_TOPIC, LaserScan, self.process_lidar, queue_size=1)
        self.drive_publisher = rospy.Publisher(self.DRIVE_TOPIC, AckermannDriveStamped, queue_size=10)

        self.CAR_WIDTH = rospy.get_param("width", 0.2032)

    def preprocess_lidar(self, ranges: list) -> np.array:
        """
            Preprocess the scan data by:
                Removing all data less than MAX_LIDAR_DIST
                Removing all data outside of the 270 in front of the car
        """
        self.radians_per_elem = (2 * np.pi) / len(ranges)

        # Remove lidar data from behind car
        proc_ranges = np.array(ranges[135:-135])

        # Set each value to the mean over a PREPROCESS_CONV_SIZE window (Averages the data)
        proc_ranges = np.convolve(proc_ranges, np.ones(self.PREPROCESS_CONV_SIZE), 'same') / self.PREPROCESS_CONV_SIZE
        proc_ranges = np.clip(proc_ranges, 0, self.MAX_LIDAR_DIST)
        return proc_ranges

    def find_max_gap(self, ranges: np.array) -> tuple:
        """
            Return: the start index & end index of the max gap in free_space_ranges
            free_space_ranges: List of scan data which contains a group of sequential zeros
        """
        return 0, len(ranges)
        # mask the bubble
        # masked = np.ma.masked_where(free_space_ranges == 0, free_space_ranges)

        # get a slice for each contiguous sequence of non-bubble data
        # slices = np.ma.notmasked_contiguous(masked)
        max_len = 0
        chosen_slice = -1

        for i, val in enumerate(ranges):
            slice_len = slic.stop - slic.start
            if slice_len > max_len:
                # IMPROVEMENT: Check car will actually fit through gap
                max_dist_i = np.argmax(ranges[slic.start:slic.stop])
                max_dist = ranges[slic.start + max_dist_i]

                smaller_slice_len = min([max_dist_i - slic.start, slic.stop - max_dist_i])
                theta = smaller_slice_len * self.radians_per_elem

                if abs(max_dist * tan(theta)) > self.CAR_WIDTH / 2:
                    max_len = slice_len
                    chosen_slice = slic

        if chosen_slice != -1:
            return chosen_slice.start, chosen_slice.stop
        else:
            return -1, -1

    def find_target(self, start_i: int, end_i: int, ranges: np.array) -> int:
        """
            indexes: Start and end indices of max-gap range
            Return: index of target within the ranges
        """
        # Do a sliding window average over the data in the max gap
        # Will help the car to avoid hitting corners (Without this the car hits tight corners)
        averaged_max_gap = np.convolve(ranges[start_i:end_i], np.ones(self.TARGET_CONVOLVE_SIZE), 'same') \
                           / self.TARGET_CONVOLVE_SIZE

        # best_dist_i = averaged_max_gap.argmax()

        # IMPROVEMENT: Check car will fit through gap with selected target.
        # If it will fit through gap is already checked in find_max_gap() but if the target is the furthest
        # left/right index it may still hit wall
        lst = averaged_max_gap#ranges[start_i:end_i]
        best_dist = -1
        best_dist_i = -1
        num_same_num = [[0, 0]]  # List of: [best distance, number of values equal to best distance]
        for i, dist in enumerate(lst):
            if dist > best_dist:
                # Check if all of the values within the car's width are far enough away that they wont get hit
                theta = atan((self.CAR_WIDTH / 2) / dist)
                i_width = math.ceil(theta / self.radians_per_elem)

                min_i = i - i_width
                max_i = i + i_width
                if min_i < 0:
                    min_i = 0
                if max_i >= len(lst):
                    max_i = len(lst) - 1
                car_width_vals = lst[min_i: max_i]
                if len(car_width_vals) != 0:
                    if car_width_vals.argmin() > self.MIN_WALL_DIST:
                        # If X numbers are the same gotta pick the middle one of them to be "best_dist_i" not first one
                        if float_equal(dist, best_dist, error=0.01):  # Within 1cm
                            for j, val in enumerate(num_same_num):
                                if float_equal(dist, val[0], error=0.01):
                                    val[1] += 1
                                    break
                        else:
                            num_same_num.append([dist, 0])

                        best_dist = dist
                        best_dist_i = i

        # If best dist is the same as the next X values next to it make sure to return the one in the middle
        # Not first one found
        # rospy.loginfo_throttle(0.25, f"LIST {num_same_num}")
        for i, val in enumerate(num_same_num):
            if float_equal(best_dist, val[0], error=0.01):
                # rospy.loginfo_throttle(0.25, f"{np.argmax(ranges)} vs {best_dist_i} -> {best_dist_i + round(val[1] / 2)}")
                best_dist_i = best_dist_i + round(val[1] / 2)

        # rospy.loginfo_throttle(0.15, f"I:{best_dist_i + start_i}vs{np.argmax(averaged_max_gap) + start_i}\n{lst[best_dist_i+start_i]}{lst[np.argmax(averaged_max_gap) + start_i]}\n{lst}")
        # if best_dist_i != np.argmax(averaged_max_gap):
        #     rospy.loginfo_throttle(1, f"{best_dist_i + start_i}, {np.argmax(averaged_max_gap) + start_i}")
        # rospy.loginfo_throttle(0.25, f"{best_dist_i} / {len(ranges)}")
        return best_dist_i# + start_i

    def get_angle(self, range_index, range_len) -> float:
        """
            Calculate the angle of a particular element in the scan data and transform it into an appropriate steering
            angle
        """
        lidar_angle = (range_index - (range_len / 2)) * self.radians_per_elem
        steering_angle = lidar_angle / 2
        return steering_angle

    def publish_drive_msg(self, angle: float, speed: float = float("inf")) -> None:
        """
            Publish the final steering angle and speed to self.DRIVE_TOPIC
            Speed is determined by:
                self.STRAIGHTS_STEERING_ANGLE
                self.CORNERS_SPEED
                self.STRAIGHTS_SPEED
        """
        # If using default speed value (`float("inf")` is default as it will never be used or calculated in code)
        if speed == float("inf"):
            if abs(angle) > self.STRAIGHTS_STEERING_ANGLE:
                speed = self.CORNERS_SPEED
            else:
                speed = self.STRAIGHTS_SPEED
        else:
            speed = 0

        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = rospy.Time.now()
        drive_msg.header.frame_id = "laser"

        drive_msg.drive.steering_angle = angle
        drive_msg.drive.speed = speed

        self.drive_publisher.publish(drive_msg)

    def process_lidar(self, laser_scan: LaserScan) -> None:
        """
            Process each scan as described by the FollowTheGap algorithm then publish drive message
        """
        # Preprocess the Lidar Information (Remove extra info)
        proc_ranges = self.preprocess_lidar(laser_scan.ranges)

        # Find closest point to car
        # closest = proc_ranges.argmin()

        # IMPROVEMENT: Do not bother with zeroing if not necessary
        # if closest > self.MIN_ZEROING_RANGE:
        #     # Eliminate all points inside 'bubble' (set them to zero)
        #     min_index = closest - self.BUBBLE_RADIUS
        #     max_index = closest + self.BUBBLE_RADIUS
        #     if min_index < 0:
        #         min_index = 0
        #     if max_index >= len(proc_ranges):
        #         max_index = len(proc_ranges) - 1
        #     proc_ranges[min_index:max_index] = 0

        # Find the target
        indexes = self.find_max_gap(proc_ranges)
        # If car will fit through gap
        if indexes != (-1, -1):
            target = self.find_target(*indexes, proc_ranges)

            # Get the final steering angle and publish it
            angle = self.get_angle(target, len(proc_ranges))
            self.publish_drive_msg(angle)
        else:
            rospy.logerr(f"Car will not fit through any gaps. Stopping car")
            # self.publish_drive_msg(0, 0)


def float_equal(val1: float, val2: float, error: float = 0.05) -> bool:
    """
        Returns True if val1 and val2 are within error
    """
    return val1 - error <= val2 <= val1 + error


def main() -> None:
    rospy.init_node("ftg_improv", anonymous=True)
    FollowTheGap()
    rospy.spin()


if __name__ == '__main__':
    print("FTGi running...")
    try:
        main()
    except rospy.ROSInterruptException:
        pass
