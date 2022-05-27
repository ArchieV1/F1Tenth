#!/usr/bin/env python
import math
import time

import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from math import radians
import numpy as np


class FollowTheGapi:
    SCAN_TOPIC = "scan"
    DRIVE_TOPIC = "ftg_improv_drive"

    PREPROCESS_CONV_SIZE = 3
    BEST_POINT_CONV_SIZE = 80
    MAX_LIDAR_DIST = 3000000  # 3m

    STRAIGHTS_SPEED = 5.0  # mps
    CORNERS_SPEED = 3.0  # mps
    CAR_DIVIDER = 1.9
    """What do divide car width by to get "half" of the car's width"""
    STRAIGHTS_STEERING_ANGLE = radians(10)

    BUBBLE_RADIUS = None
    radians_per_elem = None
    """To be set every time a scan is received"""

    cycle_times = []

    def __init__(self):
        self.scan_subscriber = rospy.Subscriber(self.SCAN_TOPIC, LaserScan, self.process_lidar, queue_size=1)
        self.drive_publisher = rospy.Publisher(self.DRIVE_TOPIC, AckermannDriveStamped, queue_size=10)

        self.CAR_WIDTH = rospy.get_param("width", 0.2032)

    def preprocess_lidar(self, ranges: list) -> np.array:
        """
            Preprocess the scan data by:
                Removing all data less than MAX_LIDAR_DIST
                Removing all data outside of the 270degrees in front of the car
        """
        self.radians_per_elem = (2 * np.pi) / len(ranges)

        # Remove lidar data from behind car
        proc_ranges = np.array(ranges[135:-135])

        # Set each value to the mean over a given window (Averages the data)
        proc_ranges = np.convolve(proc_ranges, np.ones(self.PREPROCESS_CONV_SIZE), 'same') / self.PREPROCESS_CONV_SIZE
        proc_ranges = np.clip(proc_ranges, 0, self.MAX_LIDAR_DIST)
        return proc_ranges

    def find_max_gap(self, free_space_ranges: np.array) -> tuple:
        """
            Return: the start index & end index of the max gap in free_space_ranges
            free_space_ranges: List of scan data which contains a group of sequential zeros
        """
        # mask the bubble
        masked = np.ma.masked_where(free_space_ranges == 0, free_space_ranges)

        # get a slice for each contigous sequence of non-bubble data
        slices = np.ma.notmasked_contiguous(masked)
        max_len = slices[0].stop - slices[0].start
        chosen_slice = slices[0]

        for slic in slices[1:]:
            slice_len = slic.stop - slic.start
            if slice_len > max_len:
                max_len = slice_len
                chosen_slice = slic

        # Check if car will fit through gap
        theta = math.atan((self.CAR_WIDTH / self.CAR_DIVIDER) / min(free_space_ranges[chosen_slice.start: chosen_slice.stop]))
        bubble_radius = int(math.ceil(theta / self.radians_per_elem))

        if chosen_slice.stop - chosen_slice.start < bubble_radius * 2:
            # Car will not fit through gap
            return -1, -1
        return chosen_slice.start, chosen_slice.stop

    def find_target(self, start_i: int, end_i: int, ranges: np.array) -> int:
        """
            indexes: Start and end indices of max-gap range
            Return: index of target within the ranges
        """
        # Do a sliding window average over the data in the max gap this will help the car to avoid hitting corners
        averaged_max_gap = np.convolve(ranges[start_i:end_i], np.ones(self.BEST_POINT_CONV_SIZE),
                                       'same') / self.BEST_POINT_CONV_SIZE
        return averaged_max_gap.argmax() + start_i

    def get_angle(self, range_index, range_len) -> float:
        """
            Calculate the angle of a particular element in the scan data and transform it into an appropriate steering
            angle
        """
        lidar_angle = (range_index - (range_len / 2)) * self.radians_per_elem
        steering_angle = lidar_angle / 2
        return steering_angle

    def publish_drive_msg(self, angle: float, vel: float = float("inf")) -> None:
        """
            Publish the final steering angle and speed to self.DRIVE_TOPIC
            Speed is determined by:
                self.STRAIGHTS_STEERING_ANGLE
                self.CORNERS_SPEED
                self.STRAIGHTS_SPEED
        """
        # If speed has been inputted use it. Else calculate it
        if vel != float("inf"):
            speed = vel
        else:
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

    def process_lidar(self, laser_scan: LaserScan) -> None:
        """
            Process each scan as described by the FollowTheGap algorithm then publish drive message
        """
        start_time = time.time_ns() / 10**9
        # Preprocess the Lidar Information (Remove extra info)
        proc_ranges = self.preprocess_lidar(laser_scan.ranges)

        # Find closest point to car
        closest = proc_ranges.argmin()

        # Do not bother doing bubble stuff unless car is close to needing it
        if proc_ranges[closest] < 0.75:
            # Determine bubble radius
            theta = math.atan((self.CAR_WIDTH / self.CAR_DIVIDER) / proc_ranges[closest])  # Should be div by 2 but 1.9 for safety
            bubble_radius = int(math.ceil(theta / self.radians_per_elem))

            # Eliminate all points inside 'bubble' (Set them to zero)
            min_index = closest - bubble_radius
            max_index = closest + bubble_radius
            if min_index < 0:
                min_index = 0
            if max_index >= len(proc_ranges):
                max_index = len(proc_ranges) - 1
            proc_ranges[min_index:max_index] = 0

        # Find the target
        indexes = self.find_max_gap(proc_ranges)
        if indexes == (-1, -1):
            rospy.logerr("Car will not fit through gap. Stopping car")
            self.publish_drive_msg(0, 0)
            return
        target = self.find_target(*indexes, proc_ranges)

        # Get the final steering angle and publish it
        angle = self.get_angle(target, len(proc_ranges))
        self.publish_drive_msg(angle)

        end_time = time.time_ns() / 10**9
        self.cycle_times.append(end_time - start_time)
        # rospy.loginfo_throttle(15, f"{np.mean(self.cycle_times)}")


def main() -> None:
    rospy.init_node("ftg_improv", anonymous=True)
    FollowTheGapi()
    rospy.spin()


if __name__ == '__main__':
    print("FTGi running...")
    try:
        main()
    except rospy.ROSInterruptException:
        pass
