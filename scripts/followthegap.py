#!/usr/bin/env python
import time

import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from math import radians
import numpy as np


class FollowTheGap:
    SCAN_TOPIC = "scan"
    DRIVE_TOPIC = "ftg_drive"

    BUBBLE_RADIUS = 160  # 160cm
    PREPROCESS_CONV_SIZE = 3
    BEST_POINT_CONV_SIZE = 80
    MAX_LIDAR_DIST = 3000000  # 3m

    STRAIGHTS_SPEED = 5.0
    CORNERS_SPEED = 3.0
    STRAIGHTS_STEERING_ANGLE = radians(10)

    cycle_times = []

    def __init__(self):
        self.radians_per_elem = None

        self.scan_subscriber = rospy.Subscriber(self.SCAN_TOPIC, LaserScan, self.process_lidar, queue_size=1)
        self.drive_publisher = rospy.Publisher(self.DRIVE_TOPIC, AckermannDriveStamped, queue_size=10)

    def preprocess_lidar(self, ranges: list) -> np.array:
        """
            Preprocess the scan data by:
                Removing all data less than MAX_LIDAR_DIST
                Removing all data outside of the 270 in front of the car
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

    def process_lidar(self, laser_scan: LaserScan) -> None:
        """
            Process each scan as described by the FollowTheGap algorithm then publish drive message
        """
        start_time = time.time_ns() / 10**9

        # Preprocess the Lidar Information (Remove extra info)
        proc_ranges = self.preprocess_lidar(laser_scan.ranges)

        # Find closest point to car
        closest = proc_ranges.argmin()

        # Eliminate all points inside 'bubble' (Set them to zero)
        min_index = closest - self.BUBBLE_RADIUS
        max_index = closest + self.BUBBLE_RADIUS
        if min_index < 0:
            min_index = 0
        if max_index >= len(proc_ranges):
            max_index = len(proc_ranges) - 1
        proc_ranges[min_index:max_index] = 0

        # Find the target
        indexes = self.find_max_gap(proc_ranges)
        target = self.find_target(*indexes, proc_ranges)

        # Get the final steering angle and publish it
        angle = self.get_angle(target, len(proc_ranges))
        self.publish_drive_msg(angle)

        end_time = time.time_ns() / 10**9
        self.cycle_times.append(end_time - start_time)
        rospy.loginfo_throttle(15, f"{np.mean(self.cycle_times)}")


def main() -> None:
    rospy.init_node("ftg", anonymous=True)
    FollowTheGap()
    rospy.spin()


if __name__ == '__main__':
    print("FTG running...")
    try:
        main()
    except rospy.ROSInterruptException:
        pass
