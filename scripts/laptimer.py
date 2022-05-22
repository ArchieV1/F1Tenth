#!/usr/bin/env python
import os
import sys
import time
from datetime import datetime

import rospy
from geometry_msgs.msg import PoseStamped
import numpy as np
import pandas as pd
from std_msgs.msg import Int32MultiArray
from tf.transformations import euler_from_quaternion


class LapTimer:
    POSE_TOPIC = "gt_pose"
    MUX_TOPIC = "mux"

    df_lap_times = None
    """Dataframe with column names: race_start, race_end, race_time"""
    race_not_started = True
    """Set to False when car moves for first time"""
    race_start_time = None
    """The unixtime the race starts in nanoseconds"""
    race_complete = False
    """Set to True when race is complete"""

    pose_previous = None
    pose_current = None
    header = None

    curr_x = None
    curr_y = None
    curr_angle = None
    """From +X to +Y in radians"""
    curr_time = None
    """Current unixtime in nanoseconds. Set every time pose_callback is called"""

    map_name = None
    """The name of the map in use"""
    controller_name = None
    """The name of the controller used in this race"""

    def __init__(self, map_name: str):
        none_array = [None]
        self.df_lap_times = pd.DataFrame({"race_start": none_array, "race_end": none_array, "race_time": none_array})
        self.map_name = map_name

        self.mux_subscriber = rospy.Subscriber(self.MUX_TOPIC, Int32MultiArray, self.mux_callback, queue_size=1)
        self.pose_subscriber = rospy.Subscriber(self.POSE_TOPIC, PoseStamped, self.pose_callback, queue_size=1)

    def pose_callback(self, pose_stamped: PoseStamped) -> None:
        # If done then just make this do nothing and log that it is done
        if self.race_complete:
            rospy.loginfo_once("This race is done. Please restart the simulator with a different pathing method/map "
                               "to start again")
            return

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

        self.curr_time = time.time_ns()

        # If race has not started
        if self.race_not_started and self.pose_previous is not None:
            # If moving then the race has begun
            if self.is_moving():
                rospy.loginfo("fRace has begun!")
                self.df_lap_times.iloc[0]["race_start"] = self.curr_time
                self.race_start_time = self.curr_time
                self.race_not_started = False
        else:
            # If race is done: Write data to file
            if self.race_ended():
                self.df_lap_times.iloc[0]["race_end"] = self.curr_time
                self.df_lap_times.iloc[0]["race_time"] = self.curr_time - self.df_lap_times.iloc[0]["race_start"]

                # Convert unix time to datetime (*1000 to convert from ns to s)
                curr_time = datetime.fromtimestamp(self.curr_time * 1000)
                curr_time = curr_time.strftime("%Y:%m:%d:%H:%M:%S")  # Year:Month:Day:Hour:Minute:Second

                file_name = "../results/" + curr_time + "." + self.controller_name + "." + self.map_name + ".csv"
                self.df_lap_times.to_csv(file_name, sep=',', index=False)

                rospy.loginfo(f"Race has ended!")
                self.race_complete = True
            # else we keep waiting until the race has ended

        self.pose_previous = self.pose_current

    def mux_callback(self, multiarray: Int32MultiArray) -> None:
        """
            Sets controller name to index of first value that is enabled
            Check params.yaml to see which index corresponds to which controller
        """
        ls = multiarray.data
        self.controller_name = ls.index(1)  # Index of first val of "1" (Enabled)

    def race_ended(self, error: float = 0.5, min_lap_time_ns: float = 1000) -> bool:
        """
            Lap has ended if:
                Current position is (0, 0) (Within `error`)
                Is moving (Previous pose != Current pose)
                If `min_lap_time` has elapsed since race start
        """
        return self.pose_previous is not None and \
               float_equal_double(self.curr_x, 0, self.curr_y, 0, error=error) and \
               self.is_moving(error=error) and \
               self.curr_time - self.race_start_time > min_lap_time_ns

    def is_moving(self, error: float = 0.00003) -> bool:
        return not self.is_not_moving(error=error)

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


def main(args: list) -> None:
    rospy.init_node("laptimer", anonymous=True)

    map_uri = args[1]
    map_uri = os.path.split(map_uri)
    map_name = map_uri[-1].replace("map_yaml", "")

    LapTimer(map_name)

    rospy.spin()


if __name__ == '__main__':
    print("LapTimer running")
    try:
        main(sys.argv)
    except rospy.ROSInterruptException:
        pass
