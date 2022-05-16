from math import isnan, isinf, degrees

import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan


class WallFollowing:
    SCAN_TOPIC = "scan"
    DRIVE_TOPIC = "wf_drive"

    def __init__(self):
        self.scan_subscriber = rospy.Subscriber(self.SCAN_TOPIC, LaserScan, self.lidar_callback, queue_size=1)
        self.drive_publisher = rospy.Publisher(self.DRIVE_TOPIC, AckermannDriveStamped, queue_size=10)

    def lidar_callback(self, laser_scan: LaserScan) -> None:
        angle = 55
        return
        left_dist = float(getRange(data, 270))
        proc_ranges = self.get_distance(laser_scan.ranges, angle)



        error = self.followLeft()
        # send error to pid_control
        self.pid_control(error, VELOCITY)
        return

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

    def pid_control(self, error, velocity):
        global integral
        global prev_error
        global kp
        global ki
        global kd
        angle = 0.0
        # TODO: Use kp, ki & kd to implement a PID controller for
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = rospy.Time.now()
        drive_msg.header.frame_id = "laser"
        drive_msg.drive.steering_angle = angle
        drive_msg.drive.speed = velocity
        self.drive_pub.publish(drive_msg)

    def followLeft(self, data, leftDist):
        # Follow left wall as per the algorithm
        # TODO:implement
        return 0.0


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