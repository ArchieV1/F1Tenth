#!/usr/bin/env python
import rospy


def main() -> None:
    rospy.init_node("scanmatching", anonymous=True)
    rospy.spin()


if __name__ == '__main__':
    try:
        rospy.loginfo(f"{__file__} initialised")
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo(f"{__file__} failed to run")
        pass
