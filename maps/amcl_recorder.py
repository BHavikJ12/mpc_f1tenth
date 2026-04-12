#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
import csv
import os
from math import atan2

class AmclPoseLogger(Node):

    def __init__(self):
        super().__init__('amcl_pose_logger')

        self.subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.listener_callback,
            10
        )

        file_path = '/home/bhavik/amcl_poses.csv'  # <-- change this

        file_exists = os.path.isfile(file_path)
        self.csv_file = open(file_path, 'a', newline='')
        self.writer = csv.writer(self.csv_file)

        if not file_exists:
            self.writer.writerow(['timestamp', 'x', 'y', 'yaw'])

        self.get_logger().info(f"Logging AMCL poses to {file_path}")

    def quaternion_to_yaw(self, q):
        # yaw (z-axis rotation)
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return atan2(siny_cosp, cosy_cosp)

    def listener_callback(self, msg):
        pose = msg.pose.pose

        x = pose.position.x
        y = pose.position.y
        yaw = self.quaternion_to_yaw(pose.orientation)

        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        self.writer.writerow([timestamp, x, y, yaw])
        self.csv_file.flush()  # ensure immediate write

        self.get_logger().info(f"x: {x:.2f}, y: {y:.2f}, yaw: {yaw:.2f}")

    def destroy_node(self):
        self.csv_file.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = AmclPoseLogger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()