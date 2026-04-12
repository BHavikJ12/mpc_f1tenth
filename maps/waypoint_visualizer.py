import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import csv

class WaypointVisualizer(Node):
    def __init__(self):
        super().__init__('waypoint_visualizer')

        self.publisher = self.create_publisher(Marker, 'visualization_marker', 10)

        self.timer = self.create_timer(1.0, self.publish_points)

        self.points = self.load_csv('amcl_poses.csv')  # <-- change path

    def load_csv(self, file_path):
        points = []
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                try:
                    x = float(row[1])
                    y = float(row[2])
                    points.append((x, y))
                except:
                    continue
        return points

    def publish_points(self):
        marker = Marker()
        marker.header.frame_id = "map"   # must match your TF
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = "waypoints"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD

        marker.scale.x = 0.15
        marker.scale.y = 0.15

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        for x, y in self.points:
            p = Point()
            p.x = x
            p.y = y
            p.z = 0.0
            marker.points.append(p)

        self.publisher.publish(marker)


def main():
    rclpy.init()
    node = WaypointVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()