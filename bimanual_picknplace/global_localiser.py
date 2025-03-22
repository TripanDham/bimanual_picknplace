# my_static_transform_publisher/static_transform_publisher.py

import rclpy
from rclpy.node import Node
import tf2_ros
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Quaternion, Pose
import tf_transformations
import numpy as np


class StaticTransformPublisher(Node):
    def __init__(self):
        super().__init__('global_localiser')

        # Create a tf2 broadcaster to publish the transform
        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        self.a_goal_pub = self.create_publisher(Pose, '/a/goal', 10)
        self.b_goal_pub = self.create_publisher(Pose, '/b/goal', 10)

        # Create the static transforms
        self.create_static_transform('odom', 'a_odom', 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        self.create_static_transform('odom', 'b_odom', 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)

        a_goal = Pose()
        a_goal.position.x = 5.0
        a_goal.position.y = 1.0
        a_goal.position.z = 1.0
        pi = np.pi
        ax = pi/2
        ay = 0
        az = pi/4
        quat = tf_transformations.quaternion_from_euler(ax,ay,az)
        a_goal.orientation.x = quat[0]
        a_goal.orientation.y = quat[1]
        a_goal.orientation.z = quat[2]
        a_goal.orientation.w = quat[3]

        self.a_goal_pub.publish(a_goal)

        b_goal = Pose()
        b_goal.position.x = 5.0
        b_goal.position.y = 1.0
        b_goal.position.z = 1.0
        pi = np.pi
        ax = pi/2
        ay = 0
        az = pi/4
        quat = tf_transformations.quaternion_from_euler(ax,ay,az)
        b_goal.orientation.x = quat[0]
        b_goal.orientation.y = quat[1]
        b_goal.orientation.z = quat[2]
        b_goal.orientation.w = quat[3]
        
        self.b_goal_pub.publish(b_goal)

    def create_static_transform(self, parent_frame, child_frame, x, y, z, qx, qy, qz, qw):
        transform = TransformStamped()

        # Fill in the transform details
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = parent_frame
        transform.child_frame_id = child_frame

        transform.transform.translation = Vector3(x=x, y=y, z=z)
        transform.transform.rotation = Quaternion(x=qx, y=qy, z=qz, w=qw)

        # Broadcast the static transform
        self.tf_broadcaster.sendTransform(transform)


def main(args=None):
    rclpy.init(args=args)

    static_transform_publisher = StaticTransformPublisher()

    rclpy.spin(static_transform_publisher)

    static_transform_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
