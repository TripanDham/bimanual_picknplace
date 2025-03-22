import numpy as np
import rclpy 
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, TransformStamped
import tf2_ros

import tf2_geometry_msgs
import csv
import tf_transformations

import hello_helpers.hello_misc as hm
import os
import time

os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms/libqxcb.so'

class WeightedController(hm.HelloNode):
    def __init__(self):
        hm.HelloNode.__init__(self)
        hm.HelloNode.main(self, 'weighted_controller','weighted_controller',wait_for_first_pointcloud=False)
        
        self.callback_group = ReentrantCallbackGroup()
        self.base_pub = self.create_publisher(Twist, '/b/stretch/cmd_vel', 10, callback_group=self.callback_group)
        
        self.global_localisation = 1  #1: transform from odom to grasp center used, 0: transform from b_odom used

        time.sleep(2)
        self.localize()

    def localize(self):
        global_transform = TransformStamped()
        if self.global_localisation == 1:
            parent_frame = 'odom'
        else:
            parent_frame = 'b_odom'
        
        try:
            global_transform = self.tf2_buffer.lookup_transform('b_link_grasp_center', parent_frame, rclpy.time.Time())
        except:
            self.get_logger().info("Could not recieve TF")

        x = global_transform.transform.translation.x
        y = global_transform.transform.translation.y
        z = global_transform.transform.translation.z
        
        orientation = global_transform.transform.rotation
        quat = [orientation.x, orientation.y, orientation.z, orientation.w]
        ax, ay, az = tf_transformations.euler_from_quaternion(quat)
        
        self.pose = [x,y,z,ax,ay,az]

        return self.pose

        
