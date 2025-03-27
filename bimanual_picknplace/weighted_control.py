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
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from rclpy.duration import Duration

os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms/libqxcb.so'

class WeightedController(hm.HelloNode):
    def __init__(self):
        hm.HelloNode.__init__(self)
        hm.HelloNode.main(self, 'weighted_controller','weighted_controller',wait_for_first_pointcloud=False)
        
        self.callback_group = ReentrantCallbackGroup()
        self.base_pub = self.create_publisher(Twist, '/stretch/cmd_vel', 10, callback_group=self.callback_group)

        self.odom_sub = self.create_subscription(Odometry, '/b/odom', self.odom_callback, 1, callback_group=self.callback_group)
        self.tube_vel_sub = self.create_subscription(Twist, '/b/velocity', self.vel_cb, 1, callback_group=self.callback_group)

        time.sleep(2)

    def odom_callback(self, odom_msg):
        orientation = odom_msg.pose.pose.orientation
        self.tb = tf_transformations.euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])[2]
        
        self.tr = self.joint_state['b_joint_wrist_roll']
        self.ty = self.joint_state['b_joint_wrist_yaw']
        self.tp = self.joint_state['b_joint_wrist_pitch']
        self.da = self.joint_state['b_joint_arm_l0'] * 4
        self.x_base = odom_msg.pose.pose.position.x
        self.y_base = odom_msg.pose.pose.position.y
        self.dl = self.joint_state['b_joint_lift']

    def weighted_control(self, v_ee):
        tb = self.tb
        tr = self.tr
        ty = self.ty
        tp = self.tp
        da = self.da
        j = np.array([[np.cos(tb), 0.1687*np.cos(tb) + 0.017*np.sin(tb) - 0.251*np.cos(tr)*np.cos(tb + ty) - 1.0*np.cos(tb)*da - 0.251*np.cos(tp)*np.sin(tr)*np.sin(tb + ty),   0, -1.0*np.sin(tb), - 0.251*np.cos(tr)*np.cos(tb + ty) - 0.251*np.cos(tp)*np.sin(tr)*np.sin(tb + ty), -0.251*np.sin(tp)*np.sin(tr)*np.cos(tb + ty), 0.251*np.sin(tr)*np.sin(tb + ty) + 0.251*np.cos(tp)*np.cos(tr)*np.cos(tb + ty)],
                      [np.sin(tb), 0.1687*np.sin(tb) - 0.017*np.cos(tb) - 0.251*np.cos(tr)*np.sin(tb + ty) - 1.0*np.sin(tb)*da + 0.251*np.cos(tp)*np.sin(tr)*np.cos(tb + ty),   0,      np.cos(tb),   0.251*np.cos(tp)*np.sin(tr)*np.cos(tb + ty) - 0.251*np.cos(tr)*np.sin(tb + ty), -0.251*np.sin(tp)*np.sin(tr)*np.sin(tb + ty), 0.251*np.cos(tp)*np.cos(tr)*np.sin(tb + ty) - 0.251*np.sin(tr)*np.cos(tb + ty)],
                      [0,0, 1.0, 0,  0,  0.251*np.cos(tp)*np.sin(tr), 0.251*np.cos(tr)*np.sin(tp)],
                      [0,0,  0,  0,0, np.sin(tb(1) + ty(1)),np.cos(tb(1) + ty(1))*np.sin(tp(1))],
                      [0, 0, 0, 0,0,-1.0*np.cos(tb(1) + ty(1)),np.sin(tb(1) + ty(1))*np.sin(tp(1))],
                      [0, 1.0,   0,0,1.0,0,-1.0*np.cos(tp(1))]])
        
        w_arm = 1
        w_base = 1
        w_wrist = 1

        w = np.diag([(w_arm + w_base)/2, (w_arm + w_base)/2, w_arm, w_arm, w_wrist, w_wrist, w_wrist])

        j_inv = w @ np.transpose(j) @ np.linalg.inv(j @ w @ np.transpose(j))
        v_robot = j_inv @ v_ee

        return v_robot
    
    def vel_cb(self, vel):
        vx = vel.linear.x
        vy = vel.linear.y
        vz = vel.linear.z

        wx = vel.angular.x
        wy = vel.angular.y
        wz = vel.angular.z

        vel = np.array([vx, vy, vz, wx, wy, wz])

        v_robot = self.weighted_control(vel)
        self.send_vel(v_robot)

    def send_vel(self, v_robot):
        robot_joints = ['joint_lift', 'joint_arm_l0', 'joint_arm_l1', 'joint_arm_l2', 'joint_arm_l3', 'joint_wrist_yaw', 'joint_wrist_pitch', 'joint_wrist_roll']
        robot_velocities = [v_robot[2], v_robot[3], v_robot[3], v_robot[3], v_robot[3], v_robot[4], v_robot[5], v_robot[6]]

        point = JointTrajectoryPoint()
        point.time_from_start = Duration(seconds=0.0).to_msg()

        point.velocities = robot_velocities
        joint_names = robot_joints

        trajectory_goal = FollowJointTrajectory.Goal()
        trajectory_goal.goal_time_tolerance = Duration(seconds=1.0).to_msg()
        trajectory_goal.trajectory.joint_names = joint_names
        trajectory_goal.trajectory.points = [point]
        self.trajectory_client.send_goal_async(trajectory_goal)

        v_base = Twist()

        v_base.linear.x = v_robot[0]
        v_base.angular.z = v_robot[1]
        self.base_pub.publish()