import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
import rclpy.logging
import rclpy.time

from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, TransformStamped, Pose
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
import numpy as np
import matplotlib.pyplot as plt
import tf_transformations

from control_msgs.action import FollowJointTrajectory
import time
import hello_helpers.hello_misc as hm
import os

os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms/libqxcb.so'

class DrivePublisherNode(hm.HelloNode):
    def __init__(self):
        hm.HelloNode.__init__(self)
        hm.HelloNode.main(self, 'stt_kinematics','stt_kinematics',wait_for_first_pointcloud=False)
        self.logger = self.get_logger()
        self.callback_group = ReentrantCallbackGroup()

        self.goal_set = []
        self.create_subscription(Pose, "/b/goal", self.goal_callback)

        self.vel_pub = self.create_publisher(Twist, "/b/velocity", 10)

        self.localize()

        # x = self.pose[0]
        # y = self.pose[1]
        # z = self.pose[2]
        # ax = self.pose[3]
        # ay = self.pose[4]
        # az = self.pose[5]

        self.tube_thicknesses = [0.1, 0.1, 0.05, 0.01, 0.01, 0.01]

        self.start_set = np.array([[self.pose[i] - self.tube_thicknesses[i], self.pose[i] + self.tube_thicknesses[i]] for i in range(0,6)])

        self.t_final = 20

        self.create_tube()
        
        self.start_time = round(self.get_clock().now().nanoseconds/1e9, 4)
        self.timer = self.create_timer(0.1, self.timer_callback, callback_group = self.callback_group)  # Publish every 0.01 seconds

        self.ulim = 0.3
        self.rlim = 0.4
        self.vlift_lim = 0.1
        self.varm_lim = 0.05

        self.rhod_0 = 1.0
        self.rhoo_0 = 1.0
        
        self.decay = 0.01

        self.rhod_inf = 0.01
        self.rhod_lower = -0.1
        
        self.rhoo_inf = 0.01

        self.rholift_0 = 0.2
        self.rholift_inf = 0.01
        self.rhoarm_0 = 0.05
        self.rhoarm_inf = 0.001

        self.data = {}

        self.count = 0

    def goal_callback(self, goal_pose):
        x = goal_pose.position.x 
        y = goal_pose.position.y
        z = goal_pose.position.z = 1.0
        qx = goal_pose.orientation.x 
        qy = goal_pose.orientation.y 
        qz = goal_pose.orientation.z 
        qw = goal_pose.orientation.w 

        ax, ay, az = tf_transformations.euler_from_quaternion([qx,qy,qz,qw])

        goal = [x,y,z,ax,ay,az]

        self.goal_set = np.array([[self.pose[i] - self.tube_thicknesses[i], self.pose[i] + self.tube_thicknesses[i]] for i in range(0,6)])

    def localize(self):
        x_base = self.base_pose[0]
        y_base = self.base_pose[1]
        theta = self.base_pose[2]

        base_arm_transform = TransformStamped()
        
        try:
            base_arm_transform = self.tf2_buffer.lookup_transform('b_link_aruco_top_wrist', 'b_base_link', rclpy.time.Time())
            delX = base_arm_transform.transform.translation.x
            delY = base_arm_transform.transform.translation.y
            delZ = base_arm_transform.transform.translation.z
        except:
            delX = 0
            delY = 0
            delZ = 0
            self.get_logger().info("Could not recieve TF base to wrist")

        x = x_base + delX
        y = y_base + delY
        z = delZ

        self.pose = np.array([x_base + delX, y_base + delY, delZ, theta])
        return self.pose

    def timer_callback(self):
        
        self.localize()

        x = self.pose[0]
        y = self.pose[1]
        z = self.pose[2]
        ax = self.pose[3]
        ay = self.pose[4]
        az = self.pose[5]

        t = round(self.get_clock().now().nanoseconds/1e9, 4) - self.start_time
        
        rhol, rhou = self.tube(self.start_set, self.goal_set, t, self.t_final)
        
        names = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
        rho = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        rhoinf = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        lims = [0.5, 0.5, 0.3, 0.1, 0.1, 0.1]

        e = [((rhol[i] + rhou[i])/2 - self.pose[i])/(rhou[i] - rhol[i]) for i in range(6)]

        vel = [self.funnel(e[i], t, names[i], lims[i], rho[i], rhoinf[i]) for i in range(6)]

        pub_vel = Twist()

        pub_vel.linear.x = vel[0]
        pub_vel.linear.y = vel[1]
        pub_vel.linear.z = vel[2]
        pub_vel.angular.x = vel[3]
        pub_vel.angular.y = vel[4]
        pub_vel.angular.z = vel[5]

        self.vel_pub.publish(pub_vel)

    def transform(self, x):
        a = 2
        return (1-np.exp(-(a*x)**2)) * np.tanh(a*x)
    
    def funnel(self, e, t, name, vlim, rho = 0.5, rho_inf = 0.01, decay = 0.01):
        rho_t = rho_inf + (rho - rho_inf) * np.exp(-decay * t)
        Xi = e / rho_t
        eps = self.transform(Xi) * vlim
        v = eps

        if self.plot_on:
            if name not in self.data:
                self.data[name] = {
                    'rho_t': [],
                    'v': [],
                    'e': []
                }
            
            self.data[name]['rho_t'].append(rho_t)
            self.data[name]['v'].append(v)
            self.data[name]['e'].append(e)
        return v

    def tube(self, start_set, goal_set, t, t_final):
        rhou = np.zeros(6)
        rhol = np.zeros(6)
        for i in range(6):
            start_lower, start_upper = start_set[i]
            goal_lower, goal_upper = goal_set[i]
            tanh_term = np.tanh(t / (t_final - t))
            
            rhou[i] = start_upper + (goal_upper - start_upper) * tanh_term
            rhol[i] = start_lower + (goal_lower - start_lower) * tanh_term

        return rhol, rhou

    def get_ref_pose(self,t):
        gamL_arr = [self.stt_val[i,:] for i in range(1,8,2)]
        gamU_arr = [self.stt_val[i,:] for i in range(2,9,2)]

        time_arr = self.stt_val[0,:]

        tube_lower = np.array([np.interp(t, time_arr, x) for x in gamL_arr])
        tube_upper = np.array([np.interp(t, time_arr, x) for x in gamU_arr])

        return tube_lower, tube_upper
    
    def zero_vel(self):
        v_arm = 0.0
        v_lift = 0.0
        arm_speed = {'joint_arm_l0': v_arm, 'joint_arm_l1': v_arm, 'joint_arm_l2': v_arm, 'joint_arm_l3': v_arm}
        arm_speed['joint_lift'] = v_lift

        point = JointTrajectoryPoint()
        point.time_from_start = Duration(seconds=0.0).to_msg()

        point.velocities = [joint_velocity for joint_velocity in arm_speed.values()]
        joint_names = [key for key in arm_speed]

        trajectory_goal = FollowJointTrajectory.Goal()
        trajectory_goal.goal_time_tolerance = Duration(seconds=1.0).to_msg()
        trajectory_goal.trajectory.joint_names = joint_names
        trajectory_goal.trajectory.points = [point]
        self.trajectory_client.send_goal_async(trajectory_goal)    

def main(args=None):
    # rclpy.init(args=args)
    node = DrivePublisherNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node stopped by user.")
        node.zero_vel()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
