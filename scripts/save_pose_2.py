#!/usr/bin/env python

import yaml
import rospy
import os
from geometry_msgs.msg import PoseWithCovarianceStamped

POSE_FILE = "/home/lcastor/ros_ws/src/LCASTOR/examples/goal.yaml"
captured_pose = None

def callback(data):
    global captured_pose
    if captured_pose is None:
        # Convert pose to the array format [x, y, z, qx, qy, qz, qw]
        captured_pose = [
            data.pose.pose.position.x,
            data.pose.pose.position.y,
            data.pose.pose.position.z,
            data.pose.pose.orientation.x,
            data.pose.pose.orientation.y,
            data.pose.pose.orientation.z,
            data.pose.pose.orientation.w
        ]
        rospy.signal_shutdown("Pose captured")

def save_pose(room_name):
    global captured_pose
    
    rospy.init_node('amcl_pose_saver', anonymous=True)
    rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, callback)

    print(f"[INFO] Waiting for AMCL pose to be published...")
    rospy.spin()

    # Load existing poses
    poses = {}
    if os.path.exists(POSE_FILE):
        with open(POSE_FILE, 'r') as f:
            content = yaml.safe_load(f)
            if content and 'room_dict_b' in content:
                poses = content['room_dict_b']
            elif content:
                poses = content

    # Add the new pose
    poses[room_name] = captured_pose

    # Save in the format shown in your example
    output_data = {'room_dict_b': poses}

    with open(POSE_FILE, 'w') as f:
        yaml.dump(output_data, f, default_flow_style=False)

    print(f"[SUCCESS] Pose for '{room_name}' saved to {POSE_FILE}")
    print(f"[INFO] Saved pose: {captured_pose}")

if __name__ == "__main__":
    try:
        room = input("Enter room name to save pose: ").strip()
        if room:
            save_pose(room)
        else:
            print("[ERROR] Room name is empty.")
    except rospy.ROSInterruptException:
        print("[INFO] ROS node interrupted")
    except KeyboardInterrupt:
        print("\n[INFO] Script interrupted by user")