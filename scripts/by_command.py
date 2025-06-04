#!/usr/bin/env python
# sd_goal.py - Send Goal Action

import rospy
import yaml
import actionlib
from geometry_msgs.msg import PoseStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import sys

def load_pose_from_yaml(file_path, room_name):
    """Load pose from YAML file"""
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
            if room_name not in data:
                rospy.logerr(f"Room '{room_name}' not found in YAML file.")
                return None
            return data[room_name]
    except Exception as e:
        rospy.logerr(f"Error loading YAML: {e}")
        return None

def send_goal(pose_dict):
    """Send navigation goal to move_base"""
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()

    # Set position
    goal.target_pose.pose.position.x = pose_dict['position']['x']
    goal.target_pose.pose.position.y = pose_dict['position']['y']
    goal.target_pose.pose.position.z = pose_dict['position']['z']

    # Set orientation
    goal.target_pose.pose.orientation.x = pose_dict['orientation']['x']
    goal.target_pose.pose.orientation.y = pose_dict['orientation']['y']
    goal.target_pose.pose.orientation.z = pose_dict['orientation']['z']
    goal.target_pose.pose.orientation.w = pose_dict['orientation']['w']

    # Connect to move_base action server
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    rospy.loginfo("Waiting for move_base action server...")
    client.wait_for_server()

    rospy.loginfo(f"Sending navigation goal...")
    client.send_goal(goal)
    client.wait_for_result()

    if client.get_state() == actionlib.GoalStatus.SUCCEEDED:
        rospy.loginfo("✅ Navigation goal reached!")
        return True
    else:
        rospy.logwarn("❌ Failed to reach navigation goal.")
        return False

if __name__ == '__main__':
    try:
        rospy.init_node('sd_goal_action')
        
        # Path to your poses.yaml file
        yaml_path = '/home/lcastor/ros_ws/src/LCASTOR/examples/goal.yaml'
        
        # Get room name from input (from Brain or command line)
        if len(sys.argv) > 1:
            room = sys.argv[1]
        else:
            room = input().strip()  # Get from stdin (from Brain subprocess)
        
        rospy.loginfo(f"sd_goal.py: Navigating to '{room}'")
        
        # Load pose and navigate
        pose = load_pose_from_yaml(yaml_path, room)
        if pose:
            success = send_goal(pose)
            if success:
                rospy.loginfo(f"✅ sd_goal.py completed successfully")
                sys.exit(0)
            else:
                rospy.logerr(f"❌ sd_goal.py navigation failed")
                sys.exit(1)
        else:
            rospy.logerr(f"❌ sd_goal.py: Room '{room}' not found")
            sys.exit(1)
            
    except Exception as e:
        rospy.logerr(f"❌ sd_goal.py error: {e}")
        sys.exit(1)