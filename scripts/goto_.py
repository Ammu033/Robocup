#!/usr/bin/env python3

import rospy
import sys
import math
import actionlib
import rospkg
import yaml
import os
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus

class GotoNode:
    def __init__(self):
        rospy.init_node('goto_node', anonymous=True)
        
        # Connect to move_base action server
        self.client = actionlib.SimpleActionClient('/move_base', MoveBaseAction)
        rospy.loginfo("Connecting to /move_base action server...")
        
        if not self.client.wait_for_server(timeout=rospy.Duration(10.0)):
            rospy.logerr("Failed to connect to /move_base action server")
            sys.exit(1)
        
        rospy.loginfo("Connected to /move_base action server")
        
        # Load room positions from YAML
        self.room_dict = self.load_room_positions()
        
    def load_room_positions(self):
        """Load room positions from YAML file"""
        try:
            # Try multiple possible paths for the YAML file
            possible_paths = [
                '/home/lcastor/ros_ws/src/LCASTOR/examples/goal.yaml',
                '/home/lcastor/ros_ws/src/LCASTOR/tiago_auto/config/room_positions.yaml',
                os.path.expanduser('~/ros_ws/src/LCASTOR/examples/goal.yaml'),
            ]
            
            yaml_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    yaml_path = path
                    break
            
            if yaml_path is None:
                rospy.logwarn("YAML file not found, using hardcoded positions")
                return self.get_hardcoded_positions()
            
            rospy.loginfo(f"Loading room positions from: {yaml_path}")
            with open(yaml_path, 'r') as file:
                data = yaml.safe_load(file)
                rospy.loginfo(f"Loaded {len(data)} room positions from YAML")
                return data
                
        except Exception as e:
            rospy.logwarn(f"Error loading YAML: {e}, using hardcoded positions")
            return self.get_hardcoded_positions()
    
    def get_hardcoded_positions(self):
        """Fallback hardcoded room positions"""
        return { 
            "hallwaycabinet": {
                "position": {"x": -2.93, "y": -4.47, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": -0.08, "w": 0.99}
            },
            "hallway": {
                "position": {"x": -1.41, "y": -2.64, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": -0.30, "w": 0.95}
            },
            "entrance": {
                "position": {"x": -1.41, "y": -2.64, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": -0.30, "w": 0.95}
            },
            "desk": {
                "position": {"x": -2.79, "y": -0.46, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.61, "w": -0.78}
            },
            "office": {
                "position": {"x": -0.96, "y": -1.21, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.52, "w": 0.85}
            },
            "studio": {
                "position": {"x": -0.96, "y": -1.21, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.52, "w": 0.85}
            },
            "shelf": {
                "position": {"x": -3.03, "y": 1.55, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.03, "w": 0.99}
            },
            "coathanger": {
                "position": {"x": -1.73, "y": -2.89, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.44, "w": 0.89}
            },
            "exit": {
                "position": {"x": -0.99, "y": 1.61, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.73, "w": 0.67}
            },
            "TVtable": {
                "position": {"x": 1.01, "y": -4.53, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": -0.05, "w": 0.99}
            },
            "loungechair": {
                "position": {"x": 1.64, "y": -4.85, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.99, "w": -0.09}
            },
            "lamp": {
                "position": {"x": 3.26, "y": -5.12, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.99, "w": -0.12}
            },
            "couch": {
                "position": {"x": 3.6, "y": -2.60, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.93, "w": -0.35}
            },
            "coffetable": {
                "position": {"x": 2.45, "y": -3.20, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.83, "w": -0.55}
            },
            "lounge": {
                "position": {"x": 2.72, "y": -1.96, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.73, "w": -0.67}
            },
            "livingroom": {
                "position": {"x": 2.72, "y": -1.96, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.73, "w": -0.67}
            },
            "trashcan": {
                "position": {"x": 0.58, "y": -1.16, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": -0.17, "w": 0.98}
            },
            "kitchen": {
                "position": {"x": 3.34, "y": -1.76, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.54, "w": 0.84}
            },
            "kitchencabinet": {
                "position": {"x": 0.62, "y": 2.29, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.03, "w": 0.99}
            },
            "dinnertable": {
                "position": {"x": 1.44, "y": 1.28, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.99, "w": -0.02}
            },
            "dishwasher": {
                "position": {"x": 3.67, "y": 0.73, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.99, "w": 0.04}
            },
            "kitchencounter": {
                "position": {"x": 3.80, "y": 1.98, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.99, "w": 0.0}
            },
            "inspectionpoint": {
                "position": {"x": 0.19, "y": -2.69, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.87, "w": -0.48}
            },
            "findTrashEntrance": {
                "position": {"x": -0.61, "y": 5.89, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.99, "w": 0.05}
            },
            "findTrashOffice": {
                "position": {"x": 0.30, "y": 4.63, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": -0.41, "w": 0.91}
            },
            "findTrashKitchen1": {
                "position": {"x": -3.18, "y": 7.15, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": -0.24, "w": 0.97}
            },
            "findTrashKitchen2": {
                "position": {"x": -5.74, "y": 10.44, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.57, "w": -0.82}
            },
            "findTrashLivingRoom": {
                "position": {"x": -5.74, "y": 10.40, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.99, "w": -0.07}
            },
        }
    
    def goto_coordinates(self, x, y, theta):
        """
        Navigate to specific coordinates in map frame
        Args:
            x (float): X coordinate in map frame
            y (float): Y coordinate in map frame  
            theta (float): Orientation in radians
        """
        rospy.loginfo(f"Navigating to coordinates: x={x}, y={y}, theta={theta}")
        
        # Create goal message
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        
        # Set position
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.position.z = 0.0
        
        # Set orientation (convert theta to quaternion)
        goal.target_pose.pose.orientation.x = 0.0
        goal.target_pose.pose.orientation.y = 0.0
        goal.target_pose.pose.orientation.z = math.sin(theta / 2.0)
        goal.target_pose.pose.orientation.w = math.cos(theta / 2.0)
        
        # Send goal
        self.client.send_goal(goal)
        rospy.loginfo("Goal sent, waiting for result...")
        
        # Wait for result
        success = self.client.wait_for_result(timeout=rospy.Duration(60.0))
        
        if success:
            state = self.client.get_state()
            if state == GoalStatus.SUCCEEDED:
                rospy.loginfo("Navigation successful!")
                return True
            else:
                # Map state numbers to readable names
                state_names = {
                    0: "PENDING",
                    1: "ACTIVE", 
                    2: "PREEMPTED",
                    3: "SUCCEEDED",
                    4: "ABORTED",
                    5: "REJECTED",
                    6: "PREEMPTING",
                    7: "RECALLING",
                    8: "RECALLED",
                    9: "LOST"
                }
                state_name = state_names.get(state, f"UNKNOWN_{state}")
                rospy.logwarn(f"Navigation failed with state: {state} ({state_name})")
                
                # Get detailed result if available
                result = self.client.get_result()
                if result:
                    rospy.logwarn(f"Navigation result: {result}")
                
                return False
        else:
            rospy.logwarn("Navigation timed out")
            self.client.cancel_goal()
            return False
    
    def goto_pose_dict(self, pose_dict):
        """
        Navigate using pose dictionary from YAML
        Args:
            pose_dict (dict): Dictionary with position and orientation keys
        """
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

        rospy.loginfo(f"Sending navigation goal...")
        self.client.send_goal(goal)
        
        # Wait for result
        success = self.client.wait_for_result(timeout=rospy.Duration(60.0))
        
        if success:
            state = self.client.get_state()
            if state == GoalStatus.SUCCEEDED:
                rospy.loginfo("✅ Navigation goal reached!")
                return True
            else:
                state_names = {
                    0: "PENDING", 1: "ACTIVE", 2: "PREEMPTED", 3: "SUCCEEDED",
                    4: "ABORTED", 5: "REJECTED", 6: "PREEMPTING", 7: "RECALLING",
                    8: "RECALLED", 9: "LOST"
                }
                state_name = state_names.get(state, f"UNKNOWN_{state}")
                rospy.logwarn(f"❌ Navigation failed with state: {state} ({state_name})")
                return False
        else:
            rospy.logwarn("❌ Navigation timed out")
            self.client.cancel_goal()
            return False
    
    def goto_coordinates_with_retry(self, x, y, theta, max_retries=3):
        """
        Navigate to coordinates with retry logic and nearby goal fallback
        """
        for attempt in range(max_retries):
            rospy.loginfo(f"Navigation attempt {attempt + 1}/{max_retries}")
            
            if attempt == 0:
                # First attempt: exact coordinates
                success = self.goto_coordinates(x, y, theta)
            else:
                # Subsequent attempts: try nearby positions
                offset = 0.5 * attempt  # Increase offset each attempt
                nearby_positions = [
                    (x + offset, y, theta),
                    (x - offset, y, theta), 
                    (x, y + offset, theta),
                    (x, y - offset, theta),
                    (x + offset/2, y + offset/2, theta),
                    (x - offset/2, y - offset/2, theta)
                ]
                
                success = False
                for nx, ny, nt in nearby_positions:
                    rospy.loginfo(f"Trying nearby position: x={nx:.2f}, y={ny:.2f}")
                    if self.goto_coordinates(nx, ny, nt):
                        success = True
                        break
            
            if success:
                return True
            
            if attempt < max_retries - 1:
                rospy.logwarn(f"Attempt {attempt + 1} failed, retrying...")
                rospy.sleep(2)
        
        rospy.logerr(f"All {max_retries} navigation attempts failed")
        return False
    
    def goto_room(self, room_name):
        """
        Navigate to predefined room location from YAML or hardcoded dictionary
        Args:
            room_name (str): Name of the room
        """
        # Remove 'r_' prefix if present
        if room_name.startswith('r_'):
            room_name = room_name[2:]
        
        if room_name not in self.room_dict:
            rospy.logerr(f"Room '{room_name}' not found in room dictionary")
            available_rooms = list(self.room_dict.keys())
            rospy.loginfo(f"Available rooms: {available_rooms}")
            return False
        
        rospy.loginfo(f"Going to room '{room_name}'")
        pose_dict = self.room_dict[room_name]
        
        return self.goto_pose_dict(pose_dict)

def print_usage():
    print("Usage:")
    print("  rosrun tiago_auto goto_node.py coordinates <x> <y> <theta>")
    print("  rosrun tiago_auto goto_node.py room <room_name>")
    print("")
    print("Examples:")
    print("  rosrun tiago_auto goto_node.py coordinates 2.5 1.0 1.57")
    print("  rosrun tiago_auto goto_node.py room kitchen")
    print("  rosrun tiago_auto goto_node.py room r_trashcan")

def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    try:
        goto_node = GotoNode()
        
        command = sys.argv[1].lower()
        
        if command == "coordinates":
            if len(sys.argv) != 5:
                rospy.logerr("coordinates command requires exactly 3 arguments: x y theta")
                print_usage()
                sys.exit(1)
            
            x = float(sys.argv[2])
            y = float(sys.argv[3])
            theta = float(sys.argv[4])
            
            success = goto_node.goto_coordinates_with_retry(x, y, theta)
            
        elif command == "room":
            if len(sys.argv) != 3:
                rospy.logerr("room command requires exactly 1 argument: room_name")
                print_usage()
                sys.exit(1)
            
            room_name = sys.argv[2]
            success = goto_node.goto_room(room_name)
            
        else:
            rospy.logerr(f"Unknown command: {command}")
            print_usage()
            sys.exit(1)
        
        if success:
            rospy.loginfo("Navigation completed successfully")
            sys.exit(0)
        else:
            rospy.logerr("Navigation failed")
            sys.exit(1)
            
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation interrupted")
        sys.exit(1)
    except Exception as e:
        rospy.logerr(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()