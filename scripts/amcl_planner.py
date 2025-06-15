#!/usr/bin/env python3

import rospy
import numpy as np
import math
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Point, Quaternion, Twist
from std_srvs.srv import Trigger, TriggerResponse
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from actionlib_msgs.msg import GoalStatusArray
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from threading import Lock
from enum import Enum
import time

class Direction(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

class CoverageState(Enum):
    IDLE = 0
    MOVING_NORTH = 1
    MOVING_SOUTH = 2
    MOVING_EAST = 3
    MOVING_WEST = 4
    COMPLETED = 5

class AMCLIntelligentCoveragePlanner:
    def __init__(self):
        rospy.init_node('amcl_intelligent_coverage_planner', anonymous=True)
        
        # --- AMCL-Based Parameters ---
        self.step_size = rospy.get_param('~step_size', 0.4)  # 40cm steps
        self.max_distance_from_start = rospy.get_param('~max_distance_from_start', 8.0)  # 8m limit
        self.max_line_length = rospy.get_param('~max_line_length', 6.0)  # Max line before turning
        self.robot_radius = rospy.get_param('~robot_radius', 0.3)
        self.safety_margin = rospy.get_param('~safety_margin', 0.2)
        self.goal_tolerance = rospy.get_param('~goal_tolerance', 0.15)
        self.max_goal_timeout = rospy.get_param('~max_goal_timeout', 15.0)
        
        # State variables
        self.occupancy_map = None
        self.current_pose = None
        self.start_position = None
        self.visited_positions = set()
        self.current_direction = Direction.NORTH
        self.coverage_state = CoverageState.IDLE
        self.is_executing = False
        self.data_lock = Lock()
        
        # Coverage tracking
        self.current_line_length = 0.0
        self.current_goal = None
        self.total_moves = 0
        self.successful_moves = 0
        
        # --- Subscribers ---
        rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
        rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.pose_callback)
        
        # --- Publishers ---
        self.goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)
        self.markers_pub = rospy.Publisher("/coverage_markers", MarkerArray, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        
        # --- Action Client ---
        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base action server...")
        if self.move_base_client.wait_for_server(timeout=rospy.Duration(10.0)):
            rospy.loginfo("Connected to move_base action server")
        else:
            rospy.logwarn("Could not connect to move_base action server")
        
        # --- Services ---
        rospy.Service("/coverage_planner/start_coverage", Trigger, self.start_coverage_service)
        rospy.Service("/coverage_planner/stop_coverage", Trigger, self.stop_coverage_service)
        rospy.Service("/coverage_planner/generate_path", Trigger, self.generate_path_service)
        rospy.Service("/coverage_planner/get_progress", Trigger, self.get_progress_service)
        
        rospy.loginfo("AMCL Intelligent Coverage Planner initialized")
    
    def map_callback(self, msg):
        """Receive the occupancy grid map"""
        with self.data_lock:
            self.occupancy_map = msg
    
    def pose_callback(self, msg):
        """Receive current robot pose from AMCL"""
        with self.data_lock:
            self.current_pose = msg
            if self.start_position is None:
                self.start_position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
                rospy.loginfo(f"Set start position: ({self.start_position[0]:.2f}, {self.start_position[1]:.2f})")
    
    def _world_to_grid(self, world_x, world_y):
        """Convert world coordinates to grid coordinates"""
        if self.occupancy_map is None:
            return None, None
        
        grid_info = self.occupancy_map.info
        grid_x = int((world_x - grid_info.origin.position.x) / grid_info.resolution)
        grid_y = int((world_y - grid_info.origin.position.y) / grid_info.resolution)
        return grid_x, grid_y
    
    def _is_position_safe(self, world_x, world_y):
        """Check if a world position is safe (obstacle-free with safety margin)"""
        if self.occupancy_map is None:
            rospy.logwarn("🚨 No occupancy map available for safety check")
            return False
        
        grid_x, grid_y = self._world_to_grid(world_x, world_y)
        if grid_x is None or grid_y is None:
            rospy.logwarn(f"🚨 Could not convert world coords ({world_x:.2f}, {world_y:.2f}) to grid")
            return False
        
        map_info = self.occupancy_map.info
        
        # Check if grid coordinates are within map bounds
        if (grid_x < 0 or grid_x >= map_info.width or 
            grid_y < 0 or grid_y >= map_info.height):
            rospy.logwarn(f"🚨 Position ({world_x:.2f}, {world_y:.2f}) -> grid ({grid_x}, {grid_y}) is outside map bounds ({map_info.width}x{map_info.height})")
            return False
        
        safety_cells = int(self.safety_margin / map_info.resolution)
        
        # Check area around the position
        for dy in range(-safety_cells, safety_cells + 1):
            for dx in range(-safety_cells, safety_cells + 1):
                check_x = grid_x + dx
                check_y = grid_y + dy
                
                if (check_x < 0 or check_x >= map_info.width or 
                    check_y < 0 or check_y >= map_info.height):
                    continue  # Skip out-of-bounds cells
                
                idx = check_y * map_info.width + check_x
                if idx >= len(self.occupancy_map.data):
                    continue
                
                cell_value = self.occupancy_map.data[idx]
                if cell_value > 50:  # Occupied or unknown
                    return False
        
        return True
    
    def _get_distance_from_start(self, x, y):
        """Calculate distance from start position"""
        if self.start_position is None:
            return float('inf')
        
        dx = x - self.start_position[0]
        dy = y - self.start_position[1]
        return math.sqrt(dx*dx + dy*dy)
    
    def _get_next_position(self, current_x, current_y, direction):
        """Calculate next position based on current direction"""
        if direction == Direction.NORTH:
            return current_x, current_y + self.step_size
        elif direction == Direction.SOUTH:
            return current_x, current_y - self.step_size
        elif direction == Direction.EAST:
            return current_x + self.step_size, current_y
        elif direction == Direction.WEST:
            return current_x - self.step_size, current_y
        
        return current_x, current_y
    
    def _should_turn(self, current_x, current_y):
        """Determine if robot should turn based on current conditions"""
        # Check if we've reached max line length
        if self.current_line_length >= self.max_line_length:
            return True
        
        # Check if next position in current direction is blocked or too far
        next_x, next_y = self._get_next_position(current_x, current_y, self.current_direction)
        
        if not self._is_position_safe(next_x, next_y):
            return True
        
        if self._get_distance_from_start(next_x, next_y) > self.max_distance_from_start:
            return True
        
        return False
    
    def _get_next_coverage_direction(self):
        """Determine next coverage direction based on current state"""
        if self.current_direction == Direction.NORTH:
            return Direction.EAST
        elif self.current_direction == Direction.SOUTH:
            return Direction.EAST
        elif self.current_direction == Direction.EAST:
            # Alternate between North and South
            if self.coverage_state == CoverageState.MOVING_NORTH:
                return Direction.SOUTH
            else:
                return Direction.NORTH
        elif self.current_direction == Direction.WEST:
            # Alternate between North and South
            if self.coverage_state == CoverageState.MOVING_NORTH:
                return Direction.SOUTH
            else:
                return Direction.NORTH
        
        return Direction.NORTH
    
    def _find_next_valid_position(self, current_x, current_y):
        """Find next valid position for coverage"""
        rospy.loginfo(f"🔍 Finding next position from ({current_x:.2f}, {current_y:.2f})")
        rospy.loginfo(f"📏 Current line length: {self.current_line_length:.2f}m, Max: {self.max_line_length:.2f}m")
        
        # Try current direction first
        should_turn = self._should_turn(current_x, current_y)
        rospy.loginfo(f"🔄 Should turn: {should_turn}")
        
        if not should_turn:
            next_x, next_y = self._get_next_position(current_x, current_y, self.current_direction)
            rospy.loginfo(f"🎯 Trying current direction {self.current_direction.name}: ({next_x:.2f}, {next_y:.2f})")
            
            is_safe = self._is_position_safe(next_x, next_y)
            distance_ok = self._get_distance_from_start(next_x, next_y) <= self.max_distance_from_start
            
            rospy.loginfo(f"✅ Position safe: {is_safe}, Distance OK: {distance_ok} ({self._get_distance_from_start(next_x, next_y):.2f}m <= {self.max_distance_from_start}m)")
            
            if is_safe and distance_ok:
                return next_x, next_y, self.current_direction
        
        # Need to turn - try different directions
        rospy.loginfo("🔄 Turning to find new direction...")
        original_direction = self.current_direction
        
        for attempt in range(4):  # Try all directions
            old_direction = self.current_direction
            self.current_direction = self._get_next_coverage_direction()
            self.current_line_length = 0.0  # Reset line length when turning
            
            rospy.loginfo(f"🔄 Attempt {attempt + 1}: Trying direction {self.current_direction.name} (was {old_direction.name})")
            
            # Update coverage state
            if self.current_direction == Direction.NORTH:
                self.coverage_state = CoverageState.MOVING_NORTH
            elif self.current_direction == Direction.SOUTH:
                self.coverage_state = CoverageState.MOVING_SOUTH
            elif self.current_direction == Direction.EAST:
                self.coverage_state = CoverageState.MOVING_EAST
            elif self.current_direction == Direction.WEST:
                self.coverage_state = CoverageState.MOVING_WEST
            
            next_x, next_y = self._get_next_position(current_x, current_y, self.current_direction)
            
            is_safe = self._is_position_safe(next_x, next_y)
            distance_ok = self._get_distance_from_start(next_x, next_y) <= self.max_distance_from_start
            
            rospy.loginfo(f"🎯 Direction {self.current_direction.name}: ({next_x:.2f}, {next_y:.2f}) - Safe: {is_safe}, Distance OK: {distance_ok}")
            
            if is_safe and distance_ok:
                return next_x, next_y, self.current_direction
        
        # No valid position found
        rospy.logwarn("❌ No valid direction found after trying all options")
        return None, None, None
    
    def _move_to_position(self, target_x, target_y):
        """Move robot to specified position using move_base"""
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = target_x
        goal.target_pose.pose.position.y = target_y
        goal.target_pose.pose.position.z = 0.0
        
        # Set orientation based on movement direction
        yaw = 0.0
        if self.current_direction == Direction.NORTH:
            yaw = math.pi / 2
        elif self.current_direction == Direction.SOUTH:
            yaw = -math.pi / 2
        elif self.current_direction == Direction.EAST:
            yaw = 0.0
        elif self.current_direction == Direction.WEST:
            yaw = math.pi
        
        goal.target_pose.pose.orientation.z = math.sin(yaw / 2)
        goal.target_pose.pose.orientation.w = math.cos(yaw / 2)
        
        self.current_goal = goal.target_pose
        self.move_base_client.send_goal(goal)
        
        # Wait for result
        self.total_moves += 1
        success = self.move_base_client.wait_for_result(timeout=rospy.Duration(self.max_goal_timeout))
        
        if success and self.move_base_client.get_state() == actionlib.GoalStatus.SUCCEEDED:
            self.successful_moves += 1
            # Mark position as visited
            grid_x, grid_y = self._world_to_grid(target_x, target_y)
            if grid_x is not None and grid_y is not None:
                self.visited_positions.add((grid_x, grid_y))
            
            # Update line length
            self.current_line_length += self.step_size
            
            rospy.loginfo(f"✅ Moved to ({target_x:.2f}, {target_y:.2f}) - Direction: {self.current_direction.name}")
            return True
        else:
            rospy.logwarn(f"❌ Failed to reach ({target_x:.2f}, {target_y:.2f})")
            return False
    
    def _publish_visualization_markers(self):
        """Publish visualization markers"""
        if self.current_pose is None:
            return
        
        marker_array = MarkerArray()
        
        # Current goal marker
        if self.current_goal is not None:
            goal_marker = Marker()
            goal_marker.header.frame_id = "map"
            goal_marker.header.stamp = rospy.Time.now()
            goal_marker.id = 0
            goal_marker.type = Marker.ARROW
            goal_marker.action = Marker.ADD
            goal_marker.pose = self.current_goal.pose
            goal_marker.scale.x = 0.5
            goal_marker.scale.y = 0.1
            goal_marker.scale.z = 0.1
            goal_marker.color.r = 1.0
            goal_marker.color.g = 0.0
            goal_marker.color.b = 0.0
            goal_marker.color.a = 0.8
            marker_array.markers.append(goal_marker)
        
        # Visited positions markers
        for i, (grid_x, grid_y) in enumerate(list(self.visited_positions)[-50:]):  # Last 50 positions
            if self.occupancy_map is None:
                continue
            
            grid_info = self.occupancy_map.info
            world_x = grid_info.origin.position.x + (grid_x + 0.5) * grid_info.resolution
            world_y = grid_info.origin.position.y + (grid_y + 0.5) * grid_info.resolution
            
            visited_marker = Marker()
            visited_marker.header.frame_id = "map"
            visited_marker.header.stamp = rospy.Time.now()
            visited_marker.id = i + 1
            visited_marker.type = Marker.CYLINDER
            visited_marker.action = Marker.ADD
            visited_marker.pose.position.x = world_x
            visited_marker.pose.position.y = world_y
            visited_marker.pose.position.z = 0.0
            visited_marker.pose.orientation.w = 1.0
            visited_marker.scale.x = 0.2
            visited_marker.scale.y = 0.2
            visited_marker.scale.z = 0.05
            visited_marker.color.r = 0.0
            visited_marker.color.g = 1.0
            visited_marker.color.b = 0.0
            visited_marker.color.a = 0.6
            marker_array.markers.append(visited_marker)
        
        # Start position marker
        if self.start_position is not None:
            start_marker = Marker()
            start_marker.header.frame_id = "map"
            start_marker.header.stamp = rospy.Time.now()
            start_marker.id = 1000
            start_marker.type = Marker.SPHERE
            start_marker.action = Marker.ADD
            start_marker.pose.position.x = self.start_position[0]
            start_marker.pose.position.y = self.start_position[1]
            start_marker.pose.position.z = 0.0
            start_marker.pose.orientation.w = 1.0
            start_marker.scale.x = 0.4
            start_marker.scale.y = 0.4
            start_marker.scale.z = 0.4
            start_marker.color.r = 0.0
            start_marker.color.g = 0.0
            start_marker.color.b = 1.0
            start_marker.color.a = 0.8
            marker_array.markers.append(start_marker)
        
        self.markers_pub.publish(marker_array)
    
    def execute_intelligent_coverage(self):
        """Execute intelligent coverage using AMCL-based decision making"""
        rospy.loginfo("🔄 Coverage thread started")
        
        # Wait for required data
        timeout = rospy.Time.now() + rospy.Duration(10.0)
        while (self.current_pose is None or self.occupancy_map is None) and rospy.Time.now() < timeout:
            rospy.loginfo("⏳ Waiting for AMCL pose and map data...")
            rospy.sleep(1.0)
        
        if self.current_pose is None:
            rospy.logerr("❌ No AMCL pose available after timeout")
            self.is_executing = False
            return False
        
        if self.occupancy_map is None:
            rospy.logerr("❌ No map available after timeout")
            self.is_executing = False
            return False
        
        self.is_executing = True
        self.visited_positions.clear()
        self.total_moves = 0
        self.successful_moves = 0
        self.current_line_length = 0.0
        self.current_direction = Direction.NORTH
        self.coverage_state = CoverageState.MOVING_NORTH
        
        rospy.loginfo("🚀 Starting AMCL-based intelligent coverage")
        rospy.loginfo(f"📍 Start position: ({self.start_position[0]:.2f}, {self.start_position[1]:.2f})")
        rospy.loginfo(f"🗺️ Map: {self.occupancy_map.info.width}x{self.occupancy_map.info.height}, resolution: {self.occupancy_map.info.resolution}")
        
        # Test first position
        with self.data_lock:
            current_x = self.current_pose.pose.pose.position.x
            current_y = self.current_pose.pose.pose.position.y
        
        rospy.loginfo(f"🧪 Testing current position safety: ({current_x:.2f}, {current_y:.2f})")
        if not self._is_position_safe(current_x, current_y):
            rospy.logwarn("⚠️ Current position is not safe! Adjusting safety parameters...")
            self.safety_margin = 0.1  # Reduce safety margin
        
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        rate = rospy.Rate(1.0)  # 1 Hz for better debugging
        
        while self.is_executing and not rospy.is_shutdown():
            with self.data_lock:
                if self.current_pose is None:
                    rospy.logwarn("⚠️ Lost AMCL pose")
                    rate.sleep()
                    continue
                
                current_x = self.current_pose.pose.pose.position.x
                current_y = self.current_pose.pose.pose.position.y
            
            rospy.loginfo(f"🔍 Current position: ({current_x:.2f}, {current_y:.2f}), Direction: {self.current_direction.name}")
            
            # Find next valid position
            next_x, next_y, next_direction = self._find_next_valid_position(current_x, current_y)
            
            if next_x is None:
                consecutive_failures += 1
                rospy.logwarn(f"❌ No valid next position found (failure {consecutive_failures}/{max_consecutive_failures})")
                
                if consecutive_failures >= max_consecutive_failures:
                    rospy.loginfo("🎯 No more valid positions found - Coverage completed!")
                    self.coverage_state = CoverageState.COMPLETED
                    break
                
                # Try reducing constraints
                if consecutive_failures == 2:
                    rospy.loginfo("🔧 Reducing safety margin to find more positions")
                    self.safety_margin *= 0.5
                elif consecutive_failures == 3:
                    rospy.loginfo("🔧 Reducing step size to find more positions")
                    self.step_size *= 0.8
                
                rate.sleep()
                continue
            
            consecutive_failures = 0  # Reset failure counter
            rospy.loginfo(f"🎯 Next target: ({next_x:.2f}, {next_y:.2f})")
            
            # Move to next position
            if self._move_to_position(next_x, next_y):
                # Publish visualization
                self._publish_visualization_markers()
                
                # Log progress
                distance_from_start = self._get_distance_from_start(next_x, next_y)
                rospy.loginfo(f"📊 Progress: {len(self.visited_positions)} positions visited, "
                             f"Distance from start: {distance_from_start:.2f}m, "
                             f"Success rate: {(self.successful_moves/self.total_moves)*100:.1f}%")
            else:
                rospy.logwarn("❌ Failed to reach target position")
            
            rate.sleep()
        
        self.is_executing = False
        rospy.loginfo("✅ Intelligent coverage execution completed")
        rospy.loginfo(f"📊 Final stats: {len(self.visited_positions)} positions visited, "
                     f"{self.successful_moves}/{self.total_moves} successful moves")
        return True
    
    # Service implementations (compatible with your controller)
    def start_coverage_service(self, req):
        """Service to start coverage execution"""
        response = TriggerResponse()
        
        if self.is_executing:
            response.success = False
            response.message = "Coverage execution already in progress"
            return response
        
        if self.current_pose is None:
            response.success = False
            response.message = "No AMCL pose available"
            return response
        
        if self.occupancy_map is None:
            response.success = False
            response.message = "No map available"
            return response
        
        # Check if move_base is available
        if not self.move_base_client.wait_for_server(timeout=rospy.Duration(2.0)):
            response.success = False
            response.message = "move_base action server not available"
            return response
        
        rospy.loginfo("🚀 Starting coverage execution...")
        rospy.loginfo(f"__ Current pose: ({self.current_pose.pose.pose.position.x:.2f}, {self.current_pose.pose.pose.position.y:.2f})")
        rospy.loginfo(f"🗺️ Map size: {self.occupancy_map.info.width}x{self.occupancy_map.info.height}")
        
        # Start execution in a separate thread
        import threading
        execution_thread = threading.Thread(target=self.execute_intelligent_coverage)
        execution_thread.daemon = True
        execution_thread.start()
        
        response.success = True
        response.message = "Started AMCL-based intelligent coverage"
        return response
    
    def stop_coverage_service(self, req):
        """Service to stop coverage execution"""
        response = TriggerResponse()
        
        self.is_executing = False
        self.move_base_client.cancel_all_goals()
        
        response.success = True
        response.message = "Coverage execution stopped"
        return response
    
    def generate_path_service(self, req):
        """Service to generate coverage path (for compatibility)"""
        response = TriggerResponse()
        
        if self.current_pose is None:
            response.success = False
            response.message = "No AMCL pose available for path generation"
            return response
        
        response.success = True
        response.message = "AMCL-based planner generates path dynamically during execution"
        return response
    
    def get_progress_service(self, req):
        """Service to get coverage progress"""
        response = TriggerResponse()
        
        visited_count = len(self.visited_positions)
        success_rate = (self.successful_moves / max(1, self.total_moves)) * 100
        
        if self.current_pose and self.start_position:
            current_distance = self._get_distance_from_start(
                self.current_pose.pose.pose.position.x,
                self.current_pose.pose.pose.position.y
            )
            
            response.success = True
            response.message = (f"Visited: {visited_count} positions, "
                               f"Success rate: {success_rate:.1f}%, "
                               f"Distance from start: {current_distance:.2f}m, "
                               f"Direction: {self.current_direction.name}, "
                               f"State: {self.coverage_state.name}")
        else:
            response.success = True
            response.message = f"Visited: {visited_count} positions, Success rate: {success_rate:.1f}%"
        
        return response
    
    def run(self):
        """Main run loop"""
        rate = rospy.Rate(1.0)  # 1 Hz
        
        while not rospy.is_shutdown():
            if self.is_executing:
                self._publish_visualization_markers()
            rate.sleep()

if __name__ == '__main__':
    try:
        planner = AMCLIntelligentCoveragePlanner()
        planner.run()
    except rospy.ROSInterruptException:
        pass


