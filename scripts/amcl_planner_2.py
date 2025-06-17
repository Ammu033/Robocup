

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
from collections import deque

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
        self.coverage_strategy = rospy.get_param('~coverage_strategy', 'grid_dfs')  # grid_dfs, grid_bfs, boustrophedon
        
        # State variables
        self.occupancy_map = None
        self.current_pose = None
        self.start_position = None  # world coords
        self.visited_positions = set()  # grid cells visited set (x, y)
        self.current_direction = Direction.NORTH
        self.coverage_state = CoverageState.IDLE
        self.is_executing = False
        self.data_lock = Lock()
        
        # Coverage tracking
        self.current_line_length = 0.0
        self.current_goal = None
        self.total_moves = 0
        self.successful_moves = 0
        
        # Map grid info
        self.grid_width = None
        self.grid_height = None
        self.grid_resolution = None
        
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
            self.grid_width = msg.info.width
            self.grid_height = msg.info.height
            self.grid_resolution = msg.info.resolution
    
    def pose_callback(self, msg):
        """Receive current robot pose from AMCL"""
        with self.data_lock:
            self.current_pose = msg
            if self.start_position is None:
                self.start_position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
                rospy.loginfo(f"Set start position: ({self.start_position[0]:.2f}, {self.start_position[1]:.2f})")
    
    def _get_current_position(self):
        """Get current robot position in world coordinates"""
        with self.data_lock:
            if self.current_pose is None:
                return None, None
            return self.current_pose.pose.pose.position.x, self.current_pose.pose.pose.position.y
    
    def _world_to_grid(self, world_x, world_y):
        """Convert world coordinates to grid coordinates"""
        if self.occupancy_map is None:
            return None, None
        
        grid_info = self.occupancy_map.info
        grid_x = int((world_x - grid_info.origin.position.x) / grid_info.resolution)
        grid_y = int((world_y - grid_info.origin.position.y) / grid_info.resolution)
        return grid_x, grid_y
    
    def _grid_to_world(self, grid_x, grid_y):
        """Convert grid coordinates to world coordinates"""
        if self.occupancy_map is None:
            return None, None
        
        grid_info = self.occupancy_map.info
        world_x = grid_info.origin.position.x + (grid_x + 0.5) * grid_info.resolution
        world_y = grid_info.origin.position.y + (grid_y + 0.5) * grid_info.resolution
        return world_x, world_y
    
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
        
        # Check area around the position for obstacles
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
            # Mark position as visited in grid cells
            grid_x, grid_y = self._world_to_grid(target_x, target_y)
            if grid_x is not None and grid_y is not None:
                with self.data_lock:
                    self.visited_positions.add((grid_x, grid_y))
            
            # Update line length (only relevant for boustrophedon)
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
        
        # Visited positions markers (last 50)
        with self.data_lock:
            visited_list = list(self.visited_positions)[-50:]
        for i, (grid_x, grid_y) in enumerate(visited_list):
            if self.occupancy_map is None:
                continue
            
            world_x, world_y = self._grid_to_world(grid_x, grid_y)
            if world_x is None or world_y is None:
                continue
            
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

    def _grid_based_coverage(self, start_world_x=None, start_world_y=None):
        """Grid-based coverage using DFS or BFS on the occupancy grid."""
        rospy.loginfo("Starting grid-based coverage")
        if self.occupancy_map is None:
            rospy.logwarn("No occupancy map available for grid-based coverage")
            return False
        
        # Use current position if no start position provided
        if start_world_x is None or start_world_y is None:
            start_world_x, start_world_y = self._get_current_position()
            if start_world_x is None or start_world_y is None:
                rospy.logwarn("Cannot get current position for grid-based coverage")
                return False
        
        # Lock access to map and pose
        with self.data_lock:
            grid_info = self.occupancy_map.info
            grid_width = grid_info.width
            grid_height = grid_info.height
            grid_resolution = grid_info.resolution
            occupancy_data = np.array(self.occupancy_map.data).reshape((grid_height, grid_width))
            start_grid_x, start_grid_y = self._world_to_grid(start_world_x, start_world_y)

        if start_grid_x is None or start_grid_y is None:
            rospy.logwarn("Cannot convert current position to grid coordinates for coverage")
            return False
        
        rospy.loginfo(f"Grid-based coverage starting from current position: ({start_world_x:.2f}, {start_world_y:.2f}) -> grid ({start_grid_x}, {start_grid_y})")
        
        # Data structures for traversal
        if self.coverage_strategy == 'grid_bfs':
            frontier = deque()
            frontier.append((start_grid_x, start_grid_y))
        else:
            frontier = [(start_grid_x, start_grid_y)]  # DFS stack
        visited = set()
        visited.add((start_grid_x, start_grid_y))
        
        while frontier and not rospy.is_shutdown() and self.is_executing:
            if self.coverage_strategy == 'grid_bfs':
                current_x, current_y = frontier.popleft()
            else:
                current_x, current_y = frontier.pop()
            
            # Convert grid to world
            world_x, world_y = self._grid_to_world(current_x, current_y)
            if world_x is None or world_y is None:
                continue
            
            # Check if position is safe
            if not self._is_position_safe(world_x, world_y):
                rospy.loginfo(f"Skipping unsafe cell ({current_x}, {current_y})")
                continue
            
            # Move robot to this position
            rospy.loginfo(f"Moving to grid cell ({current_x}, {current_y}) world({world_x:.2f},{world_y:.2f})")
            self.current_direction = self._determine_direction_to(current_x, current_y)
            success = self._move_to_position(world_x, world_y)
            if not success:
                rospy.logwarn(f"Failed to move to cell ({current_x}, {current_y})")
                # Try boustrophedon from current position
                curr_x, curr_y = self._get_current_position()
                if curr_x is not None and curr_y is not None:
                    rospy.loginfo("Trying boustrophedon coverage from current position")
                    boustrophedon_success = self._boustrophedon_coverage(curr_x, curr_y)
                    if boustrophedon_success:
                        rospy.loginfo("Boustrophedon coverage succeeded, resuming grid-based coverage.")
                    else:
                        rospy.logwarn("Boustrophedon coverage failed. Stopping coverage execution.")
                        return False
                
                # Rebuild frontier to avoid duplicates
                frontier = deque([pos for pos in frontier if pos not in visited])
                continue  # Go to next cell after fallback
            
            # Mark as visited
            with self.data_lock:
                self.visited_positions.add((current_x, current_y))
            
            # Add neighbors
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = current_x + dx, current_y + dy
                if (0 <= nx < grid_width and 0 <= ny < grid_height):
                    if (nx, ny) in visited:
                        continue
                    # Check occupancy
                    if occupancy_data[ny][nx] < 50:
                        visited.add((nx, ny))
                        if self.coverage_strategy == 'grid_bfs':
                            frontier.append((nx, ny))
                        else:
                            frontier.append((nx, ny))
            
            # Publish visualization markers
            self._publish_visualization_markers()
            rospy.sleep(0.1)
        
        rospy.loginfo("Grid-based coverage completed")
        return True
    
    def _grid_based_bfs_coverage(self, start_world_x=None, start_world_y=None):
        """Grid-based BFS coverage on the occupancy grid with fallback to DFS on failure."""
        rospy.loginfo("Starting grid-based BFS coverage")
        if self.occupancy_map is None:
            rospy.logwarn("No occupancy map available for BFS coverage")
            return False
        
        # Use current position if no start position provided
        if start_world_x is None or start_world_y is None:
            start_world_x, start_world_y = self._get_current_position()
            if start_world_x is None or start_world_y is None:
                rospy.logwarn("Cannot get current position for BFS coverage")
                return False
        
        with self.data_lock:
            grid_info = self.occupancy_map.info
            grid_width = grid_info.width
            grid_height = grid_info.height
            occupancy_data = np.array(self.occupancy_map.data).reshape((grid_height, grid_width))
            start_grid_x, start_grid_y = self._world_to_grid(start_world_x, start_world_y)

        if start_grid_x is None or start_grid_y is None:
            rospy.logwarn("Cannot convert current position to grid coordinates for BFS coverage")
            return False
        
        rospy.loginfo(f"BFS coverage starting from current position: ({start_world_x:.2f}, {start_world_y:.2f}) -> grid ({start_grid_x}, {start_grid_y})")
        
        frontier = deque()
        frontier.append((start_grid_x, start_grid_y))
        visited = set()
        visited.add((start_grid_x, start_grid_y))
        
        while frontier and not rospy.is_shutdown() and self.is_executing:
            current_x, current_y = frontier.popleft()
            
            world_x, world_y = self._grid_to_world(current_x, current_y)
            if world_x is None or world_y is None:
                continue
            
            if not self._is_position_safe(world_x, world_y):
                rospy.loginfo(f"BFS skipping unsafe cell ({current_x}, {current_y})")
                continue
            
            rospy.loginfo(f"BFS moving to grid cell ({current_x}, {current_y}) world({world_x:.2f},{world_y:.2f})")
            self.current_direction = self._determine_direction_to(current_x, current_y)
            
            success = self._move_to_position(world_x, world_y)
            if not success:
                rospy.logwarn("BFS move failed, switching to Grid DFS coverage from current position")
                curr_x, curr_y = self._get_current_position()
                if curr_x is not None and curr_y is not None:
                    return self._grid_based_coverage(curr_x, curr_y)
                else:
                    return False
            
            with self.data_lock:
                self.visited_positions.add((current_x, current_y))
            
            # Add neighbors (4-connected)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = current_x + dx, current_y + dy
                if (0 <= nx < grid_width and 0 <= ny < grid_height):
                    if (nx, ny) in visited:
                        continue
                    if occupancy_data[ny][nx] < 50:
                        visited.add((nx, ny))
                        frontier.append((nx, ny))
            
            self._publish_visualization_markers()
            rospy.sleep(0.1)
        
        rospy.loginfo("Grid-based BFS coverage completed")
        return True

    
    def _determine_direction_to(self, grid_x, grid_y):
        """Determine movement direction to a grid cell relative to current robot position in grid coords."""
        current_world_x, current_world_y = self._get_current_position()
        if current_world_x is None or current_world_y is None:
            return self.current_direction
            
        with self.data_lock:
            if self.occupancy_map is None:
                return self.current_direction
            cur_gx, cur_gy = self._world_to_grid(current_world_x, current_world_y)

        if cur_gx is None or cur_gy is None:
            return self.current_direction
        
        # Calculate the difference in grid coordinates
        dx = grid_x - cur_gx
        dy = grid_y - cur_gy

        # Determine potential next positions based on current direction
        potential_directions = []
        if dx > 0:  # Target is to the east
            potential_directions.append(Direction.EAST)
        elif dx < 0:  # Target is to the west
            potential_directions.append(Direction.WEST)

        if dy > 0:  # Target is to the north
            potential_directions.append(Direction.NORTH)
        elif dy < 0:  # Target is to the south
            potential_directions.append(Direction.SOUTH)

        # Check each potential direction for safety
        for direction in potential_directions:
            next_x, next_y = self._get_next_position(cur_gx, cur_gy, direction)
            world_next_x, world_next_y = self._grid_to_world(next_x, next_y)
            if world_next_x is not None and world_next_y is not None:
                if self._is_position_safe(world_next_x, world_next_y):
                    return direction  # Return the first safe direction found

        # If no safe direction is found, return the current direction
        return self.current_direction

    
    def _boustrophedon_coverage(self, start_world_x=None, start_world_y=None):
        """Implement boustrophedon (zig-zag) coverage pattern with backtracking."""
        rospy.loginfo("Starting boustrophedon coverage")
        if self.occupancy_map is None:
            rospy.logwarn("No occupancy map available for boustrophedon coverage")
            return False
        
        # Use current position if no start position provided
        if start_world_x is None or start_world_y is None:
            start_world_x, start_world_y = self._get_current_position()
            if start_world_x is None or start_world_y is None:
                rospy.logwarn("Cannot get current position for boustrophedon coverage")
                return False
        
        with self.data_lock:
            grid_info = self.occupancy_map.info
            grid_width = grid_info.width
            grid_height = grid_info.height
            grid_resolution = grid_info.resolution
            occupancy_data = np.array(self.occupancy_map.data).reshape((grid_height, grid_width))
            start_grid_x, start_grid_y = self._world_to_grid(start_world_x, start_world_y)
        
        if start_grid_x is None or start_grid_y is None:
            rospy.logwarn("Cannot convert current position to grid coordinates for boustrophedon")
            return False
        
        rospy.loginfo(f"Boustrophedon coverage starting from current position: ({start_world_x:.2f}, {start_world_y:.2f}) -> grid ({start_grid_x}, {start_grid_y})")
        
        visited = set()
        current_row = start_grid_y
        direction_left_to_right = True  # zigzag direction flag
        
        for row in range(start_grid_y, grid_height):
            # Define column traversal order
            if direction_left_to_right:
                col_range = range(start_grid_x if row == start_grid_y else 0, grid_width)
            else:
                col_range = range(grid_width-1 if row != start_grid_y else start_grid_x, -1, -1)
            
            for col in col_range:
                if not self.is_executing:
                    return False
                    
                # Check if cell is safe (free)
                if occupancy_data[row][col] < 50:
                    world_x, world_y = self._grid_to_world(col, row)
                    if world_x is None or world_y is None:
                        continue
                    if not self._is_position_safe(world_x, world_y):
                        continue
                    
                    # Skip if visited
                    if (col, row) in visited:
                        continue
                    
                    # Set direction for movement
                    if direction_left_to_right:
                        self.current_direction = Direction.EAST
                    else:
                        self.current_direction = Direction.WEST
                    
                    # Move to position
                    rospy.loginfo(f"Boustrophedon moving to grid ({col},{row}) world({world_x:.2f},{world_y:.2f})")
                    success = self._move_to_position(world_x, world_y)
                    if not success:
                        rospy.logwarn(f"Failed to move to boustrophedon cell ({col},{row}), trying grid-based coverage from current position.")
                        curr_x, curr_y = self._get_current_position()
                        if curr_x is not None and curr_y is not None:
                            return self._grid_based_coverage(curr_x, curr_y)
                        else:
                            return False
                    
                    with self.data_lock:
                        visited.add((col, row))
                        self.visited_positions.add((col, row))
                    
                    self._publish_visualization_markers()
                    rospy.sleep(0.1)
                
                
                else:
                    rospy.loginfo(f"Boustrophedon skipping occupied cell ({col},{row})")
            
            # Switch direction for next row (zigzag pattern)
            direction_left_to_right = not direction_left_to_right
            
            # Move to next row
            if row < grid_height - 1:
                next_row_world_x, next_row_world_y = self._grid_to_world(col, row + 1)
                if next_row_world_x is not None and next_row_world_y is not None:
                    if self._is_position_safe(next_row_world_x, next_row_world_y):
                        self.current_direction = Direction.NORTH
                        rospy.loginfo(f"Boustrophedon moving to next row: ({next_row_world_x:.2f}, {next_row_world_y:.2f})")
                        success = self._move_to_position(next_row_world_x, next_row_world_y)
                        if not success:
                            rospy.logwarn("Failed to move to next row, trying grid coverage from current position")
                            curr_x, curr_y = self._get_current_position()
                            if curr_x is not None and curr_y is not None:
                                return self._grid_based_coverage(curr_x, curr_y)
                            else:
                                return False
        
        rospy.loginfo("Boustrophedon coverage completed")
        return True
    
    def _get_coverage_progress(self):
        """Calculate coverage progress as percentage"""
        if self.occupancy_map is None:
            return 0.0
        
        with self.data_lock:
            grid_info = self.occupancy_map.info
            grid_width = grid_info.width
            grid_height = grid_info.height
            occupancy_data = np.array(self.occupancy_map.data).reshape((grid_height, grid_width))
            
            # Count free cells
            free_cells = np.sum(occupancy_data < 50)
            if free_cells == 0:
                return 100.0
            
            # Count visited cells
            visited_count = len(self.visited_positions)
            
            progress = min(100.0, (visited_count / free_cells) * 100.0)
            return progress
    
    def _execute_coverage(self):
        """Main coverage execution method"""
        rospy.loginfo(f"Starting coverage execution with strategy: {self.coverage_strategy}")
        
        # Get current position as starting point
        current_x, current_y = self._get_current_position()
        if current_x is None or current_y is None:
            rospy.logwarn("Cannot get current position to start coverage")
            return False
        
        self.coverage_state = CoverageState.MOVING_NORTH
        
        try:
            if self.coverage_strategy == 'grid_dfs':
                success = self._grid_based_coverage(current_x, current_y)
            elif self.coverage_strategy == 'grid_bfs':
                success = self._grid_based_bfs_coverage(current_x, current_y)
            elif self.coverage_strategy == 'boustrophedon':
                success = self._boustrophedon_coverage(current_x, current_y)
            else:
                rospy.logwarn(f"Unknown coverage strategy: {self.coverage_strategy}")
                success = self._grid_based_coverage(current_x, current_y)
            
            if success:
                self.coverage_state = CoverageState.COMPLETED
                rospy.loginfo("Coverage execution completed successfully")
            else:
                self.coverage_state = CoverageState.IDLE
                rospy.logwarn("Coverage execution failed")
            
            return success
            
        except Exception as e:
            rospy.logerr(f"Exception during coverage execution: {e}")
            self.coverage_state = CoverageState.IDLE
            return False
    
    def start_coverage_service(self, req):
        """Service to start coverage"""
        if self.is_executing:
            return TriggerResponse(success=False, message="Coverage is already executing")
        
        if self.occupancy_map is None:
            return TriggerResponse(success=False, message="No occupancy map available")
        
        if self.current_pose is None:
            return TriggerResponse(success=False, message="No current pose available")
        
        # Reset coverage state
        with self.data_lock:
            self.visited_positions.clear()
        self.total_moves = 0
        self.successful_moves = 0
        self.current_line_length = 0.0
        
        self.is_executing = True
        rospy.loginfo("Starting coverage execution...")
        
        # Execute coverage in a separate thread-like manner
        success = self._execute_coverage()
        
        self.is_executing = False
        
        if success:
            progress = self._get_coverage_progress()
            return TriggerResponse(
                success=True, 
                message=f"Coverage completed successfully. Progress: {progress:.1f}%, Moves: {self.successful_moves}/{self.total_moves}"
            )
        else:
            progress = self._get_coverage_progress()
            return TriggerResponse(
                success=False, 
                message=f"Coverage failed. Progress: {progress:.1f}%, Moves: {self.successful_moves}/{self.total_moves}"
            )
    
    def stop_coverage_service(self, req):
        """Service to stop coverage"""
        if not self.is_executing:
            return TriggerResponse(success=False, message="Coverage is not executing")
        
        self.is_executing = False
        self.coverage_state = CoverageState.IDLE
        
        # Cancel current goal
        if self.move_base_client.get_state() in [actionlib.GoalStatus.PENDING, actionlib.GoalStatus.ACTIVE]:
            self.move_base_client.cancel_goal()
        
        # Stop robot
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)
        
        progress = self._get_coverage_progress()
        return TriggerResponse(
            success=True, 
            message=f"Coverage stopped. Progress: {progress:.1f}%, Moves: {self.successful_moves}/{self.total_moves}"
        )
    
    def generate_path_service(self, req):
        """Service to generate coverage path without executing"""
        if self.occupancy_map is None:
            return TriggerResponse(success=False, message="No occupancy map available")
        
        if self.current_pose is None:
            return TriggerResponse(success=False, message="No current pose available")
        
        # This is a simplified path generation - in practice, you might want to
        # implement a more sophisticated path planning algorithm
        rospy.loginfo("Generating coverage path...")
        
        # Calculate approximate path length based on free space
        with self.data_lock:
            grid_info = self.occupancy_map.info
            grid_width = grid_info.width
            grid_height = grid_info.height
            occupancy_data = np.array(self.occupancy_map.data).reshape((grid_height, grid_width))
            
            free_cells = np.sum(occupancy_data < 50)
            estimated_path_length = free_cells * self.step_size
        
        return TriggerResponse(
            success=True, 
            message=f"Path generated. Estimated length: {estimated_path_length:.2f}m, Free cells: {free_cells}"
        )
    
    def get_progress_service(self, req):
        """Service to get current coverage progress"""
        progress = self._get_coverage_progress()
        
        # Get current position
        current_x, current_y = self._get_current_position()
        position_str = f"({current_x:.2f}, {current_y:.2f})" if current_x is not None else "Unknown"
        
        # Calculate distance from start
        distance_from_start = 0.0
        if current_x is not None and current_y is not None and self.start_position is not None:
            distance_from_start = self._get_distance_from_start(current_x, current_y)
        
        message = (
            f"Progress: {progress:.1f}%, "
            f"State: {self.coverage_state.name}, "
            f"Position: {position_str}, "
            f"Distance from start: {distance_from_start:.2f}m, "
            f"Moves: {self.successful_moves}/{self.total_moves}, "
            f"Visited cells: {len(self.visited_positions)}"
        )
        
        return TriggerResponse(success=True, message=message)
    
    def run(self):
        """Main run loop"""
        rate = rospy.Rate(10)  # 10 Hz
        
        while not rospy.is_shutdown():
            try:
                # Publish visualization markers
                self._publish_visualization_markers()
                
                # Check if we're executing and handle timeouts
                if self.is_executing:
                    if self.move_base_client.get_state() in [actionlib.GoalStatus.PENDING, actionlib.GoalStatus.ACTIVE]:
                        # Check for timeout
                        if self.current_goal is not None:
                            current_time = rospy.Time.now()
                            if hasattr(self, 'goal_start_time'):
                                elapsed = (current_time - self.goal_start_time).to_sec()
                                if elapsed > self.max_goal_timeout:
                                    rospy.logwarn("Goal timeout reached, canceling current goal")
                                    self.move_base_client.cancel_goal()
                
                rate.sleep()
                
            except rospy.ROSInterruptException:
                break
            except Exception as e:
                rospy.logerr(f"Error in main loop: {e}")
                rate.sleep()
        
        rospy.loginfo("Coverage planner shutting down")

if __name__ == '__main__':
    try:
        planner = AMCLIntelligentCoveragePlanner()
        planner.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Coverage planner interrupted")
    except Exception as e:
        rospy.logerr(f"Failed to start coverage planner: {e}")