#!/usr/bin/env python3

import rospy
import numpy as np
import math
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from std_srvs.srv import Trigger, TriggerResponse
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from actionlib_msgs.msg import GoalStatusArray
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from threading import Lock
from enum import Enum

class CoveragePattern(Enum):
    BOUSTROPHEDON = "boustrophedon"  # Back-and-forth pattern
    SPIRAL = "spiral"                # Spiral inward/outward
    ZIGZAG = "zigzag"               # Zigzag pattern

class CompleteCoveragePlanner:
    def __init__(self):
        rospy.init_node('complete_coverage_planner', anonymous=True)
        
        # --- Parameters ---
        self.robot_radius = rospy.get_param('~robot_radius', 0.3)
        self.coverage_resolution = rospy.get_param('~coverage_resolution', 0.2)
        self.overlap_factor = rospy.get_param('~overlap_factor', 0.1)  # 10% overlap
        self.pattern_type = rospy.get_param('~pattern_type', 'boustrophedon')
        self.goal_tolerance = rospy.get_param('~goal_tolerance', 0.2)
        self.max_goal_timeout = rospy.get_param('~max_goal_timeout', 30.0)
        
        # State variables
        self.occupancy_map = None
        self.current_pose = None
        self.coverage_path = []
        self.current_goal_index = 0
        self.is_executing = False
        self.data_lock = Lock()
        
        # Coverage grid for tracking
        self.coverage_grid = None
        self.visited_cells = set()
        self.total_cells_to_visit = 0
        
        # --- Subscribers ---
        rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
        rospy.Subscriber("/amcl_pose", PoseStamped, self.pose_callback)
        rospy.Subscriber("/coverage_map", OccupancyGrid, self.coverage_map_callback)
        
        # --- Publishers ---
        self.path_pub = rospy.Publisher("/coverage_path", Path, queue_size=1)
        self.goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)
        self.markers_pub = rospy.Publisher("/coverage_path_markers", MarkerArray, queue_size=1)
        
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
        
        rospy.loginfo("Complete Coverage Planner initialized")

    def map_callback(self, msg):
        """Receive the occupancy grid map"""
        with self.data_lock:
            self.occupancy_map = msg
            rospy.loginfo(f"Received map: {msg.info.width}x{msg.info.height}")

    def pose_callback(self, msg):
        """Receive current robot pose"""
        with self.data_lock:
            self.current_pose = msg

    def coverage_map_callback(self, msg):
        """Receive coverage map from the coverage tracker"""
        with self.data_lock:
            self.coverage_grid = msg
            self._update_visited_cells()

    def _update_visited_cells(self):
        """Update the set of visited cells based on coverage map"""
        if self.coverage_grid is None:
            return
            
        self.visited_cells.clear()
        for y in range(self.coverage_grid.info.height):
            for x in range(self.coverage_grid.info.width):
                idx = y * self.coverage_grid.info.width + x
                if self.coverage_grid.data[idx] > 0:  # Visited cell
                    self.visited_cells.add((x, y))

    def _is_cell_free(self, x, y):
        """Check if a cell in the map is free space"""
        if self.occupancy_map is None:
            return False
            
        if x < 0 or x >= self.occupancy_map.info.width or y < 0 or y >= self.occupancy_map.info.height:
            return False
            
        idx = y * self.occupancy_map.info.width + x
        return self.occupancy_map.data[idx] < 50  # Free space threshold

    def _world_to_grid(self, world_x, world_y, grid_info):
        """Convert world coordinates to grid coordinates"""
        grid_x = int((world_x - grid_info.origin.position.x) / grid_info.resolution)
        grid_y = int((world_y - grid_info.origin.position.y) / grid_info.resolution)
        return grid_x, grid_y

    def _grid_to_world(self, grid_x, grid_y, grid_info):
        """Convert grid coordinates to world coordinates"""
        world_x = grid_info.origin.position.x + (grid_x + 0.5) * grid_info.resolution
        world_y = grid_info.origin.position.y + (grid_y + 0.5) * grid_info.resolution
        return world_x, world_y

    def _generate_boustrophedon_path(self):
        """Generate boustrophedon (back-and-forth) coverage path"""
        if self.occupancy_map is None:
            return []

        path_points = []
        map_info = self.occupancy_map.info
        
        # Calculate step size based on robot radius and overlap
        step_size_cells = max(1, int((self.robot_radius * 2 * (1 - self.overlap_factor)) / map_info.resolution))
        
        # Find the bounds of free space
        min_x, max_x = map_info.width, 0
        min_y, max_y = map_info.height, 0
        
        for y in range(map_info.height):
            for x in range(map_info.width):
                if self._is_cell_free(x, y):
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)
                    min_y = min(min_y, y)
                    max_y = max(max_y, y)
        
        self.total_cells_to_visit = 0
        
        # Generate boustrophedon pattern
        going_right = True
        for y in range(min_y, max_y + 1, step_size_cells):
            if going_right:
                x_range = range(min_x, max_x + 1, step_size_cells)
            else:
                x_range = range(max_x, min_x - 1, -step_size_cells)
            
            row_points = []
            for x in x_range:
                if self._is_cell_free(x, y):
                    world_x, world_y = self._grid_to_world(x, y, map_info)
                    row_points.append((world_x, world_y))
                    self.total_cells_to_visit += 1
            
            # Add points to path
            path_points.extend(row_points)
            going_right = not going_right
        
        rospy.loginfo(f"Generated boustrophedon path with {len(path_points)} waypoints")
        return path_points

    def _generate_spiral_path(self):
        """Generate spiral coverage path"""
        if self.occupancy_map is None:
            return []

        path_points = []
        map_info = self.occupancy_map.info
        
        # Find center of the map
        center_x = map_info.width // 2
        center_y = map_info.height // 2
        
        # Calculate step size
        step_size_cells = max(1, int((self.robot_radius * 2 * (1 - self.overlap_factor)) / map_info.resolution))
        
        # Generate spiral pattern
        visited = set()
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        direction_idx = 0
        steps = 1
        
        x, y = center_x, center_y
        self.total_cells_to_visit = 0
        
        while len(visited) < map_info.width * map_info.height:
            for _ in range(2):  # Each step size is used twice
                for _ in range(steps):
                    if (x, y) not in visited and self._is_cell_free(x, y):
                        world_x, world_y = self._grid_to_world(x, y, map_info)
                        path_points.append((world_x, world_y))
                        self.total_cells_to_visit += 1
                        visited.add((x, y))
                    
                    dx, dy = directions[direction_idx]
                    x += dx * step_size_cells
                    y += dy * step_size_cells
                    
                    if x < 0 or x >= map_info.width or y < 0 or y >= map_info.height:
                        break
                
                direction_idx = (direction_idx + 1) % 4
                if x < 0 or x >= map_info.width or y < 0 or y >= map_info.height:
                    break
            
            steps += 1
            if x < 0 or x >= map_info.width or y < 0 or y >= map_info.height:
                break

        rospy.loginfo(f"Generated spiral path with {len(path_points)} waypoints")
        return path_points

    def _optimize_path(self, path_points):
        """Optimize the path by removing unnecessary waypoints and smoothing"""
        if len(path_points) < 3:
            return path_points
        
        optimized_path = [path_points[0]]
        
        for i in range(1, len(path_points) - 1):
            curr_point = path_points[i]
            prev_point = optimized_path[-1]
            next_point = path_points[i + 1]
            
            # Check if current point is necessary (not collinear)
            vec1 = (curr_point[0] - prev_point[0], curr_point[1] - prev_point[1])
            vec2 = (next_point[0] - curr_point[0], next_point[1] - curr_point[1])
            
            # Cross product to check collinearity
            cross_product = vec1[0] * vec2[1] - vec1[1] * vec2[0]
            
            if abs(cross_product) > 0.01:  # Not collinear, keep the point
                optimized_path.append(curr_point)
        
        optimized_path.append(path_points[-1])
        
        rospy.loginfo(f"Optimized path from {len(path_points)} to {len(optimized_path)} waypoints")
        return optimized_path

    def generate_coverage_path(self):
        """Generate complete coverage path"""
        with self.data_lock:
            if self.occupancy_map is None:
                rospy.logwarn("No map available for path generation")
                return False

            # Generate path based on selected pattern
            if self.pattern_type == CoveragePattern.BOUSTROPHEDON.value:
                path_points = self._generate_boustrophedon_path()
            elif self.pattern_type == CoveragePattern.SPIRAL.value:
                path_points = self._generate_spiral_path()
            else:
                rospy.logwarn(f"Unknown pattern type: {self.pattern_type}")
                return False

            if not path_points:
                rospy.logwarn("Failed to generate coverage path")
                return False

            # Optimize the path
            path_points = self._optimize_path(path_points)

            # Convert to ROS Path message
            self.coverage_path = []
            path_msg = Path()
            path_msg.header.frame_id = self.occupancy_map.header.frame_id
            path_msg.header.stamp = rospy.Time.now()

            for i, (x, y) in enumerate(path_points):
                pose = PoseStamped()
                pose.header = path_msg.header
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.position.z = 0.0
                
                # Calculate orientation to next point
                if i < len(path_points) - 1:
                    next_x, next_y = path_points[i + 1]
                    yaw = math.atan2(next_y - y, next_x - x)
                else:
                    yaw = 0.0
                
                pose.pose.orientation.z = math.sin(yaw / 2)
                pose.pose.orientation.w = math.cos(yaw / 2)
                
                path_msg.poses.append(pose)
                self.coverage_path.append(pose)

            # Publish path
            self.path_pub.publish(path_msg)
            self._publish_path_markers()
            
            rospy.loginfo(f"Generated coverage path with {len(self.coverage_path)} waypoints")
            return True

    def _publish_path_markers(self):
        """Publish visualization markers for the path"""
        if not self.coverage_path:
            return

        marker_array = MarkerArray()
        
        # Path line
        line_marker = Marker()
        line_marker.header.frame_id = self.occupancy_map.header.frame_id
        line_marker.header.stamp = rospy.Time.now()
        line_marker.id = 0
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.05
        line_marker.color.r = 0.0
        line_marker.color.g = 1.0
        line_marker.color.b = 0.0
        line_marker.color.a = 0.8
        
        for pose in self.coverage_path:
            point = Point()
            point.x = pose.pose.position.x
            point.y = pose.pose.position.y
            point.z = 0.1
            line_marker.points.append(point)
        
        marker_array.markers.append(line_marker)
        
        # Waypoint markers
        for i, pose in enumerate(self.coverage_path[::5]):  # Every 5th waypoint
            waypoint_marker = Marker()
            waypoint_marker.header.frame_id = self.occupancy_map.header.frame_id
            waypoint_marker.header.stamp = rospy.Time.now()
            waypoint_marker.id = i + 1
            waypoint_marker.type = Marker.SPHERE
            waypoint_marker.action = Marker.ADD
            waypoint_marker.pose = pose.pose
            waypoint_marker.scale.x = 0.1
            waypoint_marker.scale.y = 0.1
            waypoint_marker.scale.z = 0.1
            waypoint_marker.color.r = 1.0
            waypoint_marker.color.g = 0.0
            waypoint_marker.color.b = 0.0
            waypoint_marker.color.a = 0.8
            
            marker_array.markers.append(waypoint_marker)
        
        self.markers_pub.publish(marker_array)

    def execute_coverage_path(self):
        """Execute the coverage path using move_base"""
        if not self.coverage_path:
            rospy.logwarn("No coverage path available. Generate path first.")
            return False

        self.is_executing = True
        self.current_goal_index = 0
        
        rospy.loginfo(f"Starting coverage path execution with {len(self.coverage_path)} waypoints")
        
        while self.is_executing and self.current_goal_index < len(self.coverage_path):
            if rospy.is_shutdown():
                break
                
            current_goal = self.coverage_path[self.current_goal_index]
            rospy.loginfo(f"Going to waypoint {self.current_goal_index + 1}/{len(self.coverage_path)}: "
                         f"({current_goal.pose.position.x:.2f}, {current_goal.pose.position.y:.2f})")
            
            # Send goal to move_base
            goal = MoveBaseGoal()
            goal.target_pose = current_goal
            
            self.move_base_client.send_goal(goal)
            
            # Wait for result
            success = self.move_base_client.wait_for_result(timeout=rospy.Duration(self.max_goal_timeout))
            
            if success:
                state = self.move_base_client.get_state()
                if state == actionlib.GoalStatus.SUCCEEDED:
                    rospy.loginfo(f"Reached waypoint {self.current_goal_index + 1}")
                    self.current_goal_index += 1
                else:
                    rospy.logwarn(f"Failed to reach waypoint {self.current_goal_index + 1}, trying next one")
                    self.current_goal_index += 1
            else:
                rospy.logwarn(f"Timeout waiting for waypoint {self.current_goal_index + 1}")
                self.current_goal_index += 1
        
        self.is_executing = False
        rospy.loginfo("Coverage path execution completed")
        return True

    def get_coverage_progress(self):
        """Get current coverage progress"""
        if not self.coverage_path:
            return 0.0, 0, 0
        
        progress_percent = (self.current_goal_index / len(self.coverage_path)) * 100.0
        return progress_percent, self.current_goal_index, len(self.coverage_path)

    # Service callbacks
    def start_coverage_service(self, req):
        """Service to start coverage execution"""
        response = TriggerResponse()
        
        if self.is_executing:
            response.success = False
            response.message = "Coverage execution already in progress"
            return response
        
        if not self.coverage_path:
            if not self.generate_coverage_path():
                response.success = False
                response.message = "Failed to generate coverage path"
                return response
        
        # Start execution in a separate thread
        import threading
        execution_thread = threading.Thread(target=self.execute_coverage_path)
        execution_thread.daemon = True
        execution_thread.start()
        
        response.success = True
        response.message = f"Started coverage execution with {len(self.coverage_path)} waypoints"
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
        """Service to generate coverage path"""
        response = TriggerResponse()
        
        if self.generate_coverage_path():
            response.success = True
            response.message = f"Generated coverage path with {len(self.coverage_path)} waypoints"
        else:
            response.success = False
            response.message = "Failed to generate coverage path"
        
        return response

    def get_progress_service(self, req):
        """Service to get coverage progress"""
        response = TriggerResponse()
        
        progress, current, total = self.get_coverage_progress()
        visited_cells = len(self.visited_cells)
        
        response.success = True
        response.message = (f"Progress: {progress:.1f}% ({current}/{total} waypoints), "
                          f"Visited cells: {visited_cells}")
        return response

    def run(self):
        """Main run loop"""
        rate = rospy.Rate(1.0)  # 1 Hz
        
        while not rospy.is_shutdown():
            if self.is_executing:
                progress, current, total = self.get_coverage_progress()
                if current % 10 == 0:  # Log every 10 waypoints
                    rospy.loginfo(f"Coverage progress: {progress:.1f}% ({current}/{total})")
            
            rate.sleep()

if __name__ == '__main__':
    try:
        planner = CompleteCoveragePlanner()
        planner.run()
    except rospy.ROSInterruptException:
        pass