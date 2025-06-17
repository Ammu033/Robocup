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
from threading import Lock, Thread
from enum import Enum
import time

class Direction(Enum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

class CoverageState(Enum):
    IDLE = 0
    BOUSTROPHEDON = 1
    OBSTACLE_HUGGING = 2
    STRIP_TRANSITION = 3
    GAP_FILLING = 4
    COMPLETED = 5

class StripStatus(Enum):
    NOT_STARTED = 0
    IN_PROGRESS = 1
    COMPLETED = 2
    BLOCKED = 3

class AMCLIntelligentCoveragePlanner:
    def __init__(self):
        rospy.init_node('amcl_intelligent_coverage_planner', anonymous=True)
        
        # --- Parameters ---
        self.step_size = rospy.get_param('~step_size', 0.4)  # 40cm steps
        self.max_distance_from_start = rospy.get_param('~max_distance_from_start', 10.0)  # 8m limit
        self.strip_width = rospy.get_param('~strip_width', 0.8)  # Width between strips
        self.robot_radius = rospy.get_param('~robot_radius', 0.3)
        self.safety_margin = rospy.get_param('~safety_margin', 0.2)
        self.goal_tolerance = rospy.get_param('~goal_tolerance', 0.15)
        self.max_goal_timeout = rospy.get_param('~max_goal_timeout', 15.0)
        self.obstacle_hug_timeout = rospy.get_param('~obstacle_hug_timeout', 20.0)  # Max time hugging obstacle
        self.gap_detection_distance = rospy.get_param('~gap_detection_distance', 1.5)  # Distance to look for gaps
        self.min_strip_coverage = rospy.get_param('~min_strip_coverage', 80.0)  # Minimum coverage % to consider strip complete
        
        # --- State variables ---
        self.occupancy_map = None
        self.current_pose = None
        self.start_position = None
        self.visited_positions = set()
        self.current_direction = Direction.NORTH
        self.coverage_state = CoverageState.IDLE
        self.is_executing = False
        self.data_lock = Lock()
        
        self.strips = []  # list of x coordinates of strips
        self.current_strip_index = 0
        self.strip_status = {}
        self.strip_coverage = {}
        self.moving_north = True  # True means moving north in current strip, False south
        self.strip_positions = {}
        
        self.obstacle_hug_start_time = None
        self.obstacle_hug_positions = []
        self.last_successful_position = None
        
        self.incomplete_strips = []
        
        self.current_goal = None
        self.total_moves = 0
        self.successful_moves = 0
        self.blocked_attempts = 0
        self.coverage_start_time = None
        
        # Subscribers
        rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
        rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.pose_callback)
        
        # Publishers
        self.goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)
        self.markers_pub = rospy.Publisher("/coverage_markers", MarkerArray, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        
        # Action Client
        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base action server...")
        if self.move_base_client.wait_for_server(timeout=rospy.Duration(10.0)):
            rospy.loginfo("Connected to move_base action server")
        else:
            rospy.logwarn("Could not connect to move_base action server")
        
        # Services
        rospy.Service("/coverage_planner/start_coverage", Trigger, self.start_coverage_service)
        rospy.Service("/coverage_planner/stop_coverage", Trigger, self.stop_coverage_service)
        rospy.Service("/coverage_planner/generate_path", Trigger, self.generate_path_service)
        rospy.Service("/coverage_planner/get_progress", Trigger, self.get_progress_service)
        
        rospy.loginfo("AMCL Intelligent Coverage Planner initialized")
    
    def map_callback(self, msg):
        with self.data_lock:
            self.occupancy_map = msg
    
    def pose_callback(self, msg):
        with self.data_lock:
            self.current_pose = msg
            if self.start_position is None:
                self.start_position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
                rospy.loginfo(f"Set start position: ({self.start_position[0]:.2f}, {self.start_position[1]:.2f})")
    
    def _world_to_grid(self, world_x, world_y):
        if self.occupancy_map is None:
            return None, None
        
        grid_info = self.occupancy_map.info
        grid_x = int((world_x - grid_info.origin.position.x) / grid_info.resolution)
        grid_y = int((world_y - grid_info.origin.position.y) / grid_info.resolution)
        return grid_x, grid_y
    
    def _grid_to_world(self, grid_x, grid_y):
        if self.occupancy_map is None:
            return None, None
        
        grid_info = self.occupancy_map.info
        world_x = grid_info.origin.position.x + (grid_x + 0.5) * grid_info.resolution
        world_y = grid_info.origin.position.y + (grid_y + 0.5) * grid_info.resolution
        return world_x, world_y
    
    def _is_position_safe(self, world_x, world_y):
        if self.occupancy_map is None:
            rospy.logwarn("No occupancy map available for safety check")
            return False
        
        grid_x, grid_y = self._world_to_grid(world_x, world_y)
        if grid_x is None or grid_y is None:
            return False
        
        map_info = self.occupancy_map.info
        if (grid_x < 0 or grid_x >= map_info.width or 
            grid_y < 0 or grid_y >= map_info.height):
            return False
        
        safety_cells = int((self.robot_radius + self.safety_margin) / map_info.resolution)
        
        for dy in range(-safety_cells, safety_cells + 1):
            for dx in range(-safety_cells, safety_cells + 1):
                if math.sqrt(dx*dx + dy*dy) > safety_cells:
                    continue
                
                check_x = grid_x + dx
                check_y = grid_y + dy
                
                if (check_x < 0 or check_x >= map_info.width or 
                    check_y < 0 or check_y >= map_info.height):
                    return False
                
                idx = check_y * map_info.width + check_x
                if idx >= len(self.occupancy_map.data):
                    continue
                
                cell_value = self.occupancy_map.data[idx]
                if cell_value > 50 or cell_value == -1:
                    return False
        
        return True
    
    def _get_distance_from_start(self, x, y):
        if self.start_position is None:
            return float('inf')
        
        dx = x - self.start_position[0]
        dy = y - self.start_position[1]
        return math.sqrt(dx*dx + dy*dy)
    
    def _initialize_strips(self):
        if self.start_position is None:
            return False
        
        start_x, start_y = self.start_position
        
        self.strips = []
        self.strip_status = {}
        self.strip_coverage = {}
        self.strip_positions = {}
        
        max_strips_each_side = int(self.max_distance_from_start / self.strip_width)
        
        # add start strip
        self.strips.append(start_x)
        
        # east strips
        for i in range(1, max_strips_each_side + 1):
            strip_x = start_x + i * self.strip_width
            if self._has_safe_positions_in_strip(strip_x):
                self.strips.append(strip_x)
        
        # west strips
        for i in range(1, max_strips_each_side + 1):
            strip_x = start_x - i * self.strip_width
            if self._has_safe_positions_in_strip(strip_x):
                self.strips.insert(0, strip_x)
        
        self.strips.sort()
        
        for i, strip_x in enumerate(self.strips):
            self.strip_status[i] = StripStatus.NOT_STARTED
            self.strip_coverage[i] = 0.0
            self.strip_positions[i] = set()
        
        if self.strips:
            closest_strip_idx = min(range(len(self.strips)), 
                                  key=lambda i: abs(self.strips[i] - start_x))
            self.current_strip_index = closest_strip_idx
        
        rospy.loginfo(f"Initialized {len(self.strips)} strips for coverage")
        rospy.loginfo(f"Starting with strip {self.current_strip_index} at x={self.strips[self.current_strip_index]:.2f}")
        
        return len(self.strips) > 0
    
    def _has_safe_positions_in_strip(self, strip_x):
        start_y = self.start_position[1]
        test_positions = []
        for y_offset in np.linspace(-self.max_distance_from_start, self.max_distance_from_start, 20):
            test_y = start_y + y_offset
            test_positions.append((strip_x, test_y))
        
        safe_positions = 0
        for test_x, test_y in test_positions:
            if (self._is_position_safe(test_x, test_y) and
                self._get_distance_from_start(test_x, test_y) <= self.max_distance_from_start):
                safe_positions += 1
        
        return safe_positions >= len(test_positions) * 0.3
    
    def _get_next_boustrophedon_position(self, current_x, current_y):
        """Next position in boustrophedon pattern: moves vertically, then steps east when done."""
        
        if self.current_strip_index >= len(self.strips):
            return None, None
        
        strip_x = self.strips[self.current_strip_index]
        
        if self.moving_north:
            next_y = current_y + self.step_size
        else:
            next_y = current_y - self.step_size
        
        next_x = strip_x
        
        # Check boundary relative to start y
        start_y = self.start_position[1]
        upper_bound = start_y + self.max_distance_from_start
        lower_bound = start_y - self.max_distance_from_start
        
        # If next step exceeds boundary or is unsafe, move east to next strip and reverse direction
        if (self.moving_north and (next_y > upper_bound or not self._is_position_safe(next_x, next_y))) or \
           (not self.moving_north and (next_y < lower_bound or not self._is_position_safe(next_x, next_y))):
            # Step east to next strip
            next_strip_index = self.current_strip_index + 1
            if next_strip_index >= len(self.strips):
                return None, None  # No more strips to cover
            
            # Step east by strip width plus safety margin
            next_x = self.strips[next_strip_index]
            
            # Keep y at current boundary edge depending on moving direction
            if self.moving_north:
                next_y = min(current_y, upper_bound)
            else:
                next_y = max(current_y, lower_bound)
            
            # Reverse direction for next vertical move
            self.moving_north = not self.moving_north
            
            # Update current strip index to new strip position for next calls
            self.current_strip_index = next_strip_index
            
            rospy.loginfo(f"Moving east to strip {self.current_strip_index} at x={next_x:.2f}, changing vertical direction: {'north' if self.moving_north else 'south'}")
            
            # Check if next position is safe
            if not self._is_position_safe(next_x, next_y):
                return None, None
            
            return next_x, next_y
        
        # Normal step in current strip
        if not self._is_position_safe(next_x, next_y):
            return None, None
        
        return next_x, next_y
    
    def _move_to_next_strip(self):
        """When a strip is completed or blocked, move to next strip if available"""
        # Mark current strip completed or incomplete
        current_coverage = self.strip_coverage.get(self.current_strip_index, 0.0)
        if current_coverage >= self.min_strip_coverage:
            self.strip_status[self.current_strip_index] = StripStatus.COMPLETED
        else:
            if self.current_strip_index not in self.incomplete_strips:
                self.incomplete_strips.append(self.current_strip_index)
        
        # Try to find next strip needing coverage
        for i in range(len(self.strips)):
            if self.strip_status[i] in [StripStatus.NOT_STARTED, StripStatus.IN_PROGRESS]:
                self.current_strip_index = i
                self.strip_status[i] = StripStatus.IN_PROGRESS
                self.moving_north = True
                rospy.loginfo(f"Switching to strip {i} at x={self.strips[i]:.2f}")
                return True
        
        # No strips left to cover
        return False
    
    def _check_can_resume_boustrophedon(self, current_x, current_y):
        if self.current_strip_index >= len(self.strips):
            return False
        
        strip_x = self.strips[self.current_strip_index]
        if abs(current_x - strip_x) <= self.step_size * 1.5:
            next_x, next_y = self._get_next_boustrophedon_position(current_x, current_y)
            if next_x is not None and self._is_position_safe(next_x, next_y):
                return True
        return False
    
    def _find_obstacle_hug_position(self, current_x, current_y):
        """Try directions in order for obstacle hugging with left-hand rule"""
        # Try moving around obstacle with left-hand rule relative to current direction
        
        directions = []
        if self.moving_north:
            directions = [
                (0, self.step_size),       # forward (north)
                (-self.step_size, 0),      # left (west)
                (0, -self.step_size),      # backward (south)
                (self.step_size, 0),       # right (east)
            ]
        else:
            directions = [
                (0, -self.step_size),      # forward (south)
                (self.step_size, 0),       # left (east)
                (0, self.step_size),       # backward (north)
                (-self.step_size, 0),      # right (west)
            ]
        
        for dx, dy in directions:
            nx = current_x + dx
            ny = current_y + dy
            
            if (self._is_position_safe(nx, ny) and self._get_distance_from_start(nx, ny) <= self.max_distance_from_start):
                # Check if we can resume boustrophedon from this new position
                if self._check_can_resume_boustrophedon(nx, ny):
                    return nx, ny
                # Otherwise, return this position to keep hugging obstacle
                return nx, ny
        return None, None
    
    def _move_to_position(self, target_x, target_y):
        if not self._is_position_safe(target_x, target_y):
            rospy.logwarn(f"Target position ({target_x:.2f}, {target_y:.2f}) not safe")
            return False
        
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = target_x
        goal.target_pose.pose.position.y = target_y
        goal.target_pose.pose.position.z = 0.0
        
        yaw = 0.0
        if self.coverage_state == CoverageState.BOUSTROPHEDON:
            yaw = math.pi/2 if self.moving_north else -math.pi/2
        else:
            with self.data_lock:
                if self.current_pose is not None:
                    cx = self.current_pose.pose.pose.position.x
                    cy = self.current_pose.pose.pose.position.y
                    yaw = math.atan2(target_y - cy, target_x - cx)
        
        goal.target_pose.pose.orientation.z = math.sin(yaw/2)
        goal.target_pose.pose.orientation.w = math.cos(yaw/2)
        
        self.current_goal = goal.target_pose
        self.move_base_client.send_goal(goal)
        
        self.total_moves += 1
        success = self.move_base_client.wait_for_result(timeout=rospy.Duration(self.max_goal_timeout))
        
        if success and self.move_base_client.get_state() == actionlib.GoalStatus.SUCCEEDED:
            self.successful_moves += 1
            self.blocked_attempts = 0
            
            grid_x, grid_y = self._world_to_grid(target_x, target_y)
            if grid_x is not None and grid_y is not None:
                self.visited_positions.add((grid_x, grid_y))
                if self.current_strip_index < len(self.strips):
                    self.strip_positions[self.current_strip_index].add((grid_x, grid_y))
            
            self.last_successful_position = (target_x, target_y)
            rospy.loginfo(f"Moved to ({target_x:.2f}, {target_y:.2f})")
            return True
        else:
            self.blocked_attempts += 1
            rospy.logwarn(f"Failed to reach ({target_x:.2f}, {target_y:.2f}) - Blocked attempts: {self.blocked_attempts}")
            return False
    
    def _update_strip_coverage(self):
        if self.current_strip_index >= len(self.strips):
            return
        
        visited_count = len(self.strip_positions[self.current_strip_index])
        strip_length = 2 * self.max_distance_from_start
        estimated_positions = max(1, int(strip_length / self.step_size))
        coverage_percent = min(100.0, (visited_count / estimated_positions)*100)
        self.strip_coverage[self.current_strip_index] = coverage_percent
    
    def _execute_gap_filling(self):
        if not self.incomplete_strips:
            return False
        
        rospy.loginfo(f"Gap filling for {len(self.incomplete_strips)} incomplete strips")
        
        for strip_idx in self.incomplete_strips[:]:
            if not self.is_executing:
                break
            self.current_strip_index = strip_idx
            strip_x = self.strips[strip_idx]
            coverage = self.strip_coverage.get(strip_idx, 0.0)
            rospy.loginfo(f"Gap filling strip {strip_idx} coverage: {coverage:.1f}%")
            
            if self._fill_strip_gaps(strip_x):
                self._update_strip_coverage()
                new_cov = self.strip_coverage.get(strip_idx, 0.0)
                if new_cov >= self.min_strip_coverage:
                    self.strip_status[strip_idx] = StripStatus.COMPLETED
                    self.incomplete_strips.remove(strip_idx)
                    rospy.loginfo(f"Completed gap filling for strip {strip_idx} ({new_cov:.1f}%)")
                else:
                    rospy.logwarn(f"Strip {strip_idx} still incomplete after gap filling ({new_cov:.1f}%)")
        
        return len(self.incomplete_strips) == 0
    
    def _fill_strip_gaps(self, strip_x):
        start_y = self.start_position[1]
        gap_positions = []
        for y_offset in np.linspace(-self.max_distance_from_start, self.max_distance_from_start, 50):
            test_y = start_y + y_offset
            grid_x, grid_y = self._world_to_grid(strip_x, test_y)
            if (grid_x is not None and grid_y is not None and
                (grid_x, grid_y) not in self.visited_positions and
                self._is_position_safe(strip_x, test_y) and
                self._get_distance_from_start(strip_x, test_y) <= self.max_distance_from_start):
                gap_positions.append((strip_x, test_y))
        
        if gap_positions and self.current_pose is not None:
            cx = self.current_pose.pose.pose.position.x
            cy = self.current_pose.pose.pose.position.y
            gap_positions.sort(key=lambda p: math.sqrt((p[0]-cx)**2 + (p[1]-cy)**2))
        
        filled = 0
        for gx, gy in gap_positions:
            if not self.is_executing:
                break
            if self._move_to_position(gx, gy):
                filled += 1
                if filled >= 10:
                    break
        
        rospy.loginfo(f"Filled {filled} gaps in strip at x={strip_x:.2f}")
        return filled > 0
    
    def execute_intelligent_coverage(self):
        rospy.loginfo("Coverage thread started")
        
        timeout = rospy.Time.now() + rospy.Duration(10)
        while (self.current_pose is None or self.occupancy_map is None) and rospy.Time.now() < timeout:
            rospy.loginfo("Waiting for pose and map data...")
            rospy.sleep(1.0)
        
        if self.current_pose is None or self.occupancy_map is None:
            rospy.logerr("Required data not available")
            self.is_executing = False
            return False
        
        self.is_executing = True
        self.coverage_start_time = rospy.Time.now()
        self.visited_positions.clear()
        self.total_moves = 0
        self.successful_moves = 0
        self.blocked_attempts = 0
        self.moving_north = True
        self.incomplete_strips = []
        
        if not self._initialize_strips():
            rospy.logerr("Failed to initialize strips")
            self.is_executing = False
            return False
        
        self.coverage_state = CoverageState.BOUSTROPHEDON
        self.strip_status[self.current_strip_index] = StripStatus.IN_PROGRESS
        
        rospy.loginfo("Starting intelligent coverage")
        
        rate = rospy.Rate(1.0)
        
        while self.is_executing and not rospy.is_shutdown():
            with self.data_lock:
                if self.current_pose is None:
                    rospy.logwarn("Lost pose")
                    rate.sleep()
                    continue
                cx = self.current_pose.pose.pose.position.x
                cy = self.current_pose.pose.pose.position.y
            
            if self.coverage_state == CoverageState.BOUSTROPHEDON:
                nx, ny = self._get_next_boustrophedon_position(cx, cy)
                if nx is not None and ny is not None:
                    if self._move_to_position(nx, ny):
                        self._update_strip_coverage()
                        self.blocked_attempts = 0
                    else:
                        if self.blocked_attempts >= 3:
                            rospy.logwarn("Blocked multiple times, switching to obstacle hugging")
                            self.coverage_state = CoverageState.OBSTACLE_HUGGING
                            self.obstacle_hug_start_time = rospy.Time.now()
                            self.obstacle_hug_positions = []
                else:
                    # End of strip or blocked, try next strip
                    if not self._move_to_next_strip():
                        if self.incomplete_strips:
                            self.coverage_state = CoverageState.GAP_FILLING
                        else:
                            self.coverage_state = CoverageState.COMPLETED
            
            elif self.coverage_state == CoverageState.OBSTACLE_HUGGING:
                if (rospy.Time.now() - self.obstacle_hug_start_time).to_sec() > self.obstacle_hug_timeout:
                    rospy.logwarn("Obstacle hugging timeout, moving to next strip")
                    if not self._move_to_next_strip():
                        self.coverage_state = CoverageState.GAP_FILLING if self.incomplete_strips else CoverageState.COMPLETED
                    else:
                        self.coverage_state = CoverageState.BOUSTROPHEDON
                    continue
                
                hug_x, hug_y = self._find_obstacle_hug_position(cx, cy)
                if hug_x is not None and hug_y is not None:
                    if self._move_to_position(hug_x, hug_y):
                        self.obstacle_hug_positions.append((hug_x, hug_y))
                        if self._check_can_resume_boustrophedon(hug_x, hug_y):
                            rospy.loginfo("Found gap, resuming boustrophedon")
                            self.coverage_state = CoverageState.BOUSTROPHEDON
                    else:
                        rospy.logwarn("Obstacle hugging blocked, trying next strip")
                        if not self._move_to_next_strip():
                            self.coverage_state = CoverageState.GAP_FILLING if self.incomplete_strips else CoverageState.COMPLETED
                        else:
                            self.coverage_state = CoverageState.BOUSTROPHEDON
                else:
                    rospy.logwarn("No obstacle hug position found, moving to next strip")
                    if not self._move_to_next_strip():
                        self.coverage_state = CoverageState.GAP_FILLING if self.incomplete_strips else CoverageState.COMPLETED
                    else:
                        self.coverage_state = CoverageState.BOUSTROPHEDON
            
            elif self.coverage_state == CoverageState.GAP_FILLING:
                if self._execute_gap_filling():
                    rospy.loginfo("Gap filling completed")
                    self.coverage_state = CoverageState.COMPLETED
                else:
                    rospy.loginfo("Gap filling finished with gaps remaining")
                    self.coverage_state = CoverageState.COMPLETED
            
            elif self.coverage_state == CoverageState.COMPLETED:
                rospy.loginfo("Coverage task completed")
                break
            
            self._publish_coverage_markers()
            
            if self.total_moves > 0 and self.total_moves % 10 == 0:
                self._log_progress()
            
            rate.sleep()
        
        self._log_final_statistics()
        self.is_executing = False
        return True
    
    def _publish_coverage_markers(self):
        marker_array = MarkerArray()
        
        clear_marker = Marker()
        clear_marker.header.frame_id = "map"
        clear_marker.header.stamp = rospy.Time.now()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)
        
        if self.visited_positions:
            visited_marker = Marker()
            visited_marker.header.frame_id = "map"
            visited_marker.header.stamp = rospy.Time.now()
            visited_marker.ns = "visited_positions"
            visited_marker.id = 0
            visited_marker.type = Marker.POINTS
            visited_marker.action = Marker.ADD
            visited_marker.scale.x = 0.1
            visited_marker.scale.y = 0.1
            visited_marker.color.r = 0.0
            visited_marker.color.g = 1.0
            visited_marker.color.b = 0.0
            visited_marker.color.a = 0.8
            
            for gx, gy in self.visited_positions:
                wx, wy = self._grid_to_world(gx, gy)
                if wx is not None and wy is not None:
                    p = Point()
                    p.x = wx
                    p.y = wy
                    p.z = 0.0
                    visited_marker.points.append(p)
            
            marker_array.markers.append(visited_marker)
        
        if self.current_goal is not None:
            goal_marker = Marker()
            goal_marker.header.frame_id = "map"
            goal_marker.header.stamp = rospy.Time.now()
            goal_marker.ns = "current_goal"
            goal_marker.id = 1
            goal_marker.type = Marker.ARROW
            goal_marker.action = Marker.ADD
            goal_marker.pose = self.current_goal.pose
            goal_marker.scale.x = 0.5
            goal_marker.scale.y = 0.1
            goal_marker.scale.z = 0.1
            goal_marker.color.r = 1.0
            goal_marker.color.g = 0.0
            goal_marker.color.b = 0.0
            goal_marker.color.a = 1.0
            marker_array.markers.append(goal_marker)
        
        for i, strip_x in enumerate(self.strips):
            strip_marker = Marker()
            strip_marker.header.frame_id = "map"
            strip_marker.header.stamp = rospy.Time.now()
            strip_marker.ns = "strips"
            strip_marker.id = i + 10
            strip_marker.type = Marker.LINE_STRIP
            strip_marker.action = Marker.ADD
            strip_marker.scale.x = 0.05
            
            if self.strip_status[i] == StripStatus.COMPLETED:
                strip_marker.color.r = 0.0
                strip_marker.color.g = 1.0
                strip_marker.color.b = 0.0
                strip_marker.color.a = 0.5
            elif self.strip_status[i] == StripStatus.IN_PROGRESS:
                strip_marker.color.r = 1.0
                strip_marker.color.g = 1.0
                strip_marker.color.b = 0.0
                strip_marker.color.a = 0.7
            else:
                strip_marker.color.r = 0.5
                strip_marker.color.g = 0.5
                strip_marker.color.b = 0.5
                strip_marker.color.a = 0.3
            
            start_point = Point()
            start_point.x = strip_x
            start_point.y = self.start_position[1] - self.max_distance_from_start
            start_point.z = 0.0
            
            end_point = Point()
            end_point.x = strip_x
            end_point.y = self.start_position[1] + self.max_distance_from_start
            end_point.z = 0.0
            
            strip_marker.points = [start_point, end_point]
            marker_array.markers.append(strip_marker)
        
        self.markers_pub.publish(marker_array)
    
    def _log_progress(self):
        if self.coverage_start_time is None:
            return
        
        elapsed = (rospy.Time.now() - self.coverage_start_time).to_sec()
        success_rate = (self.successful_moves / self.total_moves)*100 if self.total_moves > 0 else 0
        completed_strips = sum(1 for status in self.strip_status.values() if status == StripStatus.COMPLETED)
        total_cov = sum(self.strip_coverage.values()) / len(self.strips) if self.strips else 0
        
        rospy.loginfo(f"Progress: {completed_strips}/{len(self.strips)} strips, {total_cov:.1f}% avg coverage, "
                      f"{self.successful_moves}/{self.total_moves} moves ({success_rate:.1f}%), {elapsed:.0f}s elapsed")
    
    def _log_final_statistics(self):
        if self.coverage_start_time is None:
            return
        
        duration = (rospy.Time.now() - self.coverage_start_time).to_sec()
        success_rate = (self.successful_moves / self.total_moves)*100 if self.total_moves > 0 else 0
        completed_strips = sum(1 for s in self.strip_status.values() if s == StripStatus.COMPLETED)
        total_cov = sum(self.strip_coverage.values())/len(self.strips) if self.strips else 0
        
        rospy.loginfo("="*60)
        rospy.loginfo("FINAL COVERAGE STATISTICS")
        rospy.loginfo("="*60)
        rospy.loginfo(f"Total time: {duration:.1f} seconds")
        rospy.loginfo(f"Strips completed: {completed_strips}/{len(self.strips)}")
        rospy.loginfo(f"Average coverage: {total_cov:.1f}%")
        rospy.loginfo(f"Total moves: {self.total_moves}")
        rospy.loginfo(f"Successful moves: {self.successful_moves}")
        rospy.loginfo(f"Success rate: {success_rate:.1f}%")
        rospy.loginfo(f"Positions visited: {len(self.visited_positions)}")
        rospy.loginfo(f"Incomplete strips: {len(self.incomplete_strips)}")
        rospy.loginfo("="*60)
    
    def start_coverage_service(self, req):
        if self.is_executing:
            return TriggerResponse(success=False, message="Coverage already in progress")
        if self.current_pose is None or self.occupancy_map is None:
            return TriggerResponse(success=False, message="Required data not available")
        
        coverage_thread = Thread(target=self.execute_intelligent_coverage)
        coverage_thread.daemon = True
        coverage_thread.start()
        
        return TriggerResponse(success=True, message="Coverage started successfully")
    
    def stop_coverage_service(self, req):
        if not self.is_executing:
            return TriggerResponse(success=False, message="No coverage in progress")
        
        self.is_executing = False
        self.move_base_client.cancel_all_goals()
        
        stop_twist = Twist()
        self.cmd_vel_pub.publish(stop_twist)
        
        return TriggerResponse(success=True, message="Coverage stopped successfully")
    
    def generate_path_service(self, req):
        if self.occupancy_map is None or self.current_pose is None:
            return TriggerResponse(success=False, message="Required data not available")
        
        if self._initialize_strips():
            return TriggerResponse(success=True, message=f"Generated path with {len(self.strips)} strips")
        else:
            return TriggerResponse(success=False, message="Failed to generate coverage path")
    
    def get_progress_service(self, req):
        if not self.strips:
            return TriggerResponse(success=False, message="No coverage plan available")
        
        completed_strips = sum(1 for s in self.strip_status.values() if s == StripStatus.COMPLETED)
        total_cov = sum(self.strip_coverage.values())/len(self.strips) if self.strips else 0
        
        elapsed = 0
        if self.coverage_start_time is not None:
            elapsed = (rospy.Time.now() - self.coverage_start_time).to_sec()
        
        msg = (f"Strips: {completed_strips}/{len(self.strips)}, "
               f"Coverage: {total_cov:.1f}%, "
               f"Moves: {self.successful_moves}/{self.total_moves}, "
               f"Time: {elapsed:.1f}s, "
               f"State: {self.coverage_state.name}")
        
        return TriggerResponse(success=True, message=msg)
    
    def run(self):
        rospy.loginfo("AMCL Intelligent Coverage Planner ready")
        rospy.loginfo("Available services:")
        rospy.loginfo("  - /coverage_planner/start_coverage")
        rospy.loginfo("  - /coverage_planner/stop_coverage")
        rospy.loginfo("  - /coverage_planner/generate_path")
        rospy.loginfo("  - /coverage_planner/get_progress")
        
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Shutting down coverage planner")
            if self.is_executing:
                self.is_executing = False
                self.move_base_client.cancel_all_goals()

if __name__ == '__main__':
    try:
        planner = AMCLIntelligentCoveragePlanner()
        planner.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Coverage planner node terminated")
    except Exception as e:
        rospy.logerr(f"Coverage planner error: {str(e)}")
        import traceback
        traceback.print_exc()


