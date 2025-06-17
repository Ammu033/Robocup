#!/usr/bin/env python3

import rospy
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from std_srvs.srv import Trigger, TriggerResponse
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray
import math
import numpy as np
from threading import Lock

class CoverageTrackerVisualization:
    def __init__(self):
        rospy.init_node('coverage_tracker_node', anonymous=True)

        # --- Parameters ---
        self.robot_cleaning_radius = rospy.get_param('~robot_cleaning_radius', 0.3) 
        self.coverage_map_resolution = rospy.get_param('~coverage_map_resolution', 0.1) 
        self.free_space_threshold = rospy.get_param('~free_space_threshold', 50)
        self.min_movement_threshold = rospy.get_param('~min_movement_threshold', 0.05)
        
        # Topic parameters - make them configurable
        self.map_topic = rospy.get_param('~map_topic', '/map')
        self.pose_topic = rospy.get_param('~pose_topic', '/amcl_pose')
        
        # Visualization parameters
        self.publish_markers = rospy.get_param('~publish_markers', True)
        self.marker_lifetime = rospy.get_param('~marker_lifetime', 0.0)  # 0 = permanent

        # State variables
        self.occupancy_map = None 
        self.coverage_grid = None
        self.last_robot_position = None
        self.map_received = False
        self.robot_pose_received = False
        self.data_lock = Lock()

        # Coverage statistics
        self.total_cleanable_area = 0
        self.covered_area = 0

        # Debug counters
        self.map_msg_count = 0
        self.pose_msg_count = 0

        # --- Subscribers ---
        rospy.loginfo(f"Subscribing to map topic: {self.map_topic}")
        
        rospy.Subscriber(self.map_topic, OccupancyGrid, self.map_callback)
        
        # Subscribe to AMCL pose (PoseWithCovarianceStamped)
        rospy.loginfo("Subscribing to /amcl_pose as PoseWithCovarianceStamped")
        rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.robot_pose_cov_callback)
        
        # Also try the configured pose topic as backup
        if self.pose_topic != "/amcl_pose":
            rospy.loginfo(f"Also subscribing to {self.pose_topic} as PoseStamped")
            rospy.Subscriber(self.pose_topic, PoseStamped, self.robot_pose_callback)

        # --- Publishers ---
        self.coverage_map_publisher = rospy.Publisher("/coverage_map", OccupancyGrid, queue_size=1)
        
        if self.publish_markers:
            self.coverage_markers_publisher = rospy.Publisher("/coverage_markers", MarkerArray, queue_size=1)
            self.robot_trail_publisher = rospy.Publisher("/robot_trail", Marker, queue_size=1)
        
        # Robot trail for visualization
        self.robot_trail_points = []

        # --- Services ---
        rospy.Service("/coverage_tracker/get_coverage_percentage", Trigger, self.get_coverage_percentage)
        rospy.Service("/coverage_tracker/reset_coverage", Trigger, self.reset_coverage)

        # Start diagnostics timer
        rospy.Timer(rospy.Duration(5.0), self.diagnostics_callback)

        rospy.loginfo("Coverage Tracker with Visualization initialized.")
        
        # Print available topics for debugging
        self.print_available_topics()

    def print_available_topics(self):
        """Print available topics for debugging."""
        try:
            import subprocess
            result = subprocess.run(['rostopic', 'list'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                topics = result.stdout.strip().split('\n')
                rospy.loginfo("Available topics:")
                for topic in sorted(topics):
                    if any(keyword in topic.lower() for keyword in ['map', 'pose', 'odom', 'amcl']):
                        rospy.loginfo(f"  - {topic}")
            else:
                rospy.logwarn("Could not list topics")
        except Exception as e:
            rospy.logwarn(f"Error listing topics: {e}")

    def diagnostics_callback(self, event):
        """Print diagnostic information."""
        rospy.loginfo(f"Diagnostics - Map msgs: {self.map_msg_count}, "
                     f"Pose msgs: {self.pose_msg_count}, "
                     f"Map received: {self.map_received}, "
                     f"Robot pose received: {self.robot_pose_received}")

    def map_callback(self, msg):
        """Receives the static occupancy grid map."""
        self.map_msg_count += 1
        rospy.loginfo(f"Map message #{self.map_msg_count} received - Size: {msg.info.width}x{msg.info.height}")
        
        with self.data_lock:
            if not self.map_received:
                self.occupancy_map = msg
                self.map_received = True
                rospy.loginfo("Occupancy map processed and stored.")
                self._initialize_coverage_grid()

    def robot_pose_callback(self, msg):
        """Handle PoseStamped messages."""
        self.pose_msg_count += 1
        if self.pose_msg_count % 10 == 1:  # Log every 10th message
            rospy.loginfo(f"Pose message #{self.pose_msg_count} received from {msg.header.frame_id}")
        
        self._process_robot_pose(msg.pose.position.x, msg.pose.position.y, msg.header.frame_id)

    def robot_pose_cov_callback(self, msg):
        """Handle PoseWithCovarianceStamped messages from AMCL."""
        self.pose_msg_count += 1
        if self.pose_msg_count == 1 or self.pose_msg_count % 50 == 0:  # Log first and every 50th message
            rospy.loginfo(f"AMCL pose message #{self.pose_msg_count} received from {msg.header.frame_id} "
                         f"at position ({msg.pose.pose.position.x:.2f}, {msg.pose.pose.position.y:.2f})")
        
        self._process_robot_pose(msg.pose.pose.position.x, msg.pose.pose.position.y, msg.header.frame_id)

    def robot_odom_callback(self, msg):
        """Handle Odometry messages."""
        self.pose_msg_count += 1
        if self.pose_msg_count % 10 == 1:  # Log every 10th message
            rospy.loginfo(f"Odometry message #{self.pose_msg_count} received from {msg.header.frame_id}")
        
        self._process_robot_pose(msg.pose.pose.position.x, msg.pose.pose.position.y, msg.header.frame_id)

    def _process_robot_pose(self, x, y, frame_id):
        """Process robot pose regardless of message type."""
        with self.data_lock:
            if self.coverage_grid is None:
                if self.pose_msg_count % 20 == 1:  # Don't spam this warning
                    rospy.logwarn("Received robot pose but coverage grid not initialized yet")
                return

            # Check movement threshold
            if self.last_robot_position is not None:
                dx = x - self.last_robot_position[0]
                dy = y - self.last_robot_position[1]
                distance = math.sqrt(dx*dx + dy*dy)
                if distance < self.min_movement_threshold:
                    return
                
                if self.pose_msg_count % 100 == 0:  # Occasional movement logging
                    rospy.loginfo(f"Robot moved {distance:.3f}m to ({x:.2f}, {y:.2f})")
            
            self.robot_pose_received = True
            self.last_robot_position = (x, y)
            
            # Add to robot trail
            self.robot_trail_points.append((x, y))
            if len(self.robot_trail_points) > 1000:  # Limit trail length
                self.robot_trail_points.pop(0)
            
            # Update coverage
            self._update_coverage_at_position(x, y)

    def _initialize_coverage_grid(self):
        """Initializes the coverage grid with proper visualization values."""
        if self.occupancy_map is None:
            return

        self.coverage_grid = OccupancyGrid()
        self.coverage_grid.header.frame_id = self.occupancy_map.header.frame_id
        
        # Calculate dimensions
        old_width = self.occupancy_map.info.width
        old_height = self.occupancy_map.info.height
        old_res = self.occupancy_map.info.resolution

        new_width = int(old_width * old_res / self.coverage_map_resolution)
        new_height = int(old_height * old_res / self.coverage_map_resolution)

        self.coverage_grid.info.resolution = self.coverage_map_resolution
        self.coverage_grid.info.width = new_width
        self.coverage_grid.info.height = new_height
        self.coverage_grid.info.origin = self.occupancy_map.info.origin

        # Initialize with better visualization values
        coverage_data = np.full(new_width * new_height, -1, dtype=np.int8)
        
        # Mark obstacles and calculate total cleanable area
        self.total_cleanable_area = 0
        for y in range(new_height):
            for x in range(new_width):
                world_x = (self.coverage_grid.info.origin.position.x + 
                          x * self.coverage_map_resolution)
                world_y = (self.coverage_grid.info.origin.position.y + 
                          y * self.coverage_map_resolution)
                
                if self._is_obstacle_at_world_coord(world_x, world_y):
                    coverage_data[y * new_width + x] = 0  # Obstacle
                else:
                    self.total_cleanable_area += 1
        
        self.coverage_grid.data = coverage_data.tolist()
        self.covered_area = 0
        
        # Reset robot trail
        self.robot_trail_points = []

        rospy.loginfo(f"Coverage grid initialized: {new_width}x{new_height}, "
                      f"Total cleanable area: {self.total_cleanable_area} cells")

    def _is_obstacle_at_world_coord(self, world_x, world_y):
        """Check if a world coordinate is an obstacle."""
        if self.occupancy_map is None:
            return False
            
        orig_x = int((world_x - self.occupancy_map.info.origin.position.x) / 
                     self.occupancy_map.info.resolution)
        orig_y = int((world_y - self.occupancy_map.info.origin.position.y) / 
                     self.occupancy_map.info.resolution)
        
        if (0 <= orig_x < self.occupancy_map.info.width and 
            0 <= orig_y < self.occupancy_map.info.height):
            orig_index = orig_y * self.occupancy_map.info.width + orig_x
            return self.occupancy_map.data[orig_index] >= self.free_space_threshold
        
        return True

    def _update_coverage_at_position(self, robot_x, robot_y):
        """Update coverage with better visualization values."""
        map_x_px = int((robot_x - self.coverage_grid.info.origin.position.x) / 
                       self.coverage_map_resolution)
        map_y_px = int((robot_y - self.coverage_grid.info.origin.position.y) / 
                       self.coverage_map_resolution)

        radius_px = int(self.robot_cleaning_radius / self.coverage_map_resolution)
        newly_covered = 0

        for dx in range(-radius_px, radius_px + 1):
            for dy in range(-radius_px, radius_px + 1):
                if dx*dx + dy*dy <= radius_px*radius_px:
                    cell_x = map_x_px + dx
                    cell_y = map_y_px + dy

                    if (0 <= cell_x < self.coverage_grid.info.width and 
                        0 <= cell_y < self.coverage_grid.info.height):
                        
                        index = cell_y * self.coverage_grid.info.width + cell_x
                        
                        if self.coverage_grid.data[index] == -1:  # Unvisited
                            world_x = (self.coverage_grid.info.origin.position.x + 
                                     cell_x * self.coverage_map_resolution)
                            world_y = (self.coverage_grid.info.origin.position.y + 
                                     cell_y * self.coverage_map_resolution)
                            
                            if not self._is_obstacle_at_world_coord(world_x, world_y):
                                self.coverage_grid.data[index] = 75  # Visited
                                newly_covered += 1
        
        self.covered_area += newly_covered

    def _create_coverage_markers(self):
        """Create visualization markers for coverage statistics."""
        marker_array = MarkerArray()
        
        # Coverage percentage text
        text_marker = Marker()
        text_marker.header.frame_id = self.coverage_grid.header.frame_id
        text_marker.header.stamp = rospy.Time.now()
        text_marker.id = 0
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        
        # Position text in upper corner of map
        text_marker.pose.position.x = self.coverage_grid.info.origin.position.x + 1.0
        text_marker.pose.position.y = self.coverage_grid.info.origin.position.y + \
                                      self.coverage_grid.info.height * self.coverage_map_resolution - 1.0
        text_marker.pose.position.z = 0.5
        text_marker.pose.orientation.w = 1.0
        
        # Calculate percentage
        if self.total_cleanable_area > 0:
            percentage = (self.covered_area / self.total_cleanable_area) * 100.0
        else:
            percentage = 0.0
            
        text_marker.text = f"Coverage: {percentage:.1f}%\n({self.covered_area}/{self.total_cleanable_area})"
        text_marker.scale.z = 0.5
        text_marker.color.r = 1.0
        text_marker.color.g = 1.0
        text_marker.color.b = 1.0
        text_marker.color.a = 1.0
        text_marker.lifetime = rospy.Duration(self.marker_lifetime)
        
        marker_array.markers.append(text_marker)
        return marker_array

    def _create_robot_trail_marker(self):
        """Create a line strip showing robot's path."""
        if len(self.robot_trail_points) < 2:
            return None
            
        trail_marker = Marker()
        trail_marker.header.frame_id = self.coverage_grid.header.frame_id
        trail_marker.header.stamp = rospy.Time.now()
        trail_marker.id = 0
        trail_marker.type = Marker.LINE_STRIP
        trail_marker.action = Marker.ADD
        
        trail_marker.scale.x = 0.05  # Line width
        trail_marker.color.r = 0.0
        trail_marker.color.g = 0.0
        trail_marker.color.b = 1.0  # Blue trail
        trail_marker.color.a = 0.8
        trail_marker.lifetime = rospy.Duration(self.marker_lifetime)
        
        # Add all trail points
        from geometry_msgs.msg import Point
        for x, y in self.robot_trail_points:
            point = Point()
            point.x = x
            point.y = y
            point.z = 0.1
            trail_marker.points.append(point)
            
        return trail_marker

    def get_coverage_percentage(self, req):
        """Service to return coverage percentage."""
        response = TriggerResponse()
        
        with self.data_lock:
            if self.coverage_grid is None:
                response.success = False
                response.message = "Coverage map not initialized."
                return response

            if self.total_cleanable_area > 0:
                percentage = (self.covered_area / self.total_cleanable_area) * 100.0
            else:
                percentage = 0.0

            response.success = True
            response.message = f"Coverage: {percentage:.2f}% ({self.covered_area}/{self.total_cleanable_area} cells)"
            return response

    def reset_coverage(self, req):
        """Reset coverage map."""
        response = TriggerResponse()
        
        with self.data_lock:
            if self.coverage_grid is None:
                response.success = False
                response.message = "Coverage map not initialized."
                return response
            
            self._initialize_coverage_grid()
            response.success = True
            response.message = "Coverage map reset successfully."
            return response

    def run(self):
        rate = rospy.Rate(2.0)
        publish_count = 0
        while not rospy.is_shutdown():
            with self.data_lock:
                if self.coverage_grid is not None:
                    # Always publish coverage map
                    self.coverage_grid.header.stamp = rospy.Time.now()
                    self.coverage_map_publisher.publish(self.coverage_grid)
                    
                    publish_count += 1
                    if publish_count % 10 == 0:  # Log every 5 seconds
                        rospy.loginfo(f"Published coverage map #{publish_count}. "
                                    f"Robot pose received: {self.robot_pose_received}, "
                                    f"Covered area: {self.covered_area}/{self.total_cleanable_area}")
                    
                    # Publish visualization markers
                    if self.publish_markers and self.robot_pose_received:
                        markers = self._create_coverage_markers()
                        if markers:
                            self.coverage_markers_publisher.publish(markers)
                        
                        trail = self._create_robot_trail_marker()
                        if trail:
                            self.robot_trail_publisher.publish(trail)
                elif publish_count % 10 == 0:
                    rospy.logwarn("Coverage grid not initialized yet - waiting for map")
                            
            rate.sleep()

if __name__ == '__main__':
    try:
        tracker = CoverageTrackerVisualization()
        tracker.run()
    except rospy.ROSInterruptException:
        pass