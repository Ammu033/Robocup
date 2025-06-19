#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from geometry_msgs.msg import Point, PointStamped
from cv_bridge import CvBridge
import tf2_ros
import tf2_geometry_msgs
from tiago_auto.srv import ObjectFloorPose, ObjectFloorPoseRequest, ObjectFloorPoseResponse
import sensor_msgs.point_cloud2 as pc2
from image_geometry import PinholeCameraModel

class TIAGoFloorObjectDetection:
    def __init__(self):
        rospy.init_node('tiago_floor_object_detection_service')
        
        # Initialize CV bridge for image processing
        self.bridge = CvBridge()
        
        # TF2 buffer for coordinate transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Camera model for 3D projection
        self.camera_model = PinholeCameraModel()
        
        # Subscribers for TIAGo sensor data
        self.rgb_sub = rospy.Subscriber('/xtion/rgb/image_raw', Image, self.rgb_callback)
        self.depth_sub = rospy.Subscriber('/xtion/depth_registered/image_raw', Image, self.depth_callback)
        self.camera_info_sub = rospy.Subscriber('/xtion/rgb/camera_info', CameraInfo, self.camera_info_callback)
        
        # Store latest sensor data
        self.latest_rgb_image = None
        self.latest_depth_image = None
        self.camera_info = None
        self.camera_info_received = False
        
        # Object detection parameters
        self.floor_detection_params = {
            'min_contour_area': 500,  # Minimum area for object detection
            'max_contour_area': 50000,  # Maximum area to avoid detecting large surfaces
            'blur_kernel': (5, 5),
            'threshold_value': 100,
            'morph_kernel_size': 5
        }
        
        # Service server
        self.service = rospy.Service('get_object_floor_poses', ObjectFloorPose, self.handle_floor_detection)
        
        rospy.loginfo("TIAGo Floor Object Detection Service started")
    
    def rgb_callback(self, msg):
        """Store the latest RGB image"""
        try:
            self.latest_rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"Error converting RGB image: {e}")
    
    def depth_callback(self, msg):
        """Store the latest depth image"""
        try:
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        except Exception as e:
            rospy.logerr(f"Error converting depth image: {e}")
    
    def camera_info_callback(self, msg):
        """Store camera info and setup camera model"""
        if not self.camera_info_received:
            self.camera_info = msg
            self.camera_model.fromCameraInfo(msg)
            self.camera_info_received = True
            rospy.loginfo("Camera info received and camera model initialized")
    
    def detect_floor_objects(self, rgb_image, depth_image, z_cutoff):
        """
        Detect objects on the floor using RGB-D data
        Returns list of (x, y, depth, object_type, confidence) tuples in camera coordinates
        """
        if rgb_image is None or depth_image is None:
            return []
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, self.floor_detection_params['blur_kernel'], 0)
        
        # Apply adaptive threshold to find objects
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Morphological operations to clean up the image
        kernel = np.ones((self.floor_detection_params['morph_kernel_size'], 
                         self.floor_detection_params['morph_kernel_size']), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_objects = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if (area < self.floor_detection_params['min_contour_area'] or 
                area > self.floor_detection_params['max_contour_area']):
                continue
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Get center point
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Get depth at center point
            if (center_y < depth_image.shape[0] and center_x < depth_image.shape[1]):
                depth_value = depth_image[center_y, center_x]
                
                # Skip if depth is invalid or NaN
                if np.isnan(depth_value) or depth_value <= 0:
                    continue
                
                # Convert depth from mm to meters (if needed)
                if depth_value > 10:  # Assume values > 10 are in mm
                    depth_value = depth_value / 1000.0
                
                # Check if object is within z_cutoff from floor
                # Assuming camera is at ~1.2m height, floor objects should be at depth with z < z_cutoff
                camera_height = 1.2  # Approximate TIAGo camera height
                estimated_object_height = camera_height - depth_value * np.sin(np.radians(30))  # Rough estimate
                
                if estimated_object_height <= z_cutoff:
                    # Classify object type (simple classification based on area/shape)
                    object_type = self.classify_object(contour, area)
                    confidence = min(1.0, area / 10000.0)  # Simple confidence based on area
                    
                    detected_objects.append((center_x, center_y, depth_value, object_type, confidence))
        
        return detected_objects
    
    def classify_object(self, contour, area):
        """
        Simple object classification based on contour properties
        """
        # Calculate aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        
        # Calculate extent (contour area / bounding rectangle area)
        rect_area = w * h
        extent = float(area) / rect_area
        
        # Simple classification rules
        if area < 1000:
            return "small_trash"
        elif area < 5000:
            if aspect_ratio > 2.0 or aspect_ratio < 0.5:
                return "bottle"
            else:
                return "can"
        else:
            return "large_trash"
    
    def transform_to_map_frame(self, camera_points):
        """
        Transform points from camera frame to map frame
        """
        map_points = []
        
        try:
            # Get transform from camera frame to map frame
            transform = self.tf_buffer.lookup_transform('map', 'xtion_rgb_optical_frame', 
                                                      rospy.Time(), rospy.Duration(1.0))
            
            for x_cam, y_cam, z_cam, obj_type, confidence in camera_points:
                # Convert pixel coordinates to 3D camera coordinates
                if self.camera_info_received:
                    # Use camera model to get 3D point
                    ray = self.camera_model.projectPixelTo3dRay((x_cam, y_cam))
                    
                    # Create point in camera frame
                    point_camera = PointStamped()
                    point_camera.header.frame_id = 'xtion_rgb_optical_frame'
                    point_camera.header.stamp = rospy.Time.now()
                    point_camera.point.x = ray[0] * z_cam
                    point_camera.point.y = ray[1] * z_cam
                    point_camera.point.z = z_cam
                    
                    # Transform to map frame
                    point_map = tf2_geometry_msgs.do_transform_point(point_camera, transform)
                    
                    map_points.append((point_map.point, obj_type, confidence))
        
        except Exception as e:
            rospy.logerr(f"Error transforming points to map frame: {e}")
        
        return map_points
    
    def handle_floor_detection(self, req):
        """
        Handle the service request for floor object detection
        """
        response = ObjectFloorPoseResponse()
        
        try:
            # Check if we have sensor data
            if self.latest_rgb_image is None or self.latest_depth_image is None:
                rospy.logwarn("No sensor data available")
                response.floor_poses = []
                return response
            
            if not self.camera_info_received:
                rospy.logwarn("Camera info not received")
                response.floor_poses = []
                return response
            
            rospy.loginfo(f"Starting floor object detection with z_cutoff: {req.z_cutoff}")
            
            # Detect objects in camera coordinates
            camera_detections = self.detect_floor_objects(
                self.latest_rgb_image, 
                self.latest_depth_image, 
                req.z_cutoff
            )
            
            if not camera_detections:
                rospy.loginfo("No floor objects detected")
                response.floor_poses = []
                return response
            
            # Transform to map frame
            map_detections = self.transform_to_map_frame(camera_detections)
            
            # Fill response with only floor poses
            response.floor_poses = []
            
            for point_map, obj_type, confidence in map_detections:
                response.floor_poses.append(point_map)
            
            rospy.loginfo(f"Floor detection completed: {len(map_detections)} objects found")
            
        except Exception as e:
            rospy.logerr(f"Error in floor detection service: {e}")
            response.floor_poses = []
        
        return response
    
    def run(self):
        """Keep the service running"""
        rospy.loginfo("TIAGo Floor Object Detection Service is ready")
        rospy.spin()

if __name__ == '__main__':
    try:
        service = TIAGoFloorObjectDetection()
        service.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Floor Object Detection Service interrupted")
    except Exception as e:
        rospy.logerr(f"Error starting service: {e}")