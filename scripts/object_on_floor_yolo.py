#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, PointStamped
from cv_bridge import CvBridge
import tf2_ros
import tf2_geometry_msgs
from tiago_auto.srv import ObjectFloorPose, ObjectFloorPoseRequest, ObjectFloorPoseResponse
from image_geometry import PinholeCameraModel

# YOLO imports - Initialize YOLO_AVAILABLE first
YOLO_AVAILABLE = False

try:
    import ultralytics
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("YOLO (ultralytics) is available")
except ImportError:
    print("YOLO not available, falling back to OpenCV detection")

class TIAGoYOLOFloorDetection:
    def __init__(self):
        rospy.init_node('tiago_yolo_floor_detection_service')
        
        # Initialize CV bridge for image processing
        self.bridge = CvBridge()
        
        # TF2 buffer for coordinate transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Camera model for 3D projection
        self.camera_model = PinholeCameraModel()
        
        # Initialize YOLO model
        self.yolo_model = None
        global YOLO_AVAILABLE
        
        if YOLO_AVAILABLE:
            try:
                # Load YOLOv8 model (you can also use 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt' for better accuracy)
                self.yolo_model = YOLO('yolov8n.pt')  # nano version for speed
                rospy.loginfo("YOLO model loaded successfully")
            except Exception as e:
                rospy.logerr(f"Failed to load YOLO model: {e}")
                YOLO_AVAILABLE = False
        else:
            rospy.logwarn("YOLO not available, using OpenCV fallback")
        
        # Subscribers for TIAGo sensor data
        self.rgb_sub = rospy.Subscriber('/xtion/rgb/image_raw', Image, self.rgb_callback)
        self.depth_sub = rospy.Subscriber('/xtion/depth_registered/image_raw', Image, self.depth_callback)
        self.camera_info_sub = rospy.Subscriber('/xtion/rgb/camera_info', CameraInfo, self.camera_info_callback)
        
        # Store latest sensor data
        self.latest_rgb_image = None
        self.latest_depth_image = None
        self.camera_info = None
        self.camera_info_received = False
        
        # Define object classes that could be on the floor (COCO dataset classes)
        self.floor_object_classes = {
            39: 'bottle',           # bottle
            41: 'cup',              # cup
            42: 'fork',             # fork
            43: 'knife',            # knife
            44: 'spoon',            # spoon
            45: 'bowl',             # bowl
            46: 'banana',           # banana
            47: 'apple',            # apple
            48: 'sandwich',         # sandwich
            49: 'orange',           # orange
            50: 'broccoli',         # broccoli
            51: 'carrot',           # carrot
            52: 'hot dog',          # hot dog
            53: 'pizza',            # pizza
            54: 'donut',            # donut
            55: 'cake',             # cake
            64: 'mouse',            # computer mouse
            65: 'remote',           # remote
            66: 'keyboard',         # keyboard
            67: 'cell phone',       # cell phone
            73: 'book',             # book
            74: 'clock',            # clock
            75: 'vase',             # vase
            76: 'scissors',         # scissors
            77: 'teddy bear',       # teddy bear
            78: 'hair drier',       # hair drier
            79: 'toothbrush',       # toothbrush
            # Add more classes as needed
        }
        
        # Detection parameters
        self.detection_params = {
            'confidence_threshold': 0.3,    # Minimum confidence for detection
            'height_threshold': 0.5,        # Maximum height above floor (meters)
            'min_object_size': 20,          # Minimum bounding box size (pixels)
        }
        
        # Service server
        self.service = rospy.Service('get_object_floor_poses', ObjectFloorPose, self.handle_floor_detection)
        
        rospy.loginfo("TIAGo YOLO Floor Object Detection Service started")
    
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
    
    def detect_objects_yolo(self, rgb_image, depth_image, z_cutoff):
        """
        Detect objects using YOLO and filter for floor objects
        Returns list of (x, y, depth, object_type, confidence) tuples
        """
        global YOLO_AVAILABLE
        
        if not YOLO_AVAILABLE or self.yolo_model is None:
            return self.detect_objects_opencv_fallback(rgb_image, depth_image, z_cutoff)
        
        detected_objects = []
        
        try:
            # Run YOLO inference
            results = self.yolo_model(rgb_image, conf=self.detection_params['confidence_threshold'])
            
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Check if this is a floor object class
                    if class_id not in self.floor_object_classes:
                        continue
                    
                    # Check minimum size
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    if (bbox_width < self.detection_params['min_object_size'] or 
                        bbox_height < self.detection_params['min_object_size']):
                        continue
                    
                    # Get center point
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    # Get depth at center point (with some averaging for stability)
                    depth_value = self.get_average_depth(depth_image, center_x, center_y, 5)
                    
                    if depth_value is None or depth_value <= 0:
                        continue
                    
                    # Convert depth from mm to meters if needed
                    if depth_value > 10:
                        depth_value = depth_value / 1000.0
                    
                    # Estimate object height above floor
                    camera_height = 1.2  # Approximate TIAGo camera height
                    camera_tilt_angle = np.radians(-30)  # Camera looking down
                    
                    # Calculate approximate object height
                    estimated_height = self.estimate_object_height(
                        center_y, depth_value, camera_height, camera_tilt_angle
                    )
                    
                    # Check if object is on the floor
                    if estimated_height <= z_cutoff:
                        object_name = self.floor_object_classes[class_id]
                        detected_objects.append((center_x, center_y, depth_value, object_name, confidence))
                        
                        rospy.loginfo(f"Detected {object_name} at ({center_x}, {center_y}) "
                                    f"with confidence {confidence:.2f}, estimated height: {estimated_height:.2f}m")
        
        except Exception as e:
            rospy.logerr(f"Error in YOLO detection: {e}")
            return self.detect_objects_opencv_fallback(rgb_image, depth_image, z_cutoff)
        
        return detected_objects
    
    def get_average_depth(self, depth_image, center_x, center_y, window_size):
        """Get average depth in a small window around the center point"""
        try:
            h, w = depth_image.shape
            half_window = window_size // 2
            
            y_start = max(0, center_y - half_window)
            y_end = min(h, center_y + half_window + 1)
            x_start = max(0, center_x - half_window)
            x_end = min(w, center_x + half_window + 1)
            
            depth_window = depth_image[y_start:y_end, x_start:x_end]
            valid_depths = depth_window[~np.isnan(depth_window) & (depth_window > 0)]
            
            if len(valid_depths) > 0:
                return np.median(valid_depths)  # Use median for robustness
            else:
                return None
        except:
            return None
    
    def estimate_object_height(self, pixel_y, depth, camera_height, camera_tilt):
        """
        Estimate object height above floor using camera geometry
        """
        try:
            if not self.camera_info_received:
                return 0.0
            
            # Get camera intrinsics
            fy = self.camera_info.K[4]  # Focal length in y
            cy = self.camera_info.K[5]  # Principal point y
            
            # Calculate angle to object
            angle_to_object = np.arctan((pixel_y - cy) / fy) + camera_tilt
            
            # Calculate height above floor
            horizontal_distance = depth * np.cos(angle_to_object)
            object_height_from_camera = depth * np.sin(angle_to_object)
            object_height_above_floor = camera_height + object_height_from_camera
            
            return max(0, object_height_above_floor)
        except:
            return 0.0
    
    def detect_objects_opencv_fallback(self, rgb_image, depth_image, z_cutoff):
        """
        Fallback OpenCV detection if YOLO is not available
        """
        rospy.loginfo("Using OpenCV fallback detection")
        
        # Convert to grayscale
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_objects = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < 500 or area > 50000:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            
            depth_value = self.get_average_depth(depth_image, center_x, center_y, 5)
            
            if depth_value is None or depth_value <= 0:
                continue
            
            if depth_value > 10:
                depth_value = depth_value / 1000.0
            
            estimated_height = self.estimate_object_height(center_y, depth_value, 1.2, np.radians(-30))
            
            if estimated_height <= z_cutoff:
                object_type = "unknown_object"
                confidence = min(1.0, area / 10000.0)
                detected_objects.append((center_x, center_y, depth_value, object_type, confidence))
        
        return detected_objects
    
    def transform_to_map_frame(self, camera_points):
        """
        Transform points from camera frame to map frame
        """
        map_points = []
        
        try:
            transform = self.tf_buffer.lookup_transform('map', 'xtion_rgb_optical_frame', 
                                                      rospy.Time(), rospy.Duration(1.0))
            
            for x_cam, y_cam, z_cam, obj_type, confidence in camera_points:
                if self.camera_info_received:
                    ray = self.camera_model.projectPixelTo3dRay((x_cam, y_cam))
                    
                    point_camera = PointStamped()
                    point_camera.header.frame_id = 'xtion_rgb_optical_frame'
                    point_camera.header.stamp = rospy.Time.now()
                    point_camera.point.x = ray[0] * z_cam
                    point_camera.point.y = ray[1] * z_cam
                    point_camera.point.z = z_cam
                    
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
            if self.latest_rgb_image is None or self.latest_depth_image is None:
                rospy.logwarn("No sensor data available")
                response.floor_poses = []
                return response
            
            if not self.camera_info_received:
                rospy.logwarn("Camera info not received")
                response.floor_poses = []
                return response
            
            rospy.loginfo(f"Starting YOLO floor object detection with z_cutoff: {req.z_cutoff}")
            
            # Detect objects using YOLO (or OpenCV fallback)
            camera_detections = self.detect_objects_yolo(
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
            
            # Fill response
            response.floor_poses = []
            for point_map, obj_type, confidence in map_detections:
                response.floor_poses.append(point_map)
            
            rospy.loginfo(f"YOLO detection completed: {len(map_detections)} objects found")
            
        except Exception as e:
            rospy.logerr(f"Error in YOLO floor detection service: {e}")
            response.floor_poses = []
        
        return response
    
    def run(self):
        """Keep the service running"""
        rospy.loginfo("TIAGo YOLO Floor Object Detection Service is ready")
        rospy.spin()

if __name__ == '__main__':
    try:
        service = TIAGoYOLOFloorDetection()
        service.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("YOLO Floor Object Detection Service interrupted")
    except Exception as e:
        rospy.logerr(f"Error starting service: {e}")