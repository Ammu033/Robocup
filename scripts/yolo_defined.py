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

class SmartTrashDetection:
    def __init__(self):
        rospy.init_node('tiago_smart_trash_detection_service')
        
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
                # Use YOLOv8n - good balance of speed and accuracy
                self.yolo_model = YOLO('yolov8n.pt')
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
        
        # COMPREHENSIVE TRASH CLASSIFICATION
        
        # DEFINITE TRASH when found on floor (no analysis needed)
        self.definite_trash_on_floor = {
            # === FOOD WASTE === (always trash when on floor)
            46: {'name': 'banana', 'priority': 'high', 'reason': 'food_waste', 'recyclable': True},
            47: {'name': 'apple', 'priority': 'high', 'reason': 'food_waste', 'recyclable': True},
            49: {'name': 'orange', 'priority': 'high', 'reason': 'food_waste', 'recyclable': True},
            50: {'name': 'broccoli', 'priority': 'high', 'reason': 'food_waste', 'recyclable': True},
            51: {'name': 'carrot', 'priority': 'high', 'reason': 'food_waste', 'recyclable': True},
            52: {'name': 'hot_dog', 'priority': 'high', 'reason': 'food_waste', 'recyclable': False},
            53: {'name': 'pizza', 'priority': 'high', 'reason': 'food_waste', 'recyclable': False},
            54: {'name': 'donut', 'priority': 'high', 'reason': 'food_waste', 'recyclable': False},
            48: {'name': 'sandwich', 'priority': 'high', 'reason': 'food_waste', 'recyclable': False},
            55: {'name': 'cake', 'priority': 'medium', 'reason': 'food_waste', 'recyclable': False},
        }
        
        # CONTAINER ITEMS (need context analysis - could be trash or in use)
        self.context_dependent_items = {
            # === BOTTLES & CANS ===
            39: {'name': 'bottle', 'priority': 'high', 'reason': 'container', 'recyclable': True, 'material': 'plastic_glass'},
            
            # === CUPS & CONTAINERS ===
            41: {'name': 'cup', 'priority': 'medium', 'reason': 'container', 'recyclable': True, 'material': 'paper_plastic'},
            45: {'name': 'bowl', 'priority': 'medium', 'reason': 'container', 'recyclable': False, 'material': 'ceramic_plastic'},
            
            # === BOOKS & PAPER PRODUCTS === (could be damaged/trash)
            73: {'name': 'book', 'priority': 'low', 'reason': 'paper', 'recyclable': True, 'material': 'paper'},  # Could be damaged book = trash
        }
        
        # ADDITIONAL TRASH CATEGORIES (detected via shape/color analysis)
        self.trash_patterns = {
            # === CARDBOARD & BOXES ===
            'cereal_box': {
                'detection_method': 'box_analysis',
                'priority': 'high',
                'recyclable': True,
                'material': 'cardboard',
                'shape_features': 'rectangular, cardboard_texture',
                'size_range': (800, 15000),  # pixel area range
                'aspect_ratio': (0.3, 3.0),  # width/height ratios for boxes
                'color_features': 'varied, printed_graphics'
            },
            'cardboard_box': {
                'detection_method': 'box_analysis',
                'priority': 'high',
                'recyclable': True,
                'material': 'cardboard',
                'shape_features': 'rectangular, brown_cardboard',
                'size_range': (1000, 25000),
                'aspect_ratio': (0.2, 5.0),
                'color_features': 'brown, tan'
            },
            'pizza_box': {
                'detection_method': 'box_analysis',
                'priority': 'high',
                'recyclable': True,  # Usually recyclable if not too greasy
                'material': 'cardboard',
                'shape_features': 'flat, square, large',
                'size_range': (2000, 20000),
                'aspect_ratio': (0.7, 1.4),  # Nearly square
                'color_features': 'white, brown'
            },
            'food_packaging': {
                'detection_method': 'box_analysis',
                'priority': 'medium',
                'recyclable': False,  # Mixed materials
                'material': 'mixed',
                'shape_features': 'small_rectangular',
                'size_range': (200, 2000),
                'aspect_ratio': (0.5, 2.5),
                'color_features': 'colorful_packaging'
            },
            
            # === FLEXIBLE PACKAGING ===
            'plastic_bag': {
                'detection_method': 'shape_analysis',
                'priority': 'high',
                'recyclable': False,
                'material': 'plastic',
                'shape_features': 'flexible, crumpled'
            },
            'chip_bag': {
                'detection_method': 'texture_color',
                'priority': 'medium',
                'recyclable': False,
                'material': 'foil_plastic',
                'shape_features': 'crinkled, reflective',
                'color_features': 'metallic, colorful'
            },
            'wrapper': {
                'detection_method': 'size_texture',
                'priority': 'medium',
                'recyclable': False,
                'material': 'plastic_foil',
                'shape_features': 'small, crinkled',
                'size_range': (50, 500)
            },
            
            # === METAL CONTAINERS ===
            'aluminum_can': {
                'detection_method': 'color_texture',
                'priority': 'high', 
                'recyclable': True,
                'material': 'aluminum',
                'color_features': 'metallic, cylindrical'
            },
            'tin_can': {
                'detection_method': 'color_texture',
                'priority': 'high',
                'recyclable': True,
                'material': 'steel',
                'color_features': 'metallic, cylindrical',
                'shape_features': 'shorter_wider_than_soda_can'
            },
            
            # === PAPER PRODUCTS ===
            'paper_trash': {
                'detection_method': 'texture_analysis',
                'priority': 'medium',
                'recyclable': True,
                'material': 'paper',
                'texture_features': 'flat, white_brown'
            },
            'newspaper': {
                'detection_method': 'texture_pattern',
                'priority': 'medium',
                'recyclable': True,
                'material': 'paper',
                'texture_features': 'text_pattern, thin_paper'
            },
            'magazine': {
                'detection_method': 'texture_color',
                'priority': 'low',
                'recyclable': True,
                'material': 'paper',
                'texture_features': 'glossy, colorful'
            },
            
            # === SMALL ITEMS ===
            'cigarette_butt': {
                'detection_method': 'size_color',
                'priority': 'low',
                'recyclable': False,
                'material': 'filter',
                'size_features': 'very_small, brown_white'
            },
            'straw': {
                'detection_method': 'shape_size',
                'priority': 'medium',
                'recyclable': False,
                'material': 'plastic',
                'shape_features': 'thin, cylindrical'
            },
            'bottle_cap': {
                'detection_method': 'size_shape',
                'priority': 'medium',
                'recyclable': True,
                'material': 'plastic_metal',
                'shape_features': 'small, circular'
            }
        }
        
        # NEVER consider these as trash (valuable/dangerous items)
        self.protected_items = {
            # === ELECTRONICS & DEVICES ===
            63: 'laptop',           # Expensive!
            64: 'computer_mouse',   # Expensive
            65: 'remote',          # Expensive
            66: 'keyboard',        # Expensive  
            67: 'cell_phone',      # VERY expensive!
            68: 'microwave',       # Appliance
            69: 'oven',           # Appliance
            70: 'toaster',        # Appliance
            71: 'sink',           # Fixture
            72: 'refrigerator',   # Large appliance
            
            # === PERSONAL & VALUABLE ITEMS ===
            73: 'book',           # Could be valuable (but check context)
            74: 'clock',          # Valuable
            75: 'vase',           # Fragile/valuable
            77: 'teddy_bear',     # Sentimental
            78: 'hair_drier',     # Personal item
            79: 'toothbrush',     # Personal hygiene
            
            # === UTENSILS & TOOLS ===
            42: 'fork',           # Utensil - not trash
            43: 'knife',          # Utensil - not trash (also dangerous!)
            44: 'spoon',          # Utensil - not trash
            76: 'scissors',       # Tool - dangerous/valuable
            
            # === SPORTS & RECREATION ===
            32: 'sports_ball',    # Could be expensive
            33: 'kite',          # Recreation item
            34: 'baseball_bat',   # Sports equipment
            35: 'baseball_glove', # Sports equipment
            36: 'skateboard',     # Expensive sports equipment
            37: 'surfboard',      # Expensive sports equipment
            38: 'tennis_racket',  # Sports equipment
            
            # === FURNITURE & LARGE ITEMS ===
            56: 'chair',          # Furniture
            57: 'couch',          # Furniture
            58: 'potted_plant',   # Living thing/decor
            59: 'bed',            # Furniture
            60: 'dining_table',   # Furniture
            61: 'toilet',         # Fixture
            62: 'tv',             # Expensive electronics
        }
        
        # Detection parameters - optimized for trash detection
        self.detection_params = {
            'confidence_threshold': 0.25,   # Lower threshold for trash (might be dirty/damaged)
            'height_threshold': 0.3,        # Must be close to floor
            'min_object_size': 15,          # Allow smaller objects (trash can be small)
        }
        
        # Service server
        self.service = rospy.Service('get_object_floor_poses', ObjectFloorPose, self.handle_trash_detection)
        
        rospy.loginfo("Smart Trash Detection Service started")
    
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
    
    def detect_smart_trash(self, rgb_image, depth_image, z_cutoff):
        """
        Smart trash detection that distinguishes between trash and valuable items
        """
        global YOLO_AVAILABLE
        
        if not YOLO_AVAILABLE or self.yolo_model is None:
            return self.detect_objects_opencv_fallback(rgb_image, depth_image, z_cutoff)
        
        detected_trash = []
        
        try:
            # Run YOLO inference
            results = self.yolo_model(rgb_image, conf=self.detection_params['confidence_threshold'])
            
            rospy.loginfo("=== SMART TRASH DETECTION ===")
            
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                
                rospy.loginfo(f"Analyzing {len(boxes)} detected objects...")
                
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Get class name
                    class_name = self.yolo_model.names[class_id] if hasattr(self.yolo_model, 'names') else f"class_{class_id}"
                    
                    rospy.loginfo(f"Found: {class_name} (ID: {class_id}, conf: {confidence:.2f})")
                    
                    # STEP 1: Check if this is a protected (valuable) item
                    if class_id in self.protected_items:
                        rospy.loginfo(f"  -> ✓ PROTECTED ITEM: {self.protected_items[class_id]} - NOT TRASH!")
                        continue
                    
                    # STEP 2: Check if it's definitely trash
                    is_trash = False
                    trash_info = None
                    
                    if class_id in self.definite_trash_on_floor:
                        is_trash = True
                        trash_info = self.definite_trash_on_floor[class_id]
                        rospy.loginfo(f"  -> ✓ DEFINITE TRASH: {trash_info['name']} ({trash_info['reason']})")
                    
                    # STEP 3: Check context-dependent items
                    elif class_id in self.context_dependent_items:
                        trash_info = self.context_dependent_items[class_id]
                        is_trash = self.analyze_container_context(rgb_image, x1, y1, x2, y2, trash_info['name'])
                        
                        if is_trash:
                            rospy.loginfo(f"  -> ✓ CONTEXT TRASH: {trash_info['name']} (appears discarded)")
                        else:
                            rospy.loginfo(f"  -> ✗ NOT TRASH: {trash_info['name']} (appears in use)")
                            continue
                    
                    else:
                        rospy.loginfo(f"  -> ✗ UNKNOWN CLASS: {class_name} - not in trash categories")
                        continue
                    
                    if not is_trash:
                        continue
                    
                    # STEP 4: Check object size
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    if (bbox_width < self.detection_params['min_object_size'] or 
                        bbox_height < self.detection_params['min_object_size']):
                        rospy.loginfo(f"  -> ✗ TOO SMALL: {bbox_width}x{bbox_height}")
                        continue
                    
                    # STEP 5: Get depth and check if on floor
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    depth_value = self.get_average_depth(depth_image, center_x, center_y, 5)
                    
                    if depth_value is None or depth_value <= 0:
                        rospy.loginfo(f"  -> ✗ INVALID DEPTH: {depth_value}")
                        continue
                    
                    # Convert depth from mm to meters if needed
                    if depth_value > 10:
                        depth_value = depth_value / 1000.0
                    
                    # STEP 6: Critical height check - must be on floor
                    estimated_height = self.estimate_object_height(
                        center_y, depth_value, 1.2, np.radians(-30)
                    )
                    
                    rospy.loginfo(f"  -> Height: {estimated_height:.2f}m (cutoff: {z_cutoff:.2f}m)")
                    
                    if estimated_height <= z_cutoff:
                        # THIS IS CONFIRMED TRASH ON THE FLOOR
                        trash_name = f"trash_{trash_info['name']}"
                        detected_trash.append((center_x, center_y, depth_value, trash_name, confidence))
                        
                        rospy.loginfo(f"  -> 🗑️  TRASH CONFIRMED: {trash_name} (priority: {trash_info['priority']})")
                    else:
                        rospy.loginfo(f"  -> ✗ TOO HIGH: {estimated_height:.2f}m - on table/shelf, not floor trash")
        
        except Exception as e:
            rospy.logerr(f"Error in smart trash detection: {e}")
            return self.detect_objects_opencv_fallback(rgb_image, depth_image, z_cutoff)
        
        rospy.loginfo(f"=== RESULT: {len(detected_trash)} trash items confirmed ===")
        return detected_trash
    
    def analyze_container_context(self, rgb_image, x1, y1, x2, y2, container_type):
        """
        Enhanced analysis for containers (bottles/cans/cups) to determine if trash
        """
        try:
            # Extract container region
            width = x2 - x1
            height = y2 - y1
            
            rospy.loginfo(f"    Analyzing {container_type}: {width:.0f}x{height:.0f}")
            
            # === ORIENTATION ANALYSIS ===
            # Rule 1: If lying on side, likely discarded
            if width > height * 1.2:
                rospy.loginfo(f"    -> Lying on side (w/h ratio: {width/height:.2f}) = TRASH")
                return True
            
            # === SIZE ANALYSIS ===
            # Rule 2: Very small containers likely discarded/crushed
            container_area = width * height
            if container_area < 400:
                rospy.loginfo(f"    -> Small/crushed ({container_area:.0f} pixels) = likely TRASH")
                return True
            
            # === COLOR & CONDITION ANALYSIS ===
            roi = rgb_image[int(y1):int(y2), int(x1):int(x2)]
            if roi.size > 0:
                # Convert to different color spaces for analysis
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                
                # Check for dirt/damage (dark colors)
                dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))
                dark_ratio = np.sum(dark_mask > 0) / dark_mask.size
                
                if dark_ratio > 0.4:
                    rospy.loginfo(f"    -> Dirty/damaged ({dark_ratio:.2f} dark) = TRASH")
                    return True
                
                # Check for metallic appearance (aluminum cans)
                if container_type == 'bottle':
                    # Look for metallic/reflective surfaces
                    edges = cv2.Canny(gray, 50, 150)
                    edge_density = np.sum(edges > 0) / edges.size
                    
                    if edge_density > 0.3:  # High edge density = metallic can
                        rospy.loginfo(f"    -> Metallic appearance (edges: {edge_density:.2f}) = likely ALUMINUM CAN")
                        # Check if crushed/dented
                        if width > height * 0.8:  # Not perfectly upright
                            return True
                
                # Check for transparency (plastic bottles)
                # High saturation = opaque, low saturation = transparent
                saturation = hsv[:,:,1]
                avg_saturation = np.mean(saturation)
                
                if avg_saturation < 50:  # Low saturation = transparent
                    rospy.loginfo(f"    -> Transparent plastic (sat: {avg_saturation:.0f})")
                    # Empty transparent bottles more likely trash
                    if container_area < 1000:
                        return True
            
            # === CONTEXT RULES FOR SPECIFIC CONTAINERS ===
            if container_type == 'bottle':
                # Bottles on floor are often trash unless clearly placed upright
                if height > width * 2:  # Tall and upright
                    rospy.loginfo(f"    -> Bottle upright and tall = might be in use")
                    return False
                else:
                    rospy.loginfo(f"    -> Bottle not upright = likely TRASH")
                    return True
                    
            elif container_type == 'cup':
                # Cups on floor more likely trash than bottles
                if container_area > 2000:  # Large cup, might be intentionally placed
                    rospy.loginfo(f"    -> Large cup = might be in use")
                    return False
                else:
                    rospy.loginfo(f"    -> Small cup on floor = likely TRASH")
                    return True
            
            elif container_type == 'bowl':
                # Bowls on floor usually trash (unless pet bowls, but those should be detected differently)
                rospy.loginfo(f"    -> Bowl on floor = likely TRASH")
                return True
            
            # Default: if uncertain and on floor, probably trash
            rospy.loginfo(f"    -> Default: container on floor = likely TRASH")
            return True
            
        except Exception as e:
            rospy.logerr(f"Error in container analysis: {e}")
            return False  # Conservative: don't assume it's trash
    
    def detect_additional_trash_patterns(self, rgb_image, depth_image, z_cutoff):
        """
        Detect additional trash using shape, color, and texture analysis
        """
        additional_trash = []
        
        try:
            # Convert to different color spaces
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
            
            # === DETECT CARDBOARD BOXES (CEREAL, PIZZA, SHIPPING) ===
            box_detections = self.detect_cardboard_boxes(rgb_image, gray, hsv, depth_image, z_cutoff)
            additional_trash.extend(box_detections)
            
            # === DETECT PLASTIC BAGS & WRAPPERS ===
            bag_detections = self.detect_flexible_packaging(gray, hsv, depth_image, z_cutoff)
            additional_trash.extend(bag_detections)
            
            # === DETECT ALUMINUM/TIN CANS ===
            can_detections = self.detect_metal_cans(gray, hsv, depth_image, z_cutoff)
            additional_trash.extend(can_detections)
            
            # === DETECT PAPER PRODUCTS ===
            paper_detections = self.detect_paper_trash(gray, hsv, depth_image, z_cutoff)
            additional_trash.extend(paper_detections)
            
            # === DETECT SMALL TRASH ITEMS ===
            small_detections = self.detect_small_trash(gray, hsv, depth_image, z_cutoff)
            additional_trash.extend(small_detections)
        
        except Exception as e:
            rospy.logerr(f"Error in additional trash pattern detection: {e}")
        
        return additional_trash
    
    def detect_cardboard_boxes(self, rgb_image, gray, hsv, depth_image, z_cutoff):
        """
        Detect cereal boxes, pizza boxes, cardboard packaging
        """
        box_detections = []
        
        try:
            # Method 1: Edge-based rectangle detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Check size range for boxes
                if 800 < area < 25000:  # Box-sized areas
                    # Approximate contour to see if it's rectangular
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Boxes should have 4 corners (approximately)
                    if 4 <= len(approx) <= 8:
                        x, y, w, h = cv2.boundingRect(contour)
                        center_x, center_y = x + w//2, y + h//2
                        
                        # Check aspect ratio
                        aspect_ratio = max(w, h) / min(w, h)
                        
                        # Analyze the box region
                        roi_hsv = hsv[y:y+h, x:x+w]
                        roi_rgb = rgb_image[y:y+h, x:x+w]
                        
                        if roi_hsv.size == 0:
                            continue
                        
                        # Check depth
                        depth_value = self.get_average_depth(depth_image, center_x, center_y, 5)
                        if depth_value and depth_value > 10:
                            depth_value = depth_value / 1000.0
                        
                        if not depth_value:
                            continue
                        
                        # Height check
                        if self.estimate_object_height(center_y, depth_value, 1.2, np.radians(-30)) > z_cutoff:
                            continue
                        
                        # Classify box type based on color and texture
                        box_type = self.classify_box_type(roi_rgb, roi_hsv, w, h, aspect_ratio, area)
                        
                        if box_type:
                            confidence = 0.6 + 0.2 * min(1.0, area / 5000)  # Larger boxes = higher confidence
                            box_detections.append((center_x, center_y, depth_value, f"trash_{box_type}", confidence))
                            rospy.loginfo(f"Detected {box_type} at ({center_x}, {center_y}), size: {w}x{h}")
        
        except Exception as e:
            rospy.logerr(f"Error detecting cardboard boxes: {e}")
        
        return box_detections
    
    def classify_box_type(self, roi_rgb, roi_hsv, width, height, aspect_ratio, area):
        """
        Classify what type of box this is based on visual features
        """
        try:
            # Calculate color statistics
            avg_hue = np.mean(roi_hsv[:,:,0])
            avg_saturation = np.mean(roi_hsv[:,:,1])
            avg_brightness = np.mean(roi_hsv[:,:,2])
            
            # === PIZZA BOX DETECTION ===
            # Pizza boxes: square/rectangular, white/light colored, medium-large size
            if (0.7 <= aspect_ratio <= 1.4 and  # Nearly square
                area > 2000 and  # Large enough
                avg_saturation < 60 and  # Low saturation (white/beige)
                avg_brightness > 100):  # Bright (white)
                return "pizza_box"
            
            # === CEREAL BOX DETECTION ===
            # Cereal boxes: tall rectangles, colorful graphics, medium size
            if (1.5 <= aspect_ratio <= 3.0 and  # Tall rectangle
                800 <= area <= 8000 and  # Medium size
                avg_saturation > 30):  # Some color (graphics)
                
                # Check for multiple colors (indicates printed graphics)
                unique_hues = len(np.unique(roi_hsv[:,:,0][roi_hsv[:,:,1] > 50]))
                if unique_hues > 3:  # Multiple colors = printed graphics
                    return "cereal_box"
            
            # === CARDBOARD SHIPPING BOX ===
            # Brown cardboard, rectangular, various sizes
            if (10 <= avg_hue <= 30 and  # Brown hue range
                avg_saturation > 50 and  # Saturated brown
                50 <= avg_brightness <= 150):  # Medium brightness
                return "cardboard_box"
            
            # === FOOD PACKAGING BOX ===
            # Small boxes, varied colors, compact
            if (area < 2000 and  # Small
                0.5 <= aspect_ratio <= 2.5):  # Reasonable rectangle
                return "food_packaging"
            
            # === GENERIC BOX ===
            # If rectangular but doesn't fit other categories
            if 0.3 <= aspect_ratio <= 4.0:
                return "cardboard_box"
            
            return None
            
        except Exception as e:
            rospy.logerr(f"Error classifying box type: {e}")
            return None
    
    def detect_flexible_packaging(self, gray, hsv, depth_image, z_cutoff):
        """
        Detect plastic bags, chip bags, wrappers
        """
        flex_detections = []
        
        try:
            # Look for flexible, low-contrast regions (plastic bags)
            blurred = cv2.GaussianBlur(gray, (15, 15), 0)
            low_contrast = cv2.absdiff(gray, blurred) < 15
            
            contours, _ = cv2.findContours(low_contrast.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 100 < area < 8000:  # Reasonable bag/wrapper size
                    
                    # Check if shape is irregular (bags are not rigid)
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:  # Avoid division by zero
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        
                        if circularity < 0.6:  # Irregular shape
                            x, y, w, h = cv2.boundingRect(contour)
                            center_x, center_y = x + w//2, y + h//2
                            
                            depth_value = self.get_average_depth(depth_image, center_x, center_y, 5)
                            if depth_value and depth_value > 10:
                                depth_value = depth_value / 1000.0
                            
                            if depth_value and self.estimate_object_height(center_y, depth_value, 1.2, np.radians(-30)) <= z_cutoff:
                                
                                # Check if it looks like a chip bag (metallic/reflective)
                                roi_hsv = hsv[y:y+h, x:x+w]
                                if roi_hsv.size > 0:
                                    avg_saturation = np.mean(roi_hsv[:,:,1])
                                    
                                    if avg_saturation < 50:  # Low saturation = metallic/reflective
                                        trash_type = "chip_bag"
                                    else:
                                        trash_type = "plastic_bag"
                                else:
                                    trash_type = "plastic_bag"
                                
                                flex_detections.append((center_x, center_y, depth_value, f"trash_{trash_type}", 0.5))
                                rospy.loginfo(f"Detected {trash_type} at ({center_x}, {center_y})")
        
        except Exception as e:
            rospy.logerr(f"Error detecting flexible packaging: {e}")
        
        return flex_detections
    
    def detect_metal_cans(self, gray, hsv, depth_image, z_cutoff):
        """
        Detect aluminum cans, tin cans
        """
        can_detections = []
        
        try:
            # High edge density indicates metallic surfaces
            edges = cv2.Canny(gray, 50, 150)
            
            # Dilate to connect nearby edges
            kernel = np.ones((5, 5), np.uint8)
            edge_regions = cv2.dilate(edges, kernel, iterations=1)
            
            contours, _ = cv2.findContours(edge_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 200 < area < 5000:  # Can-sized
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Check if cylindrical (reasonable aspect ratio)
                    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
                    if 1.2 < aspect_ratio < 4:  # Cylinder ratios
                        center_x, center_y = x + w//2, y + h//2
                        
                        # Check metallic appearance
                        roi_hsv = hsv[y:y+h, x:x+w]
                        if roi_hsv.size > 0:
                            avg_saturation = np.mean(roi_hsv[:,:,1])
                            
                            if avg_saturation < 80:  # Metallic = low saturation
                                depth_value = self.get_average_depth(depth_image, center_x, center_y, 5)
                                if depth_value and depth_value > 10:
                                    depth_value = depth_value / 1000.0
                                
                                if depth_value and self.estimate_object_height(center_y, depth_value, 1.2, np.radians(-30)) <= z_cutoff:
                                    
                                    # Distinguish aluminum can vs tin can by size
                                    if area > 1000 and aspect_ratio > 2:  # Tall and thin = soda can
                                        can_type = "aluminum_can"
                                    else:  # Shorter and wider = tin can
                                        can_type = "tin_can"
                                    
                                    can_detections.append((center_x, center_y, depth_value, f"trash_{can_type}", 0.7))
                                    rospy.loginfo(f"Detected {can_type} at ({center_x}, {center_y})")
        
        except Exception as e:
            rospy.logerr(f"Error detecting metal cans: {e}")
        
        return can_detections
    
    def detect_paper_trash(self, gray, hsv, depth_image, z_cutoff):
        """
        Detect newspapers, magazines, paper trash
        """
        paper_detections = []
        
        try:
            # Paper: light colored, flat, rectangular
            light_mask = cv2.inRange(hsv, np.array([0, 0, 120]), np.array([180, 80, 255]))
            
            contours, _ = cv2.findContours(light_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 500 < area < 15000:  # Paper-sized
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Paper is usually wider than tall when on ground
                    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
                    if 1.2 < aspect_ratio < 5:  # Rectangular
                        center_x, center_y = x + w//2, y + h//2
                        
                        depth_value = self.get_average_depth(depth_image, center_x, center_y, 5)
                        if depth_value and depth_value > 10:
                            depth_value = depth_value / 1000.0
                        
                        if depth_value and self.estimate_object_height(center_y, depth_value, 1.2, np.radians(-30)) <= z_cutoff:
                            
                            # Classify paper type by texture/color
                            roi_hsv = hsv[y:y+h, x:x+w]
                            if roi_hsv.size > 0:
                                avg_saturation = np.mean(roi_hsv[:,:,1])
                                
                                if avg_saturation > 40:  # Colorful = magazine
                                    paper_type = "magazine"
                                elif area > 5000:  # Large and white = newspaper
                                    paper_type = "newspaper"
                                else:  # Generic paper
                                    paper_type = "paper_trash"
                            else:
                                paper_type = "paper_trash"
                            
                            paper_detections.append((center_x, center_y, depth_value, f"trash_{paper_type}", 0.4))
                            rospy.loginfo(f"Detected {paper_type} at ({center_x}, {center_y})")
        
        except Exception as e:
            rospy.logerr(f"Error detecting paper trash: {e}")
        
        return paper_detections
    
    def detect_small_trash(self, gray, hsv, depth_image, z_cutoff):
        """
        Detect small trash: bottle caps, straws, cigarette butts
        """
        small_detections = []
        
        try:
            # Use more aggressive edge detection for small objects
            edges = cv2.Canny(gray, 30, 100)
            
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 20 < area < 400:  # Small objects
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x, center_y = x + w//2, y + h//2
                    
                    # Check if roughly circular (bottle cap)
                    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
                    
                    depth_value = self.get_average_depth(depth_image, center_x, center_y, 3)
                    if depth_value and depth_value > 10:
                        depth_value = depth_value / 1000.0
                    
                    if depth_value and self.estimate_object_height(center_y, depth_value, 1.2, np.radians(-30)) <= z_cutoff:
                        
                        if aspect_ratio < 1.5:  # Nearly circular
                            small_type = "bottle_cap"
                        elif aspect_ratio > 3:  # Very elongated
                            small_type = "straw"
                        else:  # Small and irregular
                            small_type = "small_trash"
                        
                        small_detections.append((center_x, center_y, depth_value, f"trash_{small_type}", 0.3))
                        rospy.loginfo(f"Detected {small_type} at ({center_x}, {center_y})")
        
        except Exception as e:
            rospy.logerr(f"Error detecting small trash: {e}")
        
        return small_detections
    
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
        rospy.loginfo("Using OpenCV fallback detection for trash")
        
        # Simple contour-based detection for objects on floor
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_objects = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < 300 or area > 10000:  # Reasonable trash sizes
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
                object_type = "unknown_floor_object"
                confidence = min(1.0, area / 5000.0)
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
    
    def handle_trash_detection(self, req):
        """
        Handle the service request for trash detection
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
            
            rospy.loginfo(f"Starting smart trash detection with z_cutoff: {req.z_cutoff}")
            
            # Use smart trash detection
            camera_detections = self.detect_smart_trash(
                self.latest_rgb_image, 
                self.latest_depth_image, 
                req.z_cutoff
            )
            
            # Also detect additional trash patterns (bags, cans, paper)
            additional_detections = self.detect_additional_trash_patterns(
                self.latest_rgb_image,
                self.latest_depth_image,
                req.z_cutoff
            )
            
            # Combine all detections
            all_detections = camera_detections + additional_detections
            
            if not all_detections:
                rospy.loginfo("No trash detected on floor")
                response.floor_poses = []
                return response
            
            # Transform to map frame
            map_detections = self.transform_to_map_frame(all_detections)
            
            # Fill response
            response.floor_poses = []
            for point_map, obj_type, confidence in map_detections:
                response.floor_poses.append(point_map)
            
            rospy.loginfo(f"Smart trash detection completed: {len(map_detections)} trash items found")
            rospy.loginfo(f"  - YOLO detections: {len(camera_detections)}")
            rospy.loginfo(f"  - Pattern detections: {len(additional_detections)}")
            
        except Exception as e:
            rospy.logerr(f"Error in smart trash detection service: {e}")
            response.floor_poses = []
        
        return response
    
    def run(self):
        """Keep the service running"""
        rospy.loginfo("Smart Trash Detection Service is ready")
        rospy.spin()

if __name__ == '__main__':
    try:
        service = SmartTrashDetection()
        service.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Smart Trash Detection Service interrupted")
    except Exception as e:
        rospy.logerr(f"Error starting service: {e}")