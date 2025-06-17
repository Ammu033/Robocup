#!/usr/bin/env python3

import rospy
from std_srvs.srv import Trigger
import sys

class CoverageController:
    def __init__(self):
        rospy.init_node('coverage_controller', anonymous=True)
        
        # Wait for services
        rospy.loginfo("Waiting for coverage services...")
        rospy.wait_for_service('/coverage_planner/generate_path')
        rospy.wait_for_service('/coverage_planner/start_coverage')
        rospy.wait_for_service('/coverage_planner/stop_coverage')
        rospy.wait_for_service('/coverage_planner/get_progress')
        
        # Create service proxies - renamed to avoid conflicts
        self.generate_path_srv = rospy.ServiceProxy('/coverage_planner/generate_path', Trigger)
        self.start_coverage_srv = rospy.ServiceProxy('/coverage_planner/start_coverage', Trigger)
        self.stop_coverage_srv = rospy.ServiceProxy('/coverage_planner/stop_coverage', Trigger)
        self.get_progress_srv = rospy.ServiceProxy('/coverage_planner/get_progress', Trigger)
        
        rospy.loginfo("Coverage controller ready!")

    def generate_path(self):
        """Generate coverage path"""
        try:
            response = self.generate_path_srv()  # Fixed: call the service proxy
            if response.success:
                rospy.loginfo(f"✅ {response.message}")
            else:
                rospy.logerr(f"❌ {response.message}")
            return response.success
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return False

    def start_coverage(self):
        """Start coverage execution"""
        try:
            response = self.start_coverage_srv()  # Fixed: call the service proxy
            if response.success:
                rospy.loginfo(f"✅ {response.message}")
            else:
                rospy.logerr(f"❌ {response.message}")
            return response.success
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return False

    def stop_coverage(self):
        """Stop coverage execution"""
        try:
            response = self.stop_coverage_srv()  # Fixed: call the service proxy
            if response.success:
                rospy.loginfo(f"✅ {response.message}")
            else:
                rospy.logerr(f"❌ {response.message}")
            return response.success
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return False

    def get_status(self):
        """Get current status"""
        try:
            # Get path progress
            progress_response = self.get_progress_srv()  # Fixed: call the service proxy
            if progress_response.success:
                rospy.loginfo(f"📊 Progress: {progress_response.message}")
            else:
                rospy.logwarn(f"⚠️ Could not get progress: {progress_response.message}")
            
            return progress_response.success
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return False

    def run_complete_coverage(self):
        """Run complete coverage sequence"""
        rospy.loginfo("🚀 Starting complete coverage sequence...")
        
        # Step 1: Generate path
        rospy.loginfo("Step 1: Generating coverage path...")
        if not self.generate_path():
            rospy.logerr("Failed to generate path. Aborting.")
            return False
        
        rospy.sleep(1.0)  # Brief pause
        
        # Step 2: Start coverage
        rospy.loginfo("Step 2: Starting coverage execution...")
        if not self.start_coverage():
            rospy.logerr("Failed to start coverage. Aborting.")
            return False
        
        # Step 3: Monitor progress
        rospy.loginfo("Step 3: Monitoring progress (Ctrl+C to stop)...")
        rate = rospy.Rate(0.2)  # Every 5 seconds
        
        try:
            while not rospy.is_shutdown():
                self.get_status()
                rate.sleep()
        except KeyboardInterrupt:
            rospy.loginfo("🛑 Stopping coverage...")
            self.stop_coverage()
        
        return True

def print_usage():
    print("""
Usage: python3 coverage_controller.py [command]

Commands:
  generate  - Generate coverage path
  start     - Start coverage execution  
  stop      - Stop coverage execution
  status    - Get current status
  auto      - Run complete automated coverage
  
Examples:
  python3 coverage_controller.py generate
  python3 coverage_controller.py start
  python3 coverage_controller.py auto
""")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    try:
        controller = CoverageController()
        
        if command == 'generate':
            controller.generate_path()
        elif command == 'start':
            controller.start_coverage()
        elif command == 'stop':
            controller.stop_coverage()
        elif command == 'status':
            controller.get_status()
        elif command == 'auto':
            controller.run_complete_coverage()
        else:
            print(f"Unknown command: {command}")
            print_usage()
            sys.exit(1)
            
    except rospy.ROSInterruptException:
        rospy.loginfo("Coverage controller interrupted")
    except Exception as e:
        rospy.logerr(f"Error: {e}")
        sys.exit(1)