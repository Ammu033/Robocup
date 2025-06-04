#!/usr/bin/env python
# Nav.py - The Brain of the Robot Navigation System

import rospy
import json
import subprocess
import threading
from std_msgs.msg import String
from transformers import pipeline
import re
import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class RobotBrain:
    def __init__(self):
        # Initialize OpenAI client
        self.classifier = pipeline("zero-shot-classification", model="typeform/mobilebert-uncased-mnli", max_length=512, truncation=True, device=0 )

        
        
        
        # Initialize ROS
        rospy.init_node('robot_brain', anonymous=True)
        
        # ROS Publishers and Subscribers
        self.status_publisher = rospy.Publisher('/robot_status', String, queue_size=10)
        
        # Store last position for go_last_position functionality
        self.last_position = None
        
        print(" Robot Brain (Nav.py) Started!")
        print(" Ready for direct text input!")
        print(" Publishing status on /robot_status topic")
        
    def publish_status(self, message):
        """Publish status message"""
        status_msg = String()
        status_msg.data = message
        self.status_publisher.publish(status_msg)
        print(f"📢 {message}")
        
    def execute_rostopic_command(self, command_text):
        """Execute rostopic pub command internally"""
        try:
            self.publish_status(f"🚀 Executing internal command: {command_text}")
            
            # Execute rostopic pub command
            result = subprocess.run([
                'rostopic', 'pub', '/robot_command', 'std_msgs/String', 
                f'"{command_text}"', '--once'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                self.publish_status(f"✅ ROS command executed successfully")
                return True
            else:
                self.publish_status(f"❌ ROS command failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.publish_status(f"❌ Error executing ROS command: {e}")
            return False
        
    def command_callback(self, msg):
        """Handle incoming commands from rostopic"""
        user_input = msg.data.strip()
        if user_input:
            print(f"\n📡 Brain received via ROS: '{user_input}'")
            # Process command in separate thread
            threading.Thread(target=self.process_command, args=(user_input,)).start()
    
    def classify_intent(self, user_input):
        """Use AI to classify user intent"""
        prompt = f"""
        You are a robot navigation assistant. Analyze this command and respond with ONLY a JSON object:

        Command: "{user_input}"

        Classify the intent as one of:
        1. "navigate" - user wants robot to go to a specific location
        2. "save_pose" - user wants to save current location
        3. "go_last" - user wants to go to the last/previous position

        Extract the location/room name if mentioned.

        Response format (JSON only, no other text):
        {{
            "intent": "navigate" or "save_pose" or "go_last",
            "location": "extracted_room_name" or null,
            "confidence": 0.0-1.0
        }}
        """

        try:
            labels = ["navigate", "save_pose", "go_last"]
            result = self.classifier(user_input, labels)
            intent = result["labels"][0]
            confidence = float(result["scores"][0])

            # Try to extract location (basic example)
            match = re.search(r"\b(room_\d+|kitchen|hall|bedroom|office|living room)\b", user_input, re.IGNORECASE)
            location = match.group(0) if match else None

            output = {
                "intent": intent,
                "location": location if intent != "go_last" else None,
                "confidence": round(confidence, 2)
            }
            return output

        except Exception as e:
            print(f"[ERROR] AI classification failed: {e}")
            return self.fallback_classification(user_input)
    
    def fallback_classification(self, user_input):
        """Fallback rule-based classification"""
        user_input = user_input.lower().strip()
        
        # Keywords for different intents
        nav_keywords = ['go to', 'move to', 'navigate to', 'travel to', 'head to']
        save_keywords = ['save', 'remember', 'store', 'record', 'mark']
        last_keywords = ['go back', 'return', 'previous', 'last position', 'go last']
        
        # Check for go last intent
        if any(keyword in user_input for keyword in last_keywords):
            return {"intent": "go_last", "location": None, "confidence": 0.8}
        
        # Check for navigation intent - more flexible parsing
        elif any(keyword in user_input for keyword in nav_keywords) or 'go to' in user_input:
            location = None
            # Try different patterns to extract location
            if 'go to' in user_input:
                location = user_input.split('go to')[-1].strip()
            elif 'move to' in user_input:
                location = user_input.split('move to')[-1].strip()
            elif 'navigate to' in user_input:
                location = user_input.split('navigate to')[-1].strip()
            # Handle "hey robot go to room_1" pattern
            elif 'robot' in user_input and ('go' in user_input or 'move' in user_input):
                words = user_input.split()
                if 'go' in words:
                    go_index = words.index('go')
                    if go_index + 1 < len(words):
                        location = ' '.join(words[go_index + 1:])
            
            return {"intent": "navigate", "location": location, "confidence": 0.7}
        
        # Check for save intent
        elif any(keyword in user_input for keyword in save_keywords):
            location = None
            if 'as ' in user_input:
                location = user_input.split('as ')[-1].strip()
            
            return {"intent": "save_pose", "location": location, "confidence": 0.7}
        
        return {"intent": "unknown", "location": None, "confidence": 0.0}
    
    def execute_save_pose(self, room_name):
        """Execute save_pose.py action"""
        try:
            self.publish_status(f"💾 Executing save_pose.py for '{room_name}'")
            
            # Run save_pose.py as subprocess
            result = subprocess.run([
                'rosrun', 
                'tiago_auto', 'save_pose.py'
            ], input=room_name, text=True, capture_output=True, timeout=30)
            
            if result.returncode == 0:
                self.publish_status(f"✅ save_pose.py completed successfully")
                return True
            else:
                self.publish_status(f"❌ save_pose.py failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.publish_status(f"❌ Error executing save_pose.py: {e}")
            return False
    
    def execute_send_goal(self, location):
        """Execute sd_goal.py action"""
        try:
            self.publish_status(f"🎯 Executing sd_goal.py for '{location}'")
            
            # Store current target as last position for future reference
            self.last_position = location
            
            # Run sd_goal.py as subprocess
            result = subprocess.run([
                'rosrun', 
                'tiago_auto', 'by_command.py'
            ], input=location, text=True, capture_output=True, timeout=120)
            
            if result.returncode == 0:
                self.publish_status(f"✅ sd_goal.py completed successfully")
                return True
            else:
                self.publish_status(f"❌ sd_goal.py failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.publish_status(f"❌ Error executing sd_goal.py: {e}")
            return False
    
    def execute_go_last_position(self):
        """Execute go_last_position.py action"""
        try:
            if not self.last_position:
                self.publish_status("❌ No previous position recorded")
                return False
                
            self.publish_status(f"🔄 Executing go_last_position.py to '{self.last_position}'")
            
            # Run go_last_position.py as subprocess
            result = subprocess.run([
                'rosrun', 
                'tiago_auto',
                'go_last_position.py'
            ], input=self.last_position, text=True, capture_output=True, timeout=120)
            
            if result.returncode == 0:
                self.publish_status(f"✅ go_last_position.py completed successfully")
                return True
            else:
                self.publish_status(f"❌ go_last_position.py failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.publish_status(f"❌ Error executing go_last_position.py: {e}")
            return False

    def process_command(self, user_input):
        """Main brain processing function"""
        self.publish_status(f"🧠 Brain processing: '{user_input}'")
        
        # Get AI classification
        result = self.classify_intent(user_input)
        
        self.publish_status(f"📊 Intent: {result['intent']}, Location: {result['location']}, Confidence: {result['confidence']:.2f}")
        
        if result['confidence'] < 0.5:
            self.publish_status("⚠️ Low confidence. Please rephrase command.")
            return False
        
        # Route to appropriate action based on intent
        if result['intent'] == 'navigate':
            if not result['location']:
                self.publish_status("❌ No destination specified")
                return False
            return self.execute_send_goal(result['location'])
            
        elif result['intent'] == 'save_pose':
            if not result['location']:
                self.publish_status("❌ No name specified for saving pose")
                return False
            return self.execute_save_pose(result['location'])
            
        elif result['intent'] == 'go_last':
            return self.execute_go_last_position()
            
        else:
            self.publish_status("❌ Could not understand command")
            return False

    def run(self):
        """Keep the brain running with interactive input"""
        self.publish_status("🧠 Robot Brain is online and ready!")
        self.publish_status("💬 Type commands directly or use ROS topics")
        
        # Add ROS subscriber for external commands
        self.command_subscriber = rospy.Subscriber('/robot_command', String, self.command_callback)
        
        print("\n" + "="*50)
        print("🤖 ROBOT BRAIN - INTERACTIVE MODE")
        print("="*50)
        print("Examples:")
        print("  'hey robot go to room_1'")
        print("  'robot save this as kitchen'") 
        print("  'go back to previous position'")
        print("  'quit' to exit")
        print("="*50 + "\n")
        
        # Interactive input loop
        try:
            while not rospy.is_shutdown():
                try:
                    user_input = input("🎤 You: ").strip()
                    
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        print("👋 Robot Brain shutting down...")
                        break
                        
                    if user_input:
                        self.process_direct_command(user_input)
                    else:
                        print("Please enter a command.")
                        
                except KeyboardInterrupt:
                    print("\n👋 Exiting Robot Brain...")
                    break
                    
        except Exception as e:
            print(f"❌ Error in interactive mode: {e}")
    
    def process_direct_command(self, user_input):
        """Process direct text input from user"""
        print(f"\n🧠 Brain processing: '{user_input}'")
        
        # Get AI classification
        result = self.classify_intent(user_input)
        
        print(f"📊 Intent: {result['intent']}, Location: {result['location']}, Confidence: {result['confidence']:.2f}")
        
        if result['confidence'] < 0.5:
            print("⚠️ Low confidence. Please rephrase command.")
            return False
        
        # Route to appropriate action based on intent
        if result['intent'] == 'navigate':
            if not result['location']:
                print("❌ No destination specified")
                return False
            return self.execute_send_goal(result['location'])
            
        elif result['intent'] == 'save_pose':
            if not result['location']:
                print("❌ No name specified for saving pose")
                return False
            return self.execute_save_pose(result['location'])
            
        elif result['intent'] == 'go_last':
            return self.execute_go_last_position()
            
        else:
            print("❌ Could not understand command")
            return False

def main():
    """Main function"""
    try:
        brain = RobotBrain()
        brain.run()
        
    except rospy.ROSInterruptException:
        print("🛑 Robot Brain shutting down...")
    except KeyboardInterrupt:
        print("🛑 Robot Brain interrupted by user")

if __name__ == '__main__':
    main()


# =============================================
# COMPLEMENTARY FILES NEEDED:
# =============================================

"""
You'll need to create these additional files:

1. sd_goal.py - Modified version of your by_command.py
2. go_last_position.py - Similar to sd_goal.py but for last position
3. save_pose.py - You already have this one

Here's the structure:

Brain (Nav.py) 
├── Receives command via /robot_command topic
├── AI classifies intent  
├── Routes to appropriate action:
    ├── save_pose.py (for saving poses)
    ├── sd_goal.py (for navigation to specific location)  
    └── go_last_position.py (for returning to previous location)

Each action runs independently and reports back results.
"""