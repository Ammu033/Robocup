#!/usr/bin/env python

import rospy
from people_msgs.msg import People
from std_msgs.msg import String
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose2D, PoseStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import math
import tf
import numpy as np
import actionlib



NODE_NAME = 'lcastor_person_following'
NODE_RATE = 100 # [Hz]
DES_DIST = float(rospy.get_param("/lcastor_person_following/des_dist"))
CANCEL_DISTANCE = float(rospy.get_param("/lcastor_person_following/cancel_dist"))


class PersonFollowing():
    """
    Person Following class
    """
    
    def __init__(self):
        """
        Class constructor. Init publishers and subscribers
        """
        
        self.personID_to_follow = None
        self.people = None
        self.robot_pose = Pose2D()
        self.goal = None
        self.send_goal_pos = None
        self.client = None
        
        # Robot pose subscriber
        self.sub_robot_pos = rospy.Subscriber('/robot_pose', PoseWithCovarianceStamped, callback = self.cb_robot_pose)
        
        # Person to follow subscriber
        self.sub_person_to_follow = rospy.Subscriber('/person_to_follow', String, callback = self.cb_person_to_follow)
        
        # People subscriber
        self.sub_people = rospy.Subscriber('/people_tracker/people', People, callback = self.cb_people)
        
        
    def cb_robot_pose(self, p: PoseWithCovarianceStamped):
        """
        from 3D to 2D robot pose

        Args:
            p (PoseWithCovarianceStamped): 3D robot pose
        """
        
        q = (
            p.pose.pose.orientation.x,
            p.pose.pose.orientation.y,
            p.pose.pose.orientation.z,
            p.pose.pose.orientation.w
        )
        
        m = tf.transformations.quaternion_matrix(q)
        
        self.robot_pose.x = p.pose.pose.position.x
        self.robot_pose.y = p.pose.pose.position.y
        self.robot_pose.theta = tf.transformations.euler_from_matrix(m)[2]

        if self.send_goal_pos is not None and self.client is not None and self.client.get_state() == actionlib.GoalStatus.ACTIVE:
            distance_travelled = math.dist(self.send_goal_pos, [self.robot_pose.x, self.robot_pose.y])
            if distance_travelled > CANCEL_DISTANCE: self.client.cancel_goal()
        
        
    def cb_person_to_follow(self, id: String):
        """
        Stores the person id to follow

        Args:
            id (String): Person ID detected
        """
        # check if old person ID still exist in the people list
        if self.personID_to_follow is None or not any(person.name == self.personID_to_follow for person in self.people):
            # If the old person ID is still detected, self.personID_to_follow not updated -> the robot will continue to follow the same person
            # If the old person ID is not detected anymore, self.personID_to_follow is updated -> the robot will follow the new person
            self.personID_to_follow = id.data

        # rospy.logdebug("Person to follow ID" + self.personID_to_follow)

        
    def cb_people(self, data: People):
        """
        Stores people

        Args:
            data (People): people topic from people tracker
        """
        self.people = data.people
        
        
    def calculate_goal(self, person_pos):
        """
        Calculates goal pos and orientation

        Args:
            person_pos (list): [x, y]

        Returns:
            array, float: goal pos and orientation
        """
        person_pos = np.array(person_pos)
        robot_pos = np.array([self.robot_pose.x, self.robot_pose.y])
        
        # Calculate the vector from the robot to the person
        vector_to_person = person_pos - robot_pos

        # Calculate the distance from the robot to the person
        distance_to_person = math.sqrt(vector_to_person[0]**2 + vector_to_person[1]**2)

        # Normalize the vector to the desired distance
        normalized_vector = vector_to_person / distance_to_person

        # Calculate the orientation needed to reach the person
        goal_orientation = math.atan2(normalized_vector[1], normalized_vector[0])

        # Calculate the goal position based on the desired distance
        goal_position = person_pos - DES_DIST * normalized_vector

        return goal_position, goal_orientation    
    
        
    def send_goal(self, goal_position, goal_orientation):
        """
        Creates goal msg

        Args:
            goal_position (array): x, y
            goal_orientation (float): theta
        """
        
        # Publish the goal position and orientation to the navigation system
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.client.wait_for_server()
        
        goal_msg = MoveBaseGoal()
        goal_msg.target_pose.header.frame_id = "map"
        goal_msg.target_pose.header.stamp = rospy.Time.now()
        goal_msg.target_pose.pose.position.x = goal_position[0]
        goal_msg.target_pose.pose.position.y = goal_position[1]
        goal_msg.target_pose.pose.orientation.z = math.sin(goal_orientation / 2)
        goal_msg.target_pose.pose.orientation.w = math.cos(goal_orientation / 2)

        self.client.send_goal(goal_msg)
        self.client.wait_for_result()
        
        
    def get_person_pos(self):
        """
        Get position of the person to follow if the latter exists, otherwise returns None

        Returns:
            list: [x, y]
        """
        for person in self.people:
            if person.name == self.personID_to_follow:
                return [person.position.x, person.position.y]
        return None
    
    
    def follow_person(self):
        """
        Calculates and publishes the goal position and orientation when personID_to_follow is in people list
        """
        
        # If not people tracked then just return
        if self.people is None: return

        person_position = self.get_person_pos()
        if person_position is not None:
            # Calculate the goal position and orientation based on the desired distance
            goal_position, goal_orientation = self.calculate_goal(person_position)

            # Send the goal position and orientation to the navigation system
            self.send_goal_pos = [self.robot_pose.x, self.robot_pose.y]
            self.send_goal(goal_position, goal_orientation)
            
        
if __name__ == '__main__':    
        
    # Init node
    rospy.init_node(NODE_NAME)
    
    # Set node rate
    rate = rospy.Rate(NODE_RATE)
    
    person_detector = PersonFollowing()
    
    while not rospy.is_shutdown():
        person_detector.follow_person()
        rate.sleep()