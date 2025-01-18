#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import ColorRGBA, Float32
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from std_msgs.msg import UInt8

class HandTracker:
    def __init__(self):
        # MediaPipe setup
        self.bridge = CvBridge()
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # ROS setup
        rospy.init_node('hand_tracking_node', anonymous=True)
        self.color_sub = rospy.Subscriber("/camera/color/image_raw", 
            Image, self.color_callback)
        self.depth_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", 
            Image, self.depth_callback)
        self.result_pub = rospy.Publisher("/hand_tracking/result", 
            Image, queue_size=1)
        self.marker_pub = rospy.Publisher("/hand_tracking/trajectory", 
            MarkerArray, queue_size=1)
        self.depth_pub = rospy.Publisher("/hand_tracking/palm_depth",
            Float32, queue_size=1)
        self.position_pub = rospy.Publisher('/hand_tracking/position', Vector3, queue_size=10)
        self.velocity_pub = rospy.Publisher('/hand_tracking/velocity', Vector3, queue_size=10)
        self.gesture_pub = rospy.Publisher('/hand_tracking/gesture', UInt8, queue_size=1)
        

        # State variables
        self.color_frame = None
        self.depth_frame = None
        self.trajectories = {}
        self.distances = {}
        self.prev_points = [None, None]  # For both hands
        self.last_time = [rospy.Time.now(), rospy.Time.now()]
    

    def color_callback(self, msg):
        try:
            self.color_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.process_frames()
        except Exception as e:
            rospy.logerr(f"Color conversion error: {str(e)}")

    def depth_callback(self, msg):
        try:
            self.depth_frame = self.bridge.imgmsg_to_cv2(msg)
        except Exception as e:
            rospy.logerr(f"Depth conversion error: {str(e)}")

    def calculate_distance(self, p1, p2):
        """Calculate 2D distance between two landmarks"""
        return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5

    def detect_hand_gesture(self, hand_landmarks):
        # Get wrist and thumb points
        wrist = hand_landmarks.landmark[1]
        thumb = hand_landmarks.landmark[3]
        
        # Calculate thumb distance as reference
        thumb_dist = self.calculate_distance(wrist, thumb)
        self.distances = {
            "THUMB": thumb_dist,
            "INDEX": self.calculate_distance(wrist, hand_landmarks.landmark[8]),
            "MIDDLE": self.calculate_distance(wrist, hand_landmarks.landmark[12]),
            "RING": self.calculate_distance(wrist, hand_landmarks.landmark[16]),
            "PINKY": self.calculate_distance(wrist, hand_landmarks.landmark[20])
        }

        # Detect gestures and publish byte value
        gesture_msg = UInt8()
        
        if (self.distances["INDEX"] > thumb_dist and 
            self.distances["MIDDLE"] > thumb_dist and 
            self.distances["RING"] < thumb_dist and 
            self.distances["PINKY"] < thumb_dist):
            gesture_msg.data = 200  # PEACE
            self.gesture_pub.publish(gesture_msg)
            return "PEACE"
        elif all(self.distances[finger] > thumb_dist for finger in ["INDEX", "MIDDLE", "RING", "PINKY"]):
            gesture_msg.data = 0    # OPEN
            self.gesture_pub.publish(gesture_msg)
            return "OPEN"
        elif all(self.distances[finger] < thumb_dist for finger in ["INDEX", "MIDDLE", "RING", "PINKY"]):
            gesture_msg.data = 100  # CLOSED
            self.gesture_pub.publish(gesture_msg)
            return "CLOSED"
        
        gesture_msg.data = 255  # UNKNOWN
        self.gesture_pub.publish(gesture_msg)
        return "UNKNOWN"

    def create_hand_marker(self, trajectory, hand_id, hand_state):
        marker = Marker()
        marker.header.frame_id = "camera_link"
        marker.header.stamp = rospy.Time.now()
        marker.ns = f"hand_trajectory_{hand_id}"
        marker.id = hand_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale = Vector3(0.005, 0.005, 0.005)

        colors = {
            "CLOSED": ColorRGBA(1.0, 0.0, 0.0, 1.0),  # Red for グー
            "PEACE": ColorRGBA(0.0, 0.0, 1.0, 1.0),   # Blue for チョキ
            "OPEN": ColorRGBA(0.0, 1.0, 0.0, 1.0),    # Green for パー
            "UNKNOWN": ColorRGBA(0.5, 0.5, 0.5, 1.0)   # Gray
        }
        marker.color = colors.get(hand_state, colors["UNKNOWN"])
        marker.points = list(trajectory)
        return marker

    def update_trajectory(self, point, hand_id):
        if hand_id not in self.trajectories:
            self.trajectories[hand_id] = deque(maxlen=50)
        
        trajectory = self.trajectories[hand_id]
        if len(trajectory) == 0 or \
           np.linalg.norm([point.x - trajectory[-1].x,
                          point.y - trajectory[-1].y,
                          point.z - trajectory[-1].z]) > 0.01:
            trajectory.append(point)

    def calculate_velocity(self, current_point, prev_point, dt):
        if prev_point is None or dt == 0:
            return Vector3(0, 0, 0)
        
        velocity = Vector3()
        velocity.x = (current_point.x - prev_point.x) / dt
        velocity.y = (current_point.y - prev_point.y) / dt
        velocity.z = (current_point.z - prev_point.z) / dt
        return velocity

    def process_frames(self):
        if self.color_frame is None or self.depth_frame is None:
            return

        rgb_frame = cv2.cvtColor(self.color_frame, cv2.COLOR_BGR2RGB)
        result = self.color_frame.copy()
        h, w = self.depth_frame.shape
        marker_array = MarkerArray()

        results = self.hands.process(rgb_frame)

        # Initialize gesture message with UNKNOWN
        gesture_msg = UInt8()
        gesture_msg.data = 255  # UNKNOWN

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                self.mp_draw.draw_landmarks(
                    result,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

                hand_state = self.detect_hand_gesture(hand_landmarks)

                # Calculate palm center
                palm_center_x = int(np.mean([hand_landmarks.landmark[i].x * w for i in [0,1,5,9,13,17]]))
                palm_center_y = int(np.mean([hand_landmarks.landmark[i].y * h for i in [0,1,5,9,13,17]]))

                # Define depth region
                region_size = 10
                x_min = max(0, palm_center_x - region_size)
                x_max = min(w, palm_center_x + region_size)
                y_min = max(0, palm_center_y - region_size)
                y_max = min(h, palm_center_y + region_size)

                depth_region = self.depth_frame[y_min:y_max, x_min:x_max]
                depth_values = depth_region[depth_region != 0]
                
                if len(depth_values) > 0:
                    current_palm_depth = np.mean(depth_values) / 1000.0
                    
                    # Initialize previous values if not existing
                    if not hasattr(self, 'previous_palm_depth'):
                        self.previous_palm_depth = current_palm_depth
                        self.last_valid_palm_depth = current_palm_depth
                        self.previous_time = rospy.Time.now()
                    
                    # Calculate difference (m)
                    depth_difference = abs(current_palm_depth - self.previous_palm_depth)
                    
                    # If difference is below threshold AND depth is less than 2m, update valid palm depth
                    if depth_difference < 0.04 and current_palm_depth < 2.0:
                        palm_depth = current_palm_depth
                        self.last_valid_palm_depth = palm_depth
                    else:
                        palm_depth = self.last_valid_palm_depth
                    
                    # Update previous value
                    self.previous_palm_depth = current_palm_depth
                    
                    # Publish palm depth
                    depth_msg = Float32()
                    depth_msg.data = palm_depth
                    self.depth_pub.publish(depth_msg)
                else:
                    palm_depth = self.last_valid_palm_depth if hasattr(self, 'last_valid_palm_depth') else 0.0

                # Create current position
                point = Point()
                point.z = palm_depth
                point.x = (palm_center_x - w/2) * palm_depth / 500.0
                point.y = -(palm_center_y - h/2) * palm_depth / 500.0                
                # Convert Point to Vector3 and publish position
                position = Vector3()
                position.x = point.x * 10.0
                position.y = point.y * 10.0
                position.z = point.z * 10.0
                self.position_pub.publish(position)

                # Calculate and publish velocity
                current_time = rospy.Time.now()
                dt = (current_time - self.last_time[idx]).to_sec()
                velocity = self.calculate_velocity(point, self.prev_points[idx], dt)
                self.velocity_pub.publish(velocity)

                # Update previous values
                self.prev_points[idx] = point
                self.last_time[idx] = current_time

                self.update_trajectory(point, idx)
                marker = self.create_hand_marker(self.trajectories[idx], idx, hand_state)
                marker_array.markers.append(marker)

                # Display info
                handedness = results.multi_handedness[idx].classification[0].label
                cv2.circle(result, (palm_center_x, palm_center_y), 5, (255, 0, 0), -1)
                cv2.putText(
                    result,
                    f"{handedness} Hand: {hand_state} (D:{palm_depth:.2f}m)",
                    (10, 30 + idx * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )  
        else:
            # No hands detected, publish UNKNOWN
            self.gesture_pub.publish(gesture_msg)

        self.marker_pub.publish(marker_array)
        cv2.imshow("Hand Tracking", result)
        cv2.waitKey(1)

        try:
            result_msg = self.bridge.cv2_to_imgmsg(result, "bgr8")
            self.result_pub.publish(result_msg)
        except Exception as e:
            rospy.logerr(f"Result conversion error: {str(e)}")

    def cleanup(self):
        cv2.destroyAllWindows()

def main():
    tracker = HandTracker()
    rospy.on_shutdown(tracker.cleanup)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()