#!/usr/bin/env python3
"""
Real-time emoji display based on camera pose, facial expression, and hand gesture detection.
"""

import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Solutions
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands # เพิ่ม: สำหรับตรวจจับมือ
mp_drawing = mp.solutions.drawing_utils

# Configuration
SMILE_THRESHOLD = 0.35
SAD_THRESHOLD = 0.53  
ANGRY_THRESHOLD = 0.05 
WINDOW_WIDTH = 720
WINDOW_HEIGHT = 450
EMOJI_WINDOW_SIZE = (WINDOW_WIDTH, WINDOW_HEIGHT)
HAND_RAISE_THRESHOLD = 0.05 # ระยะห่าง y-axis ระหว่างข้อมือกับไหล่

# --- ฟังก์ชันช่วยเหลือสำหรับตรวจจับท่าทางมือ ---
def is_thumbs_up(landmarks):
    # นิ้วหัวแม่มือ (Thumb Tip - 4) ต้องอยู่เหนือข้อต่อ (Thumb IP - 3)
    # นิ้วอื่นๆ (8, 12, 16, 20) ต้องหุบ (ปลายอยู่ใกล้กับข้อต่อ)
    thumb_up = landmarks[4].y < landmarks[3].y
    index_down = landmarks[8].y > landmarks[7].y
    middle_down = landmarks[12].y > landmarks[11].y
    ring_down = landmarks[16].y > landmarks[15].y
    pinky_down = landmarks[20].y > landmarks[19].y
    return thumb_up and index_down and middle_down and ring_down and pinky_down

def is_love_sign(landmarks):
    # นิ้วก้อย (Pinky Tip - 20), นิ้วชี้ (Index Tip - 8) และนิ้วหัวแม่มือ (Thumb Tip - 4) กางออก
    # นิ้วกลาง (Middle - 12) และ นิ้วนาง (Ring - 16) งอเข้า
    pinky_up = landmarks[20].y < landmarks[18].y
    index_up = landmarks[8].y < landmarks[6].y
    thumb_up = landmarks[4].x < landmarks[3].x # กางหัวแม่มือ
    middle_down = landmarks[12].y > landmarks[11].y
    ring_down = landmarks[16].y > landmarks[15].y
    return pinky_up and index_up and thumb_up and middle_down and ring_down

def is_middle_finger(landmarks):
    # นิ้วกลาง (Middle Tip - 12) ชี้ขึ้น
    # นิ้วอื่นๆ (8, 16, 20) ต้องหุบ/งอเข้า
    middle_up = landmarks[12].y < landmarks[11].y and landmarks[12].y < landmarks[10].y
    index_down = landmarks[8].y > landmarks[7].y
    ring_down = landmarks[16].y > landmarks[15].y
    pinky_down = landmarks[20].y > landmarks[19].y
    return middle_up and index_down and ring_down and pinky_down

def is_rock_on(landmarks):
    # นิ้วชี้ (Index Tip - 8) และ นิ้วก้อย (Pinky Tip - 20) ชี้ขึ้น
    # นิ้วกลาง (Middle - 12) และ นิ้วนาง (Ring - 16) งอเข้า
    index_up = landmarks[8].y < landmarks[7].y
    pinky_up = landmarks[20].y < landmarks[19].y
    middle_down = landmarks[12].y > landmarks[11].y
    ring_down = landmarks[16].y > landmarks[15].y
    return index_up and pinky_up and middle_down and ring_down

# Load emoji images
try:
    smiling_emoji = cv2.imread("smile.png")
    straight_face_emoji = cv2.imread("plain.png")
    # Pose
    hands_up_emoji = cv2.imread("air.png")
    # Face
    angry_emoji = cv2.imread("angry.png")
    sad_emoji = cv2.imread("sad.png")
    # Hands
    thumbs_up_emoji = cv2.imread("thumbs_up.png")
    love_sign_emoji = cv2.imread("love_sign.png")
    middle_finger_emoji = cv2.imread("middle_finger.png")
    rock_on_emoji = cv2.imread("rock_on.png")


    if any(e is None for e in [smiling_emoji, straight_face_emoji, hands_up_emoji, angry_emoji, sad_emoji, thumbs_up_emoji, love_sign_emoji, middle_finger_emoji, rock_on_emoji]):
        raise FileNotFoundError("One or more emoji files not found.")

    # Resize emojis
    emojis = {
        'SMILING': cv2.resize(smiling_emoji, EMOJI_WINDOW_SIZE),
        'STRAIGHT_FACE': cv2.resize(straight_face_emoji, EMOJI_WINDOW_SIZE),
        'HANDS_UP': cv2.resize(hands_up_emoji, EMOJI_WINDOW_SIZE),
        'ANGRY': cv2.resize(angry_emoji, EMOJI_WINDOW_SIZE),
        'SAD': cv2.resize(sad_emoji, EMOJI_WINDOW_SIZE),
        'THUMBS_UP': cv2.resize(thumbs_up_emoji, EMOJI_WINDOW_SIZE),
        'LOVE_SIGN': cv2.resize(love_sign_emoji, EMOJI_WINDOW_SIZE),
        'MIDDLE_FINGER': cv2.resize(middle_finger_emoji, EMOJI_WINDOW_SIZE),
        'ROCK_ON': cv2.resize(rock_on_emoji, EMOJI_WINDOW_SIZE),
    }

except Exception as e:
    print("Error loading emoji images!")
    print(f"Details: {e}")
    print("\nExpected files:")
    print("- smile.jpg, plain.png, air.jpg")
    print("- angry.png, sad.png")
    print("- thumbs_up.png, love_sign.png, middle_finger.png, rock_on.png")
    exit()

blank_emoji = np.zeros((EMOJI_WINDOW_SIZE[0], EMOJI_WINDOW_SIZE[1], 3), dtype=np.uint8)

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cv2.namedWindow('Emoji Output', cv2.WINDOW_NORMAL)
cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camera Feed', WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.resizeWindow('Emoji Output', WINDOW_WIDTH, WINDOW_HEIGHT)
cv2.moveWindow('Camera Feed', 100, 100)
cv2.moveWindow('Emoji Output', WINDOW_WIDTH + 150, 100)

print("Controls:")
print("  Press 'q' to quit")
print("  Raise hands above shoulders for HANDS UP")
print("  Make gestures like THUMBS UP, LOVE, MIDDLE FINGER, ROCK ON")
print("  Change facial expressions for SMILING, ANGRY, SAD")

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5) as face_mesh, \
     mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands: # เพิ่ม: Hands Detector

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        current_state = "STRAIGHT_FACE"
        
        # 1. ตรวจจับท่าทางมือ (Hand Gestures)
        results_hands = hands.process(image_rgb)
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                landmarks = hand_landmarks.landmark
                
                if is_middle_finger(landmarks):
                    current_state = "MIDDLE_FINGER"
                    break
                elif is_love_sign(landmarks):
                    current_state = "LOVE_SIGN"
                    break
                elif is_rock_on(landmarks):
                    current_state = "ROCK_ON"
                    break
                elif is_thumbs_up(landmarks):
                    current_state = "THUMBS_UP"
                    break
        
        # 2. ตรวจจับท่าทางร่างกาย (Pose) - เช็ค Hands Up (ยกมือ)
        if current_state == "STRAIGHT_FACE": # ถ้ามือไม่ทำท่าทางเฉพาะ ให้เช็คท่าทางร่างกาย
            results_pose = pose.process(image_rgb)
            if results_pose.pose_landmarks:
                landmarks = results_pose.pose_landmarks.landmark
                
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

                # ยกมือสูงกว่าไหล่ (ในแกน Y)
                if (left_wrist.y < left_shoulder.y - HAND_RAISE_THRESHOLD) or \
                   (right_wrist.y < right_shoulder.y - HAND_RAISE_THRESHOLD):
                    current_state = "HANDS_UP"
        
        # 3. ตรวจจับสีหน้า (Facial Expression)
        if current_state == "STRAIGHT_FACE": # ถ้าไม่ทำท่าทางมือหรือท่าทางร่างกาย ให้เช็คสีหน้า
            results_face = face_mesh.process(image_rgb)
            if results_face.multi_face_landmarks:
                for face_landmarks in results_face.multi_face_landmarks:
                    # พิกัดสำหรับ SMILE
                    left_corner = face_landmarks.landmark[291]
                    right_corner = face_landmarks.landmark[61]
                    upper_lip = face_landmarks.landmark[13]
                    lower_lip = face_landmarks.landmark[14]
                    # พิกัดสำหรับ ANGRY (คิ้ว)
                    left_inner_brow = face_landmarks.landmark[52]
                    right_inner_brow = face_landmarks.landmark[282]
                    left_outer_brow = face_landmarks.landmark[55]
                    right_outer_brow = face_landmarks.landmark[285]
                    # พิกัดอื่นๆ
                    
                    mouth_width = ((right_corner.x - left_corner.x)**2 + (right_corner.y - left_corner.y)**2)**0.5
                    mouth_height = ((lower_lip.x - upper_lip.x)**2 + (lower_lip.y - upper_lip.y)**2)**0.5
                    
                    if mouth_width > 0:
                        mouth_aspect_ratio = mouth_height / mouth_width

                        # --- ตรวจจับ ANGRY ---
                        left_brow_diff = left_outer_brow.y - left_inner_brow.y
                        right_brow_diff = right_outer_brow.y - right_inner_brow.y
                        if left_brow_diff < ANGRY_THRESHOLD and right_brow_diff < ANGRY_THRESHOLD:
                            current_state = "ANGRY"
                            
                        # --- ตรวจจับ SAD ---
                        # ใช้หลักการปากปิด, ไม่ยิ้ม, และความสูงปาก/กว้างปากต่ำกว่าค่าปกติ
                        elif mouth_aspect_ratio < SAD_THRESHOLD and mouth_height < 0.01:
                            # เป็นการประมาณการสำหรับปากคว่ำ/บึ้ง (ปรับค่า SAD_THRESHOLD ได้)
                            current_state = "SAD"
                        
                        # --- ตรวจจับ SMILING ---
                        elif mouth_aspect_ratio > SMILE_THRESHOLD:
                            current_state = "SMILING"
                        
                        # --- สถานะปกติ ---
                        else:
                            current_state = "STRAIGHT_FACE"
                        
        # 4. เลือก Emoji สำหรับแสดงผล
        emoji_map = {
            "SMILING": "😊", "STRAIGHT_FACE": "😐", "ANGRY": "😠", "SAD": "😢", 
            "HANDS_UP": "🙌", "THUMBS_UP": "👍", "LOVE_SIGN": "🤟", 
            "MIDDLE_FINGER": "🖕", "ROCK_ON": "🤘"
        }
        
        emoji_to_display = emojis.get(current_state, blank_emoji)
        emoji_name = emoji_map.get(current_state, "❓")

        # แสดงผลบนจอ Camera Feed
        camera_frame_resized = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        
        cv2.putText(camera_frame_resized, f'STATE: {current_state} {emoji_name}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(camera_frame_resized, 'Press "q" to quit', (10, WINDOW_HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Camera Feed', camera_frame_resized)
        cv2.imshow('Emoji Output', emoji_to_display)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()