#!/usr/bin/env python3
"""
Real-time emoji display based on camera pose, facial expression, and hand gesture detection.
"""

import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Solutions
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
# เพิ่ม MediaPipe Hands สำหรับการตรวจจับท่าทางมือ
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configuration
SMILE_THRESHOLD = 0.35
SAD_MOUTH_THRESHOLD = 0.05 # ค่าประมาณสำหรับปากเศร้า
ANGRY_BROW_RATIO = 0.85 # ค่าประมาณสำหรับคิ้วขมวด
SAD_BROW_RATIO = 1.05 # ค่าประมาณสำหรับคิ้วตก

WINDOW_WIDTH = 720
WINDOW_HEIGHT = 450
EMOJI_WINDOW_SIZE = (WINDOW_WIDTH, WINDOW_HEIGHT)

# --- 1. Load emoji images (ต้องมีไฟล์ภาพเหล่านี้) ---
# เพิ่มการโหลดอีโมจิใหม่
try:
    # เดิม
    smiling_emoji = cv2.imread("smile.png")
    straight_face_emoji = cv2.imread("plain.png")
    hands_up_emoji = cv2.imread("air.png")
    
    # เพิ่มใหม่
    angry_emoji = cv2.imread("angry.png") # คิ้วขมวด
    sad_emoji = cv2.imread("sad.png")     # ปากเศร้าคิ้วตก
    thumbs_up_emoji = cv2.imread("thumbs_up.png") # ยกมือเยี่ยม
    love_sign_emoji = cv2.imread("love_sign.png") # มือรัก
    rock_on_emoji = cv2.imread("rock_on.png")     # มือ Rock
    middle_finger_emoji = cv2.imread("middle_finger.png") # โชวนิ้วกลาง

    # ตรวจสอบว่าโหลดภาพครบหรือไม่
    emojis_to_check = {
        "smile.png": smiling_emoji, "plain.png": straight_face_emoji, "air.png": hands_up_emoji,
        "angry.png": angry_emoji, "sad.png": sad_emoji, "thumbs_up.png": thumbs_up_emoji,
        "love_sign.png": love_sign_emoji, "rock_on.png": rock_on_emoji, "middle_finger.png": middle_finger_emoji,
    }

    for name, img in emojis_to_check.items():
        if img is None:
            raise FileNotFoundError(f"{name} not found or could not be loaded")

    # Resize emojis ทั้งหมด
    all_emojis = [smiling_emoji, straight_face_emoji, hands_up_emoji, angry_emoji, sad_emoji, 
                  thumbs_up_emoji, love_sign_emoji, rock_on_emoji, middle_finger_emoji]
    
    resized_emojis = [cv2.resize(img, EMOJI_WINDOW_SIZE) for img in all_emojis]
    
    (smiling_emoji, straight_face_emoji, hands_up_emoji, 
     angry_emoji, sad_emoji, thumbs_up_emoji, 
     love_sign_emoji, rock_on_emoji, middle_finger_emoji) = resized_emojis

except Exception as e:
    print("Error loading emoji images! Please ensure all files are in the directory.")
    print(f"Details: {e}")
    print("\nExpected files:")
    print("- smile.png, plain.png, air.png (เดิม)")
    print("- angry.png, sad.png, thumbs_up.png, love_sign.png, rock_on.png, middle_finger.png (ใหม่)")
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
print("  Raise hands above shoulders for hands up")
print("  Smile for smiling emoji")
print("  Straight face for neutral emoji")
print("  New Gestures: Angry (คิ้วขมวด), Sad (ปากเศร้าคิ้วตก), Thumbs Up, Love Sign, Rock On, Middle Finger")

# --- 2. ฟังก์ชันช่วยสำหรับการตรวจจับท่าทางมือ (Hand Gestures) ---
def check_finger_raised(landmark_list, finger_tip, finger_pip, finger_mcp):
    """ตรวจสอบว่านิ้วยกขึ้นหรือไม่ (ยกเว้นนิ้วโป้ง)"""
    return landmark_list[finger_tip].y < landmark_list[finger_pip].y

def check_thumb_raised(landmark_list):
    """ตรวจสอบว่านิ้วโป้งยกขึ้นหรือไม่"""
    return landmark_list[mp_hands.HandLandmark.THUMB_TIP].x < landmark_list[mp_hands.HandLandmark.THUMB_MCP].x # ทิศทางอาจต้องปรับตามการหันของมือ

def get_hand_gesture(hand_landmarks):
    """ระบุท่าทางมือจาก landmark"""
    if not hand_landmarks:
        return None
    
    landmark_list = hand_landmarks.landmark
    
    # ตรวจสอบนิ้ว
    is_thumb_up = landmark_list[mp_hands.HandLandmark.THUMB_TIP].y < landmark_list[mp_hands.HandLandmark.THUMB_IP].y and \
                  landmark_list[mp_hands.HandLandmark.THUMB_TIP].x < landmark_list[mp_hands.HandLandmark.THUMB_MCP].x # ปรับเงื่อนไขนิ้วโป้ง

    is_index_raised = check_finger_raised(landmark_list, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.INDEX_FINGER_MCP)
    is_middle_raised = check_finger_raised(landmark_list, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP)
    is_ring_raised = check_finger_raised(landmark_list, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP, mp_hands.HandLandmark.RING_FINGER_MCP)
    is_pinky_raised = check_finger_raised(landmark_list, mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP, mp_hands.HandLandmark.PINKY_MCP)
    
    # THUMBS UP
    # นิ้วโป้งขึ้น, นิ้วอื่นงอลง
    if is_thumb_up and not is_index_raised and not is_middle_raised and not is_ring_raised and not is_pinky_raised:
        return "THUMBS_UP"

    # ROCK ON
    # นิ้วชี้, นิ้วก้อย, นิ้วโป้งขึ้น
    if is_index_raised and not is_middle_raised and not is_ring_raised and is_pinky_raised and is_thumb_up:
        return "ROCK_ON"

    # MIDDLE FINGER
    # นิ้วกลางขึ้นโดดเด่น
    if is_middle_raised and not is_index_raised and not is_ring_raised and not is_pinky_raised:
        return "MIDDLE_FINGER"
        
    # LOVE SIGN (Korean Heart)
    # นิ้วชี้และนิ้วโป้งเกือบจะแตะกัน หรือคาดว่านิ้วอื่นงอลง
    # นี่คือการประมาณที่ต้องอาศัยการทดลองจริง
    thumb_tip = landmark_list[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmark_list[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    distance = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5
    
    if distance < 0.05 and not is_middle_raised and not is_ring_raised and not is_pinky_raised: # ค่า 0.05 ต้องปรับให้เหมาะสม
        return "LOVE_SIGN"
        
    return None

# --- 3. ฟังก์ชันช่วยสำหรับการตรวจจับการแสดงออกทางสีหน้า (Facial Expressions) ---
def get_face_expression(face_landmarks):
    """
    ระบุการแสดงออกทางสีหน้า: SMILING, ANGRY, SAD, หรือ STRAIGHT_FACE (ตามลำดับความสำคัญ)
    """
    if not face_landmarks:
        return "STRAIGHT_FACE"

    # กำหนด landmark ที่สำคัญ
    left_corner = face_landmarks.landmark[291]
    right_corner = face_landmarks.landmark[61]
    upper_lip = face_landmarks.landmark[13]
    lower_lip = face_landmarks.landmark[14]

    # คิ้วซ้าย (Brow Landmarks) - [55, 65, 52] [35, 105, 66]
    left_brow_inner = face_landmarks.landmark[105] 
    left_brow_outer = face_landmarks.landmark[52] 
    
    # หัวตาด้านใน (Inner Eye Corner)
    left_inner_eye = face_landmarks.landmark[374]
    right_inner_eye = face_landmarks.landmark[145]

    # คำนวณอัตราส่วนปาก
    mouth_width = ((right_corner.x - left_corner.x)**2 + (right_corner.y - left_corner.y)**2)**0.5
    mouth_height = ((lower_lip.x - upper_lip.x)**2 + (lower_lip.y - upper_lip.y)**2)**0.5
    
    if mouth_width > 0:
        mouth_aspect_ratio = mouth_height / mouth_width
    else:
        mouth_aspect_ratio = 0

    # คำนวณอัตราส่วนคิ้ว (Angry: คิ้วเข้าหากัน, Sad: คิ้วตก)
    # ใช้อัตราส่วนระยะห่างระหว่างจุดคิ้วกับหัวตาด้านใน
    brow_to_eye_dist = ((left_brow_inner.x - left_inner_eye.x)**2 + (left_brow_inner.y - left_inner_eye.y)**2)**0.5
    eye_width = ((left_inner_eye.x - left_brow_outer.x)**2 + (left_inner_eye.y - left_brow_outer.y)**2)**0.5
    
    if eye_width > 0:
        brow_aspect_ratio = brow_to_eye_dist / eye_width
    else:
        brow_aspect_ratio = 1.0 # ค่าเริ่มต้น

    # 1. SMILING
    if mouth_aspect_ratio > SMILE_THRESHOLD:
        return "SMILING"
        
    # 2. ANGRY (คิ้วขมวด)
    # อัตราส่วนคิ้ว/หัวตา น้อยกว่าค่าปกติ (คิ้วเข้าใกล้ตา)
    if brow_aspect_ratio < ANGRY_BROW_RATIO:
        return "ANGRY"

    # 3. SAD (ปากเศร้าคิ้วตก)
    # ปากคว่ำ (ตรวจจับได้ยากกว่าการหุบ/อ้าปาก), และคิ้วตก (brow_aspect_ratio มากกว่าค่าปกติ)
    if mouth_aspect_ratio < SAD_MOUTH_THRESHOLD and brow_aspect_ratio > SAD_BROW_RATIO:
        return "SAD"
        
    # 4. STRAIGHT FACE
    return "STRAIGHT_FACE"


# --- 4. Main loop ---
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5) as face_mesh, \
     mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands: # เพิ่ม Hands

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        current_state = "STRAIGHT_FACE"
        emoji_name = "😐"
        
        # --- A. Check for HAND GESTURES (ท่าทางมือ) ---
        results_hands = hands.process(image_rgb)
        gesture = None
        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                gesture = get_hand_gesture(hand_landmarks)
                if gesture:
                    current_state = gesture
                    break # ใช้ท่าทางมือแรกที่ตรวจจับได้

        # --- B. Check for BODY POSE (ท่าทางร่างกาย) ---
        if not gesture: # ตรวจท่าทางร่างกายถ้าไม่พบท่าทางมือ
            results_pose = pose.process(image_rgb)
            if results_pose.pose_landmarks:
                landmarks = results_pose.pose_landmarks.landmark
                
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
                right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

                # HANDS UP
                if (left_wrist.y < left_shoulder.y) or (right_wrist.y < right_shoulder.y):
                    current_state = "HANDS_UP"
        
        # --- C. Check FACIAL EXPRESSION (การแสดงออกทางสีหน้า) ---
        if current_state == "STRAIGHT_FACE": # ตรวจสีหน้าก็ต่อเมื่อยังไม่พบท่าทางอื่น (มือ/ร่างกาย)
            results_face = face_mesh.process(image_rgb)
            if results_face.multi_face_landmarks:
                # ใช้ฟังก์ชันใหม่เพื่อตรวจจับสีหน้า
                current_state = get_face_expression(results_face.multi_face_landmarks[0])


        # --- D. Select emoji based on state ---
        if current_state == "SMILING":
            emoji_to_display = smiling_emoji
            emoji_name = "😊"
        elif current_state == "ANGRY":
            emoji_to_display = angry_emoji
            emoji_name = "😡"
        elif current_state == "SAD":
            emoji_to_display = sad_emoji
            emoji_name = "😢"
        elif current_state == "THUMBS_UP":
            emoji_to_display = thumbs_up_emoji
            emoji_name = "👍"
        elif current_state == "LOVE_SIGN":
            emoji_to_display = love_sign_emoji
            emoji_name = "🫰" # Korean heart sign
        elif current_state == "ROCK_ON":
            emoji_to_display = rock_on_emoji
            emoji_name = "🤘"
        elif current_state == "MIDDLE_FINGER":
            emoji_to_display = middle_finger_emoji
            emoji_name = "🖕"
        elif current_state == "HANDS_UP":
            emoji_to_display = hands_up_emoji
            emoji_name = "🙌"
        elif current_state == "STRAIGHT_FACE":
            emoji_to_display = straight_face_emoji
            emoji_name = "😐"
        else:
            emoji_to_display = blank_emoji
            emoji_name = "❓"

        # แสดงผล
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