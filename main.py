import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh and MediaPipe Hands
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

def ensure_four_channels(image):
    if len(image.shape) == 2:  # Grayscale image
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif image.shape[2] == 3:  # Image without alpha channel
        return cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    return image  # Image already has alpha channel

choice=int(input('1:Deadpool\n2:Wolverine\nEnter your choice:'))

# Load the image to overlay
if choice==1:
    overlay_hand_image = cv2.imread('images/shield_1.png', cv2.IMREAD_UNCHANGED)
    overlay_face_image = cv2.imread('images/deadpool1.png', cv2.IMREAD_UNCHANGED)
    ar=[9]# one shield
    head_size=2
else:
    overlay_hand_image = cv2.imread('images/claw_1.png', cv2.IMREAD_UNCHANGED)
    overlay_face_image = cv2.imread('images/wolverine_3.png', cv2.IMREAD_UNCHANGED)
    ar=[5,9,13]#claws in three fingers
    head_size=3.5

overlay_hand_image = ensure_four_channels(overlay_hand_image)
overlay_face_image = ensure_four_channels(overlay_face_image)

print(overlay_hand_image.shape)
print(overlay_face_image.shape)

# Function to check if the hand is closed (simple example)
def is_hand_closed(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    return thumb_tip < index_finger_tip

def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # Compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # Adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]
    # Perform the actual rotation and return the image
    rotated = cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_LINEAR)
    return rotated

def overlay_on_face(image, overlay, landmarks):
    h, w, _ = image.shape

    # Get coordinates for the eyes
    left_eye = landmarks[33]  # Landmark for the left eye
    right_eye = landmarks[263]  # Landmark for the right eye

    # Calculate the center between the eyes
    center_x = int((left_eye[0] + right_eye[0]) / 2)
    center_y = int((left_eye[1] + right_eye[1]) / 2)

    # Calculate width and height for the overlay
    overlay_width = int(np.linalg.norm(right_eye - left_eye) * head_size)
    aspect_ratio = overlay.shape[0] / overlay.shape[1]  # Height / Width of the overlay image
    overlay_height = int(overlay_width * aspect_ratio)

    # Calculate the angle between the eyes (in degrees)
    angle = -np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))

    # Resize overlay
    overlay_resized = cv2.resize(overlay, (overlay_width, overlay_height))

    # Rotate overlay
    overlay_resized = rotate_image(overlay_resized, angle)

    # Calculate the new dimensions of the rotated overlay
    oh, ow, _ = overlay_resized.shape

    # Calculate top-left corner of the overlay
    top_left_x = center_x - ow//2
    top_left_y = center_y - oh//2

    # Ensure the coordinates are within the image dimensions
    if top_left_x < 0 or top_left_y < 0 or top_left_x + ow > w or top_left_y + oh > h:
        return image

    # Split channels
    overlay_rgb = overlay_resized[:, :, :3]
    overlay_alpha = overlay_resized[:, :, 3] / 255.0

    # Get region of interest
    roi = image[top_left_y:top_left_y + oh, top_left_x:top_left_x + ow]

    # Blend the overlay with the ROI
    for c in range(3):
        roi[:, :, c] = (1.0 - overlay_alpha) * roi[:, :, c] + overlay_alpha * overlay_rgb[:, :, c]

    image[top_left_y:top_left_y + oh, top_left_x:top_left_x + ow] = roi

    return image

def overlay_on_hand(background, overlay, x, y):
    bg_h, bg_w, bg_channels = background.shape
    if bg_channels == 3:
        background = cv2.cvtColor(background, cv2.COLOR_BGR2BGRA)

    overlay_h, overlay_w, overlay_channels = overlay.shape

    # Ensure the coordinates are within bounds
    if x >= bg_w or y >= bg_h or x + overlay_w <= 0 or y + overlay_h <= 0:
        return background

    # Clip overlay dimensions to fit within the background
    if x + overlay_w > bg_w:
        overlay_w = bg_w - x
        overlay = overlay[:, :overlay_w]

    if y + overlay_h > bg_h:
        overlay_h = bg_h - y
        overlay = overlay[:overlay_h]

    if x < 0:
        overlay = overlay[:, -x:]
        overlay_w = overlay.shape[1]
        x = 0

    if y < 0:
        overlay = overlay[-y:, :]
        overlay_h = overlay.shape[0]
        y = 0

    if overlay_w <= 0 or overlay_h <= 0:
        return background

    overlay_image = overlay[:overlay_h, :overlay_w]



    # Apply the fade mask to the overlay image
    overlay_image = overlay_image.astype(np.float32)
    if choice==2:
        # Create a fade mask for the bottom half
        fade_mask_height = overlay_h // 2
        fade_mask = np.ones((overlay_h, overlay_w), dtype=np.float32)
        fade_values = np.linspace(1, 0, fade_mask_height)**6
        fade_mask[-fade_mask_height:] = np.tile(fade_values.reshape(-1, 1), (1, overlay_w))

        fade_mask = np.dstack((fade_mask, fade_mask, fade_mask, fade_mask))  # Repeat for all channels
        overlay_image[:, :, 3] = overlay_image[:, :, 3] * fade_mask[:, :, 3]
    else:
        overlay_image[:, :, 3] = overlay_image[:, :, 3]


    mask = overlay_image[:, :, 3:] / 255.0
    background[y:y+overlay_h, x:x+overlay_w, :3] = (1.0 - mask) * background[y:y+overlay_h, x:x+overlay_w, :3] + mask * overlay_image[:, :, :3]

    return background

def process_hand_tracking_and_face_mesh(cap, overlay_image_hand, overlay_image_face):
    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh, mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_face = face_mesh.process(frame)
            results_hand = hands.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Overlay face mesh
            if results_face.multi_face_landmarks:
                for face_landmarks in results_face.multi_face_landmarks:
                    landmarks = np.array([[lm.x * frame.shape[1], lm.y * frame.shape[0]] for lm in face_landmarks.landmark])
                    frame = overlay_on_face(frame, overlay_image_face, landmarks)

            # Overlay hand mesh
            if results_hand.multi_hand_landmarks:

                for hand_landmarks in results_hand.multi_hand_landmarks:

                    if is_hand_closed(hand_landmarks):

                        wrist_x = int(hand_landmarks.landmark[0].x * frame.shape[1])
                        wrist_y = int(hand_landmarks.landmark[0].y * frame.shape[0])

                        for i in ar:
                            mcp_x = int(hand_landmarks.landmark[i].x * frame.shape[1])
                            mcp_y = int(hand_landmarks.landmark[i].y * frame.shape[0])

                            delta_x = wrist_x - mcp_x
                            delta_y = mcp_y - wrist_y
                            angle = np.arctan2(delta_y, delta_x) * 180.0 / np.pi

                            scale_factor = 0.4
                            overlay_resized = cv2.resize(overlay_image_hand, (0, 0), fx=scale_factor, fy=scale_factor)
                            overlay_rotated = rotate_image(overlay_resized, angle+90)

                            overlay_h, overlay_w = overlay_rotated.shape[:2]
                            x = mcp_x - overlay_w // 2
                            y = mcp_y - overlay_h // 2
                            frame = overlay_on_hand(frame, overlay_rotated, x, y)


            cv2.imshow('Hand and Face Mesh Overlay', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

#camera feed
cap = cv2.VideoCapture(0)
process_hand_tracking_and_face_mesh(cap, overlay_hand_image, overlay_face_image)