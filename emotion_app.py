import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)

    emotion = "Detecting..."

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            h, w, _ = frame.shape

            # Eye corner landmarks for scaling
            left_eye_outer = landmarks[33]
            right_eye_outer = landmarks[263]
            lx = int(left_eye_outer.x * w)
            ly = int(left_eye_outer.y * h)
            rx = int(right_eye_outer.x * w)
            ry = int(right_eye_outer.y * h)

            eye_distance = ((rx - lx) ** 2 + (ry - ly) ** 2) ** 0.5

            # Mouth landmarks
            ml = (int(landmarks[61].x * w), int(landmarks[61].y * h))
            mr = (int(landmarks[291].x * w), int(landmarks[291].y * h))
            ul = (int(landmarks[13].x * w), int(landmarks[13].y * h))
            ll = (int(landmarks[14].x * w), int(landmarks[14].y * h))

            mouth_width = abs(mr[0] - ml[0])
            mouth_open = abs(ll[1] - ul[1])

            norm_mouth_width = mouth_width / eye_distance
            norm_mouth_open = mouth_open / eye_distance

            # Eyebrows and eyes (for emotion fallback)
            left_eyebrow = landmarks[65]
            left_eye = landmarks[159]
            right_eyebrow = landmarks[295]
            right_eye = landmarks[386]

            le_eyebrow_y = int(left_eyebrow.y * h)
            le_eye_y = int(left_eye.y * h)
            re_eyebrow_y = int(right_eyebrow.y * h)
            re_eye_y = int(right_eye.y * h)

            left_diff = le_eye_y - le_eyebrow_y
            right_diff = re_eye_y - re_eyebrow_y

            # Emotion detection using normalized values
            if norm_mouth_width > 1.8 and norm_mouth_open > 0.6:
                emotion = "Happy 😊"
            elif norm_mouth_open > 0.9:
                emotion = "Surprised 😲"
            elif norm_mouth_open < 0.3 and norm_mouth_width > 1.6:
                emotion = "Neutral 😐"
            else:
                if left_diff < 10 and right_diff < 10:
                    emotion = "Angry 😠"
                elif abs(left_diff - right_diff) > 10:
                    emotion = "Curious 🤨"
                elif left_diff > 15 and right_diff > 15:
                    emotion = "Sad 😢"
                else:
                    emotion = "Confused 🤔"

            # Drawing landmarks for reference
            cv2.circle(frame, ml, 3, (0, 255, 0), -1)
            cv2.circle(frame, mr, 3, (0, 255, 0), -1)
            cv2.circle(frame, ul, 3, (0, 0, 255), -1)
            cv2.circle(frame, ll, 3, (0, 0, 255), -1)
            cv2.circle(frame, (lx, ly), 3, (255, 0, 0), -1)
            cv2.circle(frame, (rx, ry), 3, (255, 0, 0), -1)

            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

            # Display values and emotion
            cv2.putText(frame, f'Mouth Width: {round(norm_mouth_width, 2)}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f'Mouth Open: {round(norm_mouth_open, 2)}', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Show emotion
    cv2.putText(frame, f'Emotion: {emotion}', (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Show window
    cv2.imshow("Live Emotion Detection (Normalized)", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
