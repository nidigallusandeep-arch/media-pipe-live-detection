import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# --------------------------------
# 1. Face model path
# --------------------------------

MODEL_PATH = "face_landmarker.task"


# --------------------------------
# 2. Base options
# --------------------------------

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)


# --------------------------------
# 3. Face Landmarker options
# --------------------------------

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)


# --------------------------------
# 4. Create Face Landmarker
# --------------------------------

landmarker = vision.FaceLandmarker.create_from_options(
    options
)


# --------------------------------
# 5. Open camera
# --------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera could not be opened")
    exit()

print("✅ Camera started")
print("Press ESC to quit")


frame_timestamp_ms = 0


# --------------------------------
# 6. Camera loop
# --------------------------------

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        print("❌ Failed to read camera")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------
    # 7. Convert to MediaPipe Image
    # --------------------------------

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # Timestamp
    frame_timestamp_ms += 33


    # --------------------------------
    # 8. Detect face
    # --------------------------------

    result = landmarker.detect_for_video(
        mp_image,
        frame_timestamp_ms
    )


    # --------------------------------
    # 9. Draw face landmarks
    # --------------------------------

    if result.face_landmarks:

        for face_landmarks in result.face_landmarks:

            h, w, _ = frame.shape

            for landmark in face_landmarks:

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                cv2.circle(
                    frame,
                    (x, y),
                    1,
                    (0, 255, 0),
                    -1
                )


    # --------------------------------
    # 10. Display
    # --------------------------------

    cv2.imshow(
        "MediaPipe Face Detection",
        frame
    )


    # ESC to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break


# --------------------------------
# 11. Release
# --------------------------------

cap.release()
cv2.destroyAllWindows()

landmarker.close()

print("✅ Camera closed")