import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -----------------------------
# Model path
# -----------------------------

MODEL_PATH = "pose_landmarker_full.task"


# -----------------------------
# Base options
# -----------------------------

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)


# -----------------------------
# Pose Landmarker options
# -----------------------------

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1
)


# -----------------------------
# Create Pose Landmarker
# -----------------------------

landmarker = vision.PoseLandmarker.create_from_options(
    options
)


# -----------------------------
# Open camera
# -----------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera could not be opened")
    exit()

print("✅ Camera started")
print("Press ESC to quit")


frame_timestamp_ms = 0


# -----------------------------
# Camera loop
# -----------------------------

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


    # -----------------------------
    # Convert to MediaPipe Image
    # -----------------------------

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # Timestamp
    frame_timestamp_ms += 33


    # -----------------------------
    # Detect pose
    # -----------------------------

    result = landmarker.detect_for_video(
        mp_image,
        frame_timestamp_ms
    )


    # -----------------------------
    # Draw pose landmarks
    # -----------------------------

    if result.pose_landmarks:

        for pose_landmarks in result.pose_landmarks:

            h, w, _ = frame.shape

            # Draw points
            for landmark in pose_landmarks:

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )


            # -----------------------------
            # Draw connections
            # -----------------------------

            connections = vision.PoseLandmarksConnections.POSE_LANDMARKS

            for connection in connections:

                start = pose_landmarks[
                    connection.start
                ]

                end = pose_landmarks[
                    connection.end
                ]

                start_point = (
                    int(start.x * w),
                    int(start.y * h)
                )

                end_point = (
                    int(end.x * w),
                    int(end.y * h)
                )

                cv2.line(
                    frame,
                    start_point,
                    end_point,
                    (255, 0, 0),
                    2
                )


    # -----------------------------
    # Display
    # -----------------------------

    cv2.imshow(
        "MediaPipe Pose Detection",
        frame
    )


    # ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break


# -----------------------------
# Release resources
# -----------------------------

cap.release()
cv2.destroyAllWindows()

landmarker.close()

print("✅ Camera closed")