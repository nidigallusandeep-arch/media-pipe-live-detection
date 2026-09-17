import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import mediapipe as mp

st.set_page_config(
    page_title="MediaPipe Computer Vision",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 MediaPipe Computer Vision")
st.write("Live Hand, Pose and Face Detection")

# ==============================
# MODEL PATHS
# ==============================

HAND_MODEL = "hand_landmarker.task"
POSE_MODEL = "pose_landmarker_full.task"
FACE_MODEL = "face_landmarker.task"


# ==============================
# CREATE HAND DETECTOR
# ==============================

@st.cache_resource
def create_hand_detector():

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=HAND_MODEL
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2
    )

    return mp.tasks.vision.HandLandmarker.create_from_options(
        options
    )


# ==============================
# CREATE POSE DETECTOR
# ==============================

@st.cache_resource
def create_pose_detector():

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=POSE_MODEL
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_poses=2
    )

    return mp.tasks.vision.PoseLandmarker.create_from_options(
        options
    )


# ==============================
# CREATE FACE DETECTOR
# ==============================

@st.cache_resource
def create_face_detector():

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=FACE_MODEL
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1
    )

    return mp.tasks.vision.FaceLandmarker.create_from_options(
        options
    )


# ==============================
# DRAW HAND
# ==============================

def draw_hand(image, result):

    h, w, _ = image.shape

    if result.hand_landmarks:

        for hand in result.hand_landmarks:

            for landmark in hand:

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                if 0 <= x < w and 0 <= y < h:

                    cv2.circle(
                        image,
                        (x, y),
                        4,
                        (0, 255, 0),
                        -1
                    )

            # Connect hand landmarks
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20)
            ]

            for start, end in connections:

                x1 = int(hand[start].x * w)
                y1 = int(hand[start].y * h)

                x2 = int(hand[end].x * w)
                y2 = int(hand[end].y * h)

                cv2.line(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

        cv2.putText(
            image,
            "HAND DETECTED",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3
        )

    else:

        cv2.putText(
            image,
            "NO HAND",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    return image


# ==============================
# DRAW POSE
# ==============================

def draw_pose(image, result):

    h, w, _ = image.shape

    if result.pose_landmarks:

        for pose in result.pose_landmarks:

            for landmark in pose:

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                if 0 <= x < w and 0 <= y < h:

                    cv2.circle(
                        image,
                        (x, y),
                        4,
                        (255, 0, 0),
                        -1
                    )

            # Simple body connections
            connections = [
                (11, 12),
                (11, 13),
                (13, 15),
                (12, 14),
                (14, 16),
                (11, 23),
                (12, 24),
                (23, 24),
                (23, 25),
                (25, 27),
                (24, 26),
                (26, 28)
            ]

            for start, end in connections:

                x1 = int(pose[start].x * w)
                y1 = int(pose[start].y * h)

                x2 = int(pose[end].x * w)
                y2 = int(pose[end].y * h)

                cv2.line(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    3
                )

        cv2.putText(
            image,
            "POSE DETECTED",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            3
        )

    else:

        cv2.putText(
            image,
            "NO POSE",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    return image


# ==============================
# DRAW FACE
# ==============================

def draw_face(image, result):

    h, w, _ = image.shape

    if result.face_landmarks:

        for face in result.face_landmarks:

            for landmark in face:

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                if 0 <= x < w and 0 <= y < h:

                    cv2.circle(
                        image,
                        (x, y),
                        2,
                        (0, 255, 255),
                        -1
                    )

        cv2.putText(
            image,
            "FACE DETECTED",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            3
        )

    else:

        cv2.putText(
            image,
            "NO FACE",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    return image


# ==============================
# SIDEBAR
# ==============================

project = st.sidebar.selectbox(
    "Select Detection",
    [
        "Face Detection",
        "Hand Detection",
        "Pose Detection",
        "All Detection"
    ]
)

st.sidebar.write(
    "Selected:",
    project
)


# ==============================
# LOAD DETECTORS
# ==============================

if project == "Face Detection":
    face_detector = create_face_detector()

elif project == "Hand Detection":
    hand_detector = create_hand_detector()

elif project == "Pose Detection":
    pose_detector = create_pose_detector()

else:
    face_detector = create_face_detector()
    hand_detector = create_hand_detector()
    pose_detector = create_pose_detector()


# ==============================
# VIDEO CALLBACK
# ==============================

def video_frame_callback(frame):

    image = frame.to_ndarray(
        format="bgr24"
    )

    image = cv2.flip(
        image,
        1
    )

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # FACE
    if project == "Face Detection":

        result = face_detector.detect(
            mp_image
        )

        image = draw_face(
            image,
            result
        )

    # HAND
    elif project == "Hand Detection":

        result = hand_detector.detect(
            mp_image
        )

        image = draw_hand(
            image,
            result
        )

    # POSE
    elif project == "Pose Detection":

        result = pose_detector.detect(
            mp_image
        )

        image = draw_pose(
            image,
            result
        )

    # ALL
    elif project == "All Detection":

        face_result = face_detector.detect(
            mp_image
        )

        hand_result = hand_detector.detect(
            mp_image
        )

        pose_result = pose_detector.detect(
            mp_image
        )

        image = draw_face(
            image,
            face_result
        )

        image = draw_hand(
            image,
            hand_result
        )

        image = draw_pose(
            image,
            pose_result
        )

    return av.VideoFrame.from_ndarray(
        image,
        format="bgr24"
    )


# ==============================
# WEBRTC CAMERA
# ==============================

webrtc_streamer(
    key="mediapipe-camera",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": True,
        "audio": False
    }
)

st.info(
    "Click START and allow camera permission."
)