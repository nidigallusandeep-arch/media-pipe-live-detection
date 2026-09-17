import streamlit as st
import cv2
import av
import mediapipe as mp

from streamlit_webrtc import webrtc_streamer, WebRtcMode


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="MediaPipe Computer Vision",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🤖 MediaPipe Computer Vision")
st.write("Live Hand, Pose and Face Detection")


# ============================================================
# MODEL PATHS
# ============================================================

HAND_MODEL = "hand_landmarker.task"
POSE_MODEL = "pose_landmarker_full.task"
FACE_MODEL = "face_landmarker.task"


# ============================================================
# MEDIAPIPE IMPORTS
# ============================================================

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# ============================================================
# CREATE HAND LANDMARKER
# ============================================================

hand_options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=HAND_MODEL
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2
)

hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
    hand_options
)


# ============================================================
# CREATE POSE LANDMARKER
# ============================================================

pose_options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=POSE_MODEL
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_poses=2
)

pose_landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(
    pose_options
)


# ============================================================
# CREATE FACE LANDMARKER
# ============================================================

face_options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=FACE_MODEL
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=2,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)

face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
    face_options
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Select Detection")

    selected_mode = st.selectbox(
        "Detection Type",
        [
            "Face Detection",
            "Hand Detection",
            "Pose Detection",
            "All Detection"
        ]
    )

    st.write(f"Selected: **{selected_mode}**")


# ============================================================
# HAND CONNECTIONS
# ============================================================

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),

    (5, 9),
    (9, 13),
    (13, 17)
]


# ============================================================
# POSE CONNECTIONS
# ============================================================

POSE_CONNECTIONS = [
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
    (26, 28),

    (27, 29),
    (29, 31),

    (28, 30),
    (30, 32)
]


# ============================================================
# DRAW HAND LANDMARKS
# ============================================================

def draw_hand_landmarks(image, result):

    if not result.hand_landmarks:
        return

    height, width, _ = image.shape

    for hand in result.hand_landmarks:

        points = []

        for landmark in hand:

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            points.append((x, y))

            cv2.circle(
                image,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )

        for start, end in HAND_CONNECTIONS:

            if start < len(points) and end < len(points):

                cv2.line(
                    image,
                    points[start],
                    points[end],
                    (255, 0, 0),
                    2
                )


# ============================================================
# DRAW POSE LANDMARKS
# ============================================================

def draw_pose_landmarks(image, result):

    if not result.pose_landmarks:
        return

    height, width, _ = image.shape

    for pose in result.pose_landmarks:

        points = []

        for landmark in pose:

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            points.append((x, y))

            cv2.circle(
                image,
                (x, y),
                5,
                (0, 255, 255),
                -1
            )

        for start, end in POSE_CONNECTIONS:

            if start < len(points) and end < len(points):

                cv2.line(
                    image,
                    points[start],
                    points[end],
                    (255, 0, 255),
                    3
                )


# ============================================================
# DRAW FACE LANDMARKS
# ============================================================

def draw_face_landmarks(image, result):

    if not result.face_landmarks:
        return

    height, width, _ = image.shape

    for face in result.face_landmarks:

        for landmark in face:

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            cv2.circle(
                image,
                (x, y),
                1,
                (0, 255, 0),
                -1
            )


# ============================================================
# PROCESS FRAME
# ============================================================

def process_frame(frame):

    image = frame.to_ndarray(format="bgr24")

    # Convert BGR -> RGB
    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_image
    )


    # ========================================================
    # FACE
    # ========================================================

    if selected_mode in [
        "Face Detection",
        "All Detection"
    ]:

        face_result = face_landmarker.detect(mp_image)

        draw_face_landmarks(
            image,
            face_result
        )


    # ========================================================
    # HAND
    # ========================================================

    if selected_mode in [
        "Hand Detection",
        "All Detection"
    ]:

        hand_result = hand_landmarker.detect(mp_image)

        draw_hand_landmarks(
            image,
            hand_result
        )


    # ========================================================
    # POSE
    # ========================================================

    if selected_mode in [
        "Pose Detection",
        "All Detection"
    ]:

        pose_result = pose_landmarker.detect(mp_image)

        draw_pose_landmarks(
            image,
            pose_result
        )


    # ========================================================
    # RETURN FRAME
    # ========================================================

    return av.VideoFrame.from_ndarray(
        image,
        format="bgr24"
    )


# ============================================================
# WEBRTC CONFIGURATION
# ============================================================

RTC_CONFIGURATION = {
    "iceServers": [
        {
            "urls": [
                "stun:stun.l.google.com:19302"
            ]
        }
    ]
}


# ============================================================
# WEBRTC STREAMER
# ============================================================

webrtc_streamer(
    key="mediapipe-computer-vision",
    mode=WebRtcMode.SENDRECV,

    rtc_configuration=RTC_CONFIGURATION,

    media_stream_constraints={
        "video": True,
        "audio": False
    },

    video_frame_callback=process_frame,

    async_processing=True
)


# ============================================================
# INFORMATION
# ============================================================

st.info(
    "Click START and allow camera permission."
)