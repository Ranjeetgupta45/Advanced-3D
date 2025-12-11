import os
import time

import cv2
import numpy as np
import open3d as o3d
from ultralytics import YOLO

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import mediapipe as mp  # NEW: for face + hand detection

# -----------------------------
# YOLO + model configuration
# -----------------------------

# Load YOLOv8 model (make sure yolov8n.pt is in the same folder or give full path)
model = YOLO("yolov8n.pt")
names = model.names  # class id -> class name

# Mapping from YOLO class names to 3D model files (relative paths)
MODEL_FILES = {
    "person": "models/person.obj",
    "cell phone": "models/cell_phone.obj",
}

# Stores bounding boxes for the current frame: (x1, y1, x2, y2, cls_id)
detected_boxes = []

# Will hold the relative path of the 3D model to open after a click
pending_model_rel_path = None

# -----------------------------
# Simple temporal tracking for one person (for physical action)
# -----------------------------
prev_person_center = None
prev_person_time = None

# -----------------------------
# MediaPipe face & hand models
# -----------------------------
mp_face = mp.solutions.face_detection
mp_hands = mp.solutions.hands

face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)
hand_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


# -----------------------------
# Activity inference (physical, not emotional)
# -----------------------------

def infer_activity_for_person(x1, y1, x2, y2, frame_height, frame_width,
                              center, speed_pixels_per_sec):
    """
    Heuristic-based physical activity guess for a person.
    Uses bounding box aspect ratio and movement speed over time.

    Returns strings like:
    - "walking / moving"
    - "standing still"
    - "sitting"
    - "lying down"
    """
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    aspect = h / float(w)

    # Thresholds (tunable)
    MOVE_THRESHOLD = 80.0      # pixels per second => considered "moving"
    TALL_ASPECT = 1.6
    WIDE_ASPECT = 0.7

    if speed_pixels_per_sec > MOVE_THRESHOLD:
        activity = "walking / moving"
    else:
        if aspect > TALL_ASPECT:
            activity = "standing still"
        elif aspect < WIDE_ASPECT:
            activity = "lying down"
        else:
            activity = "sitting"

    return activity


# -----------------------------
# 3D model loading & display
# -----------------------------

def open_3d_model(rel_path: str):
    """
    Load a 3D model using Open3D, then visualize it using Matplotlib 3D.
    This avoids Open3D's own GLFW window, which can fail on some Windows setups.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, rel_path)

    if not os.path.exists(full_path):
        print(f"[ERROR] 3D model file not found: {full_path}")
        return

    try:
        print(f"[3D] Loading mesh: {full_path}")
        mesh = o3d.io.read_triangle_mesh(full_path)
        if mesh.is_empty():
            print(f"[ERROR] Empty mesh: {full_path}")
            return

        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        if vertices.size == 0 or triangles.size == 0:
            print("[ERROR] Mesh has no vertices or triangles.")
            return

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        poly3d = [vertices[tri] for tri in triangles]
        collection = Poly3DCollection(poly3d, alpha=0.8)
        collection.set_edgecolor("k")
        ax.add_collection3d(collection)

        x = vertices[:, 0]
        y = vertices[:, 1]
        z = vertices[:, 2]
        ax.set_xlim([x.min(), x.max()])
        ax.set_ylim([y.min(), y.max()])
        ax.set_zlim([z.min(), z.max()])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(os.path.basename(full_path))

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"[3D VIEW ERROR] {e}")


# -----------------------------
# Mouse click handler
# -----------------------------

def click_event(event, x, y, flags, param):
    """
    Mouse callback for the OpenCV window.
    When the user clicks inside a bounding box, we mark a model to open.
    The actual model is opened in the main loop (on the main thread).
    """
    global detected_boxes, pending_model_rel_path

    if event == cv2.EVENT_LBUTTONDOWN:
        for (x1, y1, x2, y2, cls) in detected_boxes:
            if x1 <= x <= x2 and y1 <= y <= y2:
                class_name = names[int(cls)]
                print(f"[CLICK] {class_name}")

                if class_name in MODEL_FILES:
                    pending_model_rel_path = MODEL_FILES[class_name]
                    print(f"[3D] Queued model to open: {pending_model_rel_path}")
                else:
                    print("[INFO] No 3D model assigned for this class.")


# -----------------------------
# Drawing helpers for MediaPipe
# -----------------------------

def draw_faces(frame, face_results):
    h, w = frame.shape[:2]
    if not face_results.detections:
        return

    for detection in face_results.detections:
        bbox = detection.location_data.relative_bounding_box
        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)
        x2 = x1 + bw
        y2 = y1 + bh

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            frame,
            "face",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )


def draw_hands(frame, hand_results):
    h, w = frame.shape[:2]
    if not hand_results.multi_hand_landmarks:
        return

    for hand_landmarks in hand_results.multi_hand_landmarks:
        xs = [lm.x * w for lm in hand_landmarks.landmark]
        ys = [lm.y * h for lm in hand_landmarks.landmark]

        x1, x2 = int(min(xs)), int(max(xs))
        y1, y2 = int(min(ys)), int(max(ys))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(
            frame,
            "hand",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

        # Draw keypoints
        for lm in hand_landmarks.landmark:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 2, (255, 255, 0), -1)


# -----------------------------
# Main loop
# -----------------------------

def main():
    global detected_boxes, pending_model_rel_path
    global prev_person_center, prev_person_time

    cap = cv2.VideoCapture(0)  # 0 = default webcam
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    window_name = "YOLO + Face + Hand + 3D Actions"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, click_event)

    prev_time_frame = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from camera.")
            break

        frame_height, frame_width = frame.shape[:2]

        # Run YOLO detection
        results = model(frame, conf=0.35, verbose=False)[0]
        detected_boxes = []

        current_time = time.time()

        if hasattr(results, "boxes") and results.boxes is not None:
            xyxy = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy()

            # Track largest person for motion-based action
            person_indices = [
                i for i, c in enumerate(clss) if names[int(c)] == "person"
            ]

            largest_person_index = None
            largest_area = 0

            for i in person_indices:
                x1, y1, x2, y2 = xyxy[i]
                area = (x2 - x1) * (y2 - y1)
                if area > largest_area:
                    largest_area = area
                    largest_person_index = i

            # Draw YOLO detections
            for idx, (x1, y1, x2, y2, conf, cls) in enumerate(
                zip(xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3], confs, clss)
            ):
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                detected_boxes.append((x1, y1, x2, y2, cls))

                class_name = names[int(cls)]
                activity_text = ""

                if class_name == "person":
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    center = np.array([cx, cy])

                    speed_pixels_per_sec = 0.0

                    if idx == largest_person_index:
                        if prev_person_center is not None and prev_person_time is not None:
                            dt = max(1e-3, current_time - prev_person_time)
                            dist = np.linalg.norm(center - prev_person_center)
                            speed_pixels_per_sec = dist / dt
                        prev_person_center = center
                        prev_person_time = current_time

                        activity_text = infer_activity_for_person(
                            x1, y1, x2, y2,
                            frame_height, frame_width,
                            center, speed_pixels_per_sec
                        )
                    else:
                        activity_text = infer_activity_for_person(
                            x1, y1, x2, y2,
                            frame_height, frame_width,
                            center, 0.0
                        )

                if activity_text:
                    label = f"{class_name} ({activity_text}) {conf:.2f}"
                else:
                    label = f"{class_name} {conf:.2f}"

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )
                cv2.putText(
                    frame,
                    label,
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        # --- NEW: Face + Hand detection with MediaPipe ---
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_results = face_detector.process(rgb_frame)
        hand_results = hand_detector.process(rgb_frame)

        draw_faces(frame, face_results)
        draw_hands(frame, hand_results)
        # -----------------------------------------------

        # FPS display
        now = time.time()
        fps = 1.0 / (now - prev_time_frame) if now > prev_time_frame else 0.0
        prev_time_frame = now

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if pending_model_rel_path is not None:
            model_to_open = pending_model_rel_path
            pending_model_rel_path = None
            cv2.waitKey(1)
            open_3d_model(model_to_open)

        if key in (27, ord('q')):  # ESC or 'q'
            break

    cap.release()
    cv2.destroyAllWindows()

    # Clean up mediapipe
    face_detector.close()
    hand_detector.close()


if __name__ == "__main__":
    main()
