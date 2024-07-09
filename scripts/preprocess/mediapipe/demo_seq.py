#@markdown We implemented some functions to visualize the face landmark detection results. <br/> Run the following cell to activate the functions.

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt

def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_iris_connections_style())

    return annotated_image

def plot_face_blendshapes_bar_graph(face_blendshapes):
    # Extract the face blendshapes category names and scores.
    face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
    face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
    # The blendshapes are ordered in decreasing score value.
    face_blendshapes_ranks = range(len(face_blendshapes_names))

    fig, ax = plt.subplots(figsize=(12, 12))
    bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
    ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
    ax.invert_yaxis()

    # Label each bar with values
    for score, patch in zip(face_blendshapes_scores, bar.patches):
        plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

    ax.set_xlabel('Score')
    ax.set_title("Face Blendshapes")
    plt.tight_layout()
    plt.show()


def draw_landmarks_on_image2(rgb_image, detection_result, height, width):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)
    lms_mp = []

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        for landmark in face_landmarks:
            x, y, z =landmark.x, landmark.y, landmark.z
            lms_mp.append( np.array([x, y], dtype='float32') )
            x_pred = int(x * width)
            y_pred = int(y * height)
            cv2.circle(annotated_image, (x_pred, y_pred), 1, (0, 0, 255), 2)
    
    lms_mp = np.stack(lms_mp, axis=0)
    return annotated_image, lms_mp

import cv2
#from google.colab.patches import cv2_imshow

# img = cv2.imread("image.jpg")
# cv2_imshow(img)

# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import sys
if not len(sys.argv) == 2:
    print('Format:')
    print('python demo_seq.py input_dir')
    exit(0)
input_dir = sys.argv[1]
output_dir = input_dir

annotated_image_dir = os.path.join(output_dir, "Mediapipe_annotated_images")
landmark_dir = os.path.join(output_dir, "Mediapipe_landmarks")
os.makedirs(annotated_image_dir, exist_ok=True)
os.makedirs(landmark_dir, exist_ok=True)

# STEP 2: Create an FaceLandmarker object.
base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

for fns in sorted(os.listdir(input_dir)):
    if fns.endswith('.png'): 
        if len(fns.split("_")) == 1:
            idx = int(fns[:5])  
        else:
            idx = int(fns[4:-4])
        img_path = os.path.join(input_dir, fns)
        img = cv2.imread(img_path)        
        height, width, channels = img.shape
        print(idx, height, width, channels) 
        
        # STEP 3: Load the input image.
        image = mp.Image.create_from_file(img_path)

        # STEP 4: Detect face landmarks from the input image.
        detection_result = detector.detect(image)

        # STEP 5: Process the detection result. In this case, visualize it.
        annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

        annotated_image_file = os.path.join(annotated_image_dir, "{:05d}.png".format( idx ))
        cv2.imwrite(annotated_image_file, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

        annotated_image2, lms_mp = draw_landmarks_on_image2(img, detection_result, height, width)

        landmark_file = os.path.join(landmark_dir, "{:05d}.npy".format( idx ))
        np.save(landmark_file, lms_mp)
        print(input_dir, fns, lms_mp.shape, "done!")