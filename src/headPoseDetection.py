import numpy as np
import cv2
import mediapipe as mp
import os
import argparse

# Initialize MediaPipe Face Mesh and Drawing tools
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(color=(128, 0, 128), thickness=2, circle_radius=1)


# Define a function to get the video capture source
def get_source(source):
    if str(source).isdigit():
        return cv2.VideoCapture(int(source))  # Webcam if source is a number like 0
    elif os.path.isfile(source):  # If it is a file
        if source.lower().endswith(('.png', '.jpg', '.jpeg')):  # Image file
            return cv2.imread(source)
        else:  # Video file
            return cv2.VideoCapture(source)
    else:
        raise ValueError("Invalid source! Please provide a valid webcam index, video, or image file.")


def run(source):
    # Initialize the capture source
    capture = get_source(source)

    # Check if it is a video stream or an image
    is_video = isinstance(capture, cv2.VideoCapture)

    while True:
        if is_video:
            ret, image = capture.read()
            if not ret:
                break
        else:
            image = capture  # It's a static image

        # Process the image for face landmarks
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        img_h, img_w, img_c = image.shape
        face_2d = []
        face_3d = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx in [33, 263, 1, 61, 291, 199]:  # Landmarks for nose, eyes, etc.
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = 1 * img_w
                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])
                distortion_matrix = np.zeros((4, 1), dtype=np.float64)

                success, rotation_vec, translation_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, distortion_matrix)

                # Getting the rotation of the face
                rmat, jac = cv2.Rodrigues(rotation_vec)
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                # Determine head direction
                if y < -10:
                    text = "Looking Left"
                elif y > 10:
                    text = "Looking Right"
                elif x < -10:
                    text = "Looking Down"
                elif x > 10:
                    text = "Looking Up"
                else:
                    text = "Forward"

                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rotation_vec, translation_vec, cam_matrix,
                                                                 distortion_matrix)
                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

                cv2.line(image, p1, p2, (255, 0, 0), 3)
                cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

        # Display the processed image
        cv2.imshow('Head Pose Detection', image)

        if cv2.waitKey(5) & 0xFF == 27:  # Exit on pressing ESC
            break

#        if not is_video:
#            break  # If it's an image, break after one frame

    if is_video:
        capture.release()

#    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Argument parsing for command-line input
    parser = argparse.ArgumentParser(description='Head Pose Detection')
    parser.add_argument('--source', type=str, required=True,
                        help='Source of input (webcam index, video file, or image)')

    args = parser.parse_args()

    # Run the function with the source provided by the user
    run(args.source)

