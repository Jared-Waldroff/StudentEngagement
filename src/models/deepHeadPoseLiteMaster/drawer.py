import cv2
import numpy as np
import math
import os


# Defining a class that will be used to draw and visualize the results on the images
class Drawer:

    def __getinstance(self, image, intype):
        """Loads image if it's a string path, otherwise returns the image as is."""
        if isinstance(image, intype):
            # Load and convert image if it is a file path
            image = cv2.imread(image)
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            return image
        return image

    def __draw(self, image, bbox, color=(0, 255, 0), thickness=2):
        """Draws a bounding box on the image."""
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

    def draw_bbox(self, image, bbox, color=(255, 200, 20), thickness=2, draw_person=False, draw_face=True):
        """Draws bounding boxes for persons and faces based on the bbox dictionary."""
        image = self.__getinstance(image, str)
        for key in bbox.keys():
            if "person" in bbox[key]:
                if draw_person:
                    self.__draw(image, bbox[key]["person"], color, thickness)
                if "face" in bbox[key]:
                    if draw_face:
                        self.__draw(image, bbox[key]["face"], color, thickness)
        return cv2.flip(image, 1)

    def draw_landmarks(self, image, landmarks, color=(255, 0, 0), thickness=2):
        """Draws landmarks on the image."""
        image = self.__getinstance(image, str)
        for lms in landmarks:
            if lms is not None:
                for lm in lms:
                    x, y = lm
                    cv2.circle(image, (int(x), int(y)), thickness, color, thickness)
        return cv2.flip(image, 1)

    def draw_axis(self, image, pose, landmarks, bbox, axis_size=50):
        """Draws pose axes and orientation text on the image."""
        image = self.__getinstance(image, str)
        for i, lms in enumerate(landmarks):
            if lms is not None:
                x, y = lms[0]
                yaw, pitch, roll = pose[i]
                yaw = -yaw  # Adjusting yaw

                # Rotation matrix using Rodrigues formula
                rotation_matrix = cv2.Rodrigues(np.array([pitch, yaw, roll]))[0].astype(np.float64)
                axes_points = np.array([
                    [axis_size, 0, 0],
                    [0, axis_size, 0],
                    [0, 0, axis_size],
                    [0, 0, 0]
                ], dtype=np.float64)

                # Rotating and translating axis points
                axes_points = rotation_matrix @ axes_points.T
                axes_points = (axes_points[:2, :] + np.array([[x], [y]])).astype(int)

                # Draw the axes
                cv2.line(image, tuple(axes_points[:, 3]), tuple(axes_points[:, 0]), (255, 0, 0), 3)
                cv2.line(image, tuple(axes_points[:, 3]), tuple(axes_points[:, 1]), (0, 255, 0), 3)
                cv2.line(image, tuple(axes_points[:, 3]), tuple(axes_points[:, 2]), (0, 0, 255), 3)

                # Determine where the person is looking based on yaw/pitch
                text = "Engaged"
                if pitch > 30:
                    text = 'Not Engaged'
                elif pitch < -50:
                    text = 'Not Engaged'
                elif yaw > 30:
                    text = 'Not Engaged'
                elif yaw < -30:
                    text = 'Not Engaged'

                # Draw orientation text on the image
                if i in bbox and "face" in bbox[i]:
                    x1, y1, x2, y2 = bbox[i]["face"]
                    cv2.rectangle(image, (int(x1), int(y1) - 40), (int(x1) + 140, int(y1) - 70), (255, 200, 20), -1)
                    cv2.putText(image, text, (int(x1), int(y1) - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),
                                2)

                    # Display yaw, pitch, and roll angles below the bounding box
                    yaw_deg = math.degrees(math.asin(math.sin(yaw)))
                    pitch_deg = math.degrees(math.asin(math.sin(pitch)))
                    roll_deg = math.degrees(math.asin(math.sin(roll)))

                    cv2.putText(image, f"yaw: {yaw_deg:.2f}", (int(x1), int(y2) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                (0, 255, 0), 2)
                    cv2.putText(image, f"pitch: {pitch_deg:.2f}", (int(x1), int(y2) + 35), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (255, 0, 0), 2)
                    cv2.putText(image, f"roll: {roll_deg:.2f}", (int(x1), int(y2) + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                (0, 0, 255), 2)

                    # Draw engagement score and mode on the image
                    cv2.putText(image, "engage test", (10, image.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                               (0, 255, 0), 2)
                    cv2.putText(image, "discussion test", (10, image.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                (0, 255, 0),
                                2)
                    cv2.putText(image, "mode test", (10, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0),
                                2)

        return image


# Testing the Drawer class with a sample image and bounding box
if __name__ == "__main__":
    drawer = Drawer()
    test_image_path = "path_to_your_image.jpg"
    if os.path.exists(test_image_path):
        test_image = cv2.imread(test_image_path)
        # Example bounding box and landmarks (replace with actual data)
        bbox = {0: {"person": [50, 50, 200, 400], "face": [80, 100, 150, 180]}}
        landmarks = [[(100, 120), (120, 150), (110, 160)]]
        pose = [(0.2, 0.1, 0.3)]
        # Draw bounding boxes
        drawn_image = drawer.draw_bbox(test_image, bbox)
        # Draw landmarks
        drawn_image = drawer.draw_landmarks(drawn_image, landmarks)
        # Draw axis
        drawn_image = drawer.draw_axis(drawn_image, pose, landmarks, bbox)
        # Display the result
        cv2.imshow("Drawn Image", drawn_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Image not found at path: {test_image_path}")
