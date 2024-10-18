import cv2
import numpy as np
import math
import os

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

    def draw_bbox(self, image, bbox, face_id, color=(255, 200, 20), thickness=2, draw_person=False, draw_face=True):
        """Draws bounding boxes for persons and faces based on the bbox dictionary."""
        image = self.__getinstance(image, str)
        if "person" in bbox:
            if draw_person:
                self.__draw(image, bbox["person"], color, thickness)
        if "face" in bbox:
            if draw_face:
                self.__draw(image, bbox["face"], color, thickness)
                # Add Face ID label
                x1, y1, x2, y2 = bbox["face"]
                cv2.putText(image, f"Face {face_id}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
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

    def draw_axis(self, image, pose, landmarks, bbox, dominant_emotion, face_id, axis_size=100):
        """Draws pose axes and orientation text on the image."""
        image = self.__getinstance(image, str)
        for i, lms in enumerate(landmarks):
            if lms is not None:
                x, y = lms[0]
                yaw, pitch, roll = pose[i]

                # Convert angles from degrees to radians
                yaw_rad = np.deg2rad(yaw)
                pitch_rad = np.deg2rad(pitch)
                roll_rad = np.deg2rad(roll)

                # Adjust the yaw angle (depending on coordinate system)
                yaw_rad = -yaw_rad

                # Rotation matrix using Rodrigues formula
                rotation_vector = np.array([pitch_rad, yaw_rad, roll_rad])
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

                # Define axis points in 3D space
                axis = np.float32([[axis_size, 0, 0],
                                   [0, axis_size, 0],
                                   [0, 0, axis_size],
                                   [0, 0, 0]]).reshape(-1, 3)

                # Project 3D points to 2D image plane
                img_points, _ = cv2.projectPoints(axis, rotation_vector, np.array([0, 0, 0], dtype=float),
                                                  np.eye(3), np.zeros((4, 1)))
                img_points = img_points.reshape(-1, 2)

                # Shift points to the landmark position
                img_points += np.array([x, y])

                origin = tuple(img_points[3].astype(int))
                x_axis = tuple(img_points[0].astype(int))
                y_axis = tuple(img_points[1].astype(int))
                z_axis = tuple(img_points[2].astype(int))

                # Draw the axes
                cv2.line(image, origin, x_axis, (0, 0, 255), 2)  # X-axis (red)
                cv2.line(image, origin, y_axis, (0, 255, 0), 2)  # Y-axis (green)
                cv2.line(image, origin, z_axis, (255, 0, 0), 2)  # Z-axis (blue)

                # Determine engagement status
                engaged_text = "Engaged"
                if pitch > 30 or pitch < -50 or yaw > 30 or yaw < -30:
                    engaged_text = 'Not Engaged'

                # Draw 'Emotion' and 'Engaged' labels
                if "face" in bbox:
                    x1, y1, x2, y2 = bbox["face"]
                    # Emotion label
                    cv2.rectangle(image, (int(x1), int(y1) - 40), (int(x1) + 200, int(y1) - 0), (0, 200, 200), -1)
                    cv2.putText(image, f"Emotion: {dominant_emotion}", (int(x1) + 10, int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

                    # Engaged label (above Emotion)
                    cv2.rectangle(image, (int(x1), int(y1) - 80), (int(x1) + 200, int(y1) - 40), (255, 200, 20), -1)
                    cv2.putText(image, engaged_text, (int(x1) + 10, int(y1) - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

                    # Display yaw, pitch, and roll angles below the bounding box
                    cv2.putText(image, f"yaw: {yaw:.2f}", (int(x1), int(y2) + 0), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                (0, 255, 0), 2)
                    cv2.putText(image, f"pitch: {pitch:.2f}", (int(x1), int(y2) + 25), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (255, 0, 0), 2)
                    cv2.putText(image, f"roll: {roll:.2f}", (int(x1), int(y2) + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                (0, 0, 255), 2)

                    # **Draw Face ID directly above the face without a green box**
                    text = f'ID: {face_id}'
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    font_thickness = 2
                    text_color = (255, 255, 255)  # White color for text

                    # Get the size of the text
                    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
                    # Position the text above the bounding box
                    text_x = x1 - 30
                    text_y = y1 + 20

                    # Put the white text directly without any background
                    cv2.putText(image, text,
                                (text_x, text_y),
                                font,
                                font_scale,
                                text_color,
                                font_thickness,
                                cv2.LINE_AA)

        return image

    def draw_summary(self, image, engaged_count, total_faces, engaged_score, peer_discussion_count, discussion_score, mode):
        """
        Draws the summary information in the bottom corner of the video frame.

        Parameters:
        - image (numpy.ndarray): The video frame on which to draw.
        - engaged_count (int): Number of engaged students.
        - total_faces (int): Total number of faces detected.
        - engaged_score (float): Percentage of engaged students.
        - peer_discussion_count (int): Number of students in peer discussions.
        - discussion_score (float): Percentage of students in peer discussions.
        - mode (str): Current classroom mode.
        """

        # Define the text lines
        text_lines = [
            f'Engagement Score: {engaged_count}/{total_faces} ({engaged_score:.2f}%) engaged',
            f'Discussion Score: {peer_discussion_count}/{total_faces} ({discussion_score:.2f}%) in discussion',
            f'Current Classroom Mode: {mode}'
        ]

        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        text_color = (255, 255, 255)  # White text
        bg_color = (0, 255, 0)        # Green background

        # Calculate the height of each text line
        text_heights = []
        text_widths = []
        for text in text_lines:
            (w, h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            text_widths.append(w)
            text_heights.append(h)

        # Calculate the size of the background rectangle
        padding = 10        # Padding around the text
        line_spacing = 5    # Space between lines
        total_text_height = sum(text_heights) + (len(text_lines) - 1) * line_spacing
        max_text_width = max(text_widths)

        # Calculate the width of the box (half the screen width)
        img_height, img_width = image.shape[:2]
        box_width = img_width // 2  # Half the screen width

        # Positioning the background rectangle in the bottom-left corner
        bg_x1 = 0
        bg_y1 = img_height - total_text_height - 2 * padding
        bg_x2 = box_width
        bg_y2 = img_height

        # Draw the background rectangle
        cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, cv2.FILLED)

        # Initialize the starting y-coordinate for the first line of text
        current_y = bg_y1 + padding + text_heights[0]

        # Draw each line of text
        for idx, text in enumerate(text_lines):
            text_x = bg_x1 + padding  # 10 pixels from the left edge
            cv2.putText(image, text, (text_x, current_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
            current_y += text_heights[idx] + line_spacing  # Move to the next line

        return image  # Optional: return the modified image
