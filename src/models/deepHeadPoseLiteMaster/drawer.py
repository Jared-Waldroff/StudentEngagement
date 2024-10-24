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

    def draw_axis(self, image, pose, landmarks, bbox, emotion_category, face_id, axis_size=10):
        """
        Draws pose axes and orientation text on the image.

        Parameters:
        - image (numpy.ndarray): The video frame on which to draw.
        - pose (list): List of tuples containing yaw, pitch, and roll angles.
        - landmarks (list): List of landmark coordinates.
        - bbox (dict): Dictionary containing bounding box information.
        - emotion_category (str): Categorized emotion ('Positive', 'Negative', 'Neutral').
        - face_id (int): Unique identifier for the face.
        - axis_size (int): Length of the axes to be drawn.
        """
        image = self.__getinstance(image, str)
        for i, lms in enumerate(landmarks):
            if lms is not None:
                x, y = lms[0]
                yaw, pitch, roll = pose[i]

                # Convert angles from degrees to radians
                yaw_rad = np.deg2rad(yaw)
                pitch_rad = np.deg2rad(pitch)
                roll_rad = np.deg2rad(roll)

                # Convert yaw to negative since the axes are flipped
                yaw_rad = -yaw_rad

                # Build the rotation vector (pitch, yaw, roll)
                rotation_vector = np.array([pitch_rad, yaw_rad, roll_rad])

                # Compute rotation matrix
                rotation_matrix = cv2.Rodrigues(rotation_vector)[0].astype(np.float64)

                # Create axes points in 3D
                axes_points = np.array([
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0]
                ], dtype=np.float64)

                # Rotate axes points
                axes_points = rotation_matrix @ axes_points

                # Scale axes
                axes_points = (axes_points[:2, :] * axis_size).astype(int)

                # Displace the axes points by the nose point coordinates
                axes_points[0, :] += int(x)
                axes_points[1, :] += int(y)

                # Extract points for drawing
                origin = tuple(axes_points[:, 3].ravel())
                x_axis = tuple(axes_points[:, 0].ravel())
                y_axis = tuple(axes_points[:, 1].ravel())
                z_axis = tuple(axes_points[:, 2].ravel())

                # Draw the axes
                cv2.line(image, origin, x_axis, (255, 0, 0), 2)  # X-axis (red)
                cv2.line(image, origin, y_axis, (0, 255, 0), 2)  # Y-axis (green)
                cv2.line(image, origin, z_axis, (0, 0, 255), 2)  # Z-axis (blue)

                # Determine engagement status based on pose and emotion category
                engaged_text = "Engaged"
                if (pitch > 30 or pitch < -50) or (yaw > 30 or yaw < -30) or emotion_category == "Negative":
                    engaged_text = 'Not Engaged'

                # Draw 'Emotion' and 'Engaged' labels with adjusted positions
                if "face" in bbox:
                    x1, y1, x2, y2 = bbox["face"]

                    # Adjust positions as needed
                    # Emotion label (move up by 10 pixels)
                    cv2.rectangle(image, (int(x1), int(y1) - 60), (int(x1) + 250, int(y1) - 20), (0, 200, 200), -1)
                    cv2.putText(image, f"Emotion: {emotion_category}", (int(x1) + 10, int(y1) - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

                    # Engaged label (move up by 10 pixels)
                    cv2.rectangle(image, (int(x1), int(y1) - 100), (int(x1) + 250, int(y1) - 60), (255, 200, 20), -1)
                    cv2.putText(image, engaged_text, (int(x1) + 10, int(y1) - 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

                    # Display yaw, pitch, and roll angles below the bounding box (move down by 10 pixels)
                    cv2.putText(image, f"yaw: {yaw:.2f}", (int(x1), int(y2) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                (0, 255, 0), 2)
                    cv2.putText(image, f"pitch: {pitch:.2f}", (int(x1), int(y2) + 35), cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (255, 0, 0), 2)
                    cv2.putText(image, f"roll: {roll:.2f}", (int(x1), int(y2) + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                (0, 0, 255), 2)

                    # Draw Face ID directly above the face
                    text = f'ID: {face_id}'
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    font_thickness = 2
                    text_color = (255, 255, 255)  # White color for text

                    # Position the text above the bounding box
                    text_x = int(x1) - 30
                    text_y = int(y1) + 20

                    # Put the white text directly without any background
                    cv2.putText(image, text,
                                (text_x, text_y),
                                font,
                                font_scale,
                                text_color,
                                font_thickness,
                                cv2.LINE_AA)
        return image


    def draw_summary(self, image, engaged_count, discussion_count, total_faces, engaged_score, discussion_score,
                     total_engagement_score, mode):
        """
        Draws the summary information in the bottom corner of the video frame.

        Parameters:
        - image (numpy.ndarray): The video frame on which to draw.
        - engaged_count (int): Number of engaged students.
        - discussion_count (int): Number of students in peer discussions.
        - total_faces (int): Total number of faces detected.
        - engaged_score (float): Percentage of engaged students.
        - discussion_score (float): Percentage of students in peer discussions.
        - total_engagement_score (float): Combined engagement score.
        - mode (str): Current classroom mode.
        """

        # Define the text lines
        text_lines = [
            f'Engagement Score: {engaged_count}/{total_faces} ({engaged_score:.2f}%) engaged',
            f'Discussion Score: {discussion_count}/{total_faces} ({discussion_score:.2f}%) in discussion',
            f'Total Engagement Score: {total_engagement_score:.2f}%',
            f'Classroom Mode: {mode}'
        ]

        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        text_color = (255, 255, 255)  # White text
        bg_color = (0, 255, 0)  # Green background

        # Calculate the height of each text line
        text_heights = []
        text_widths = []
        for text in text_lines:
            (w, h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            text_widths.append(w)
            text_heights.append(h)

        # Calculate the size of the background rectangle
        padding = 10  # Padding around the text
        line_spacing = 5  # Space between lines
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
