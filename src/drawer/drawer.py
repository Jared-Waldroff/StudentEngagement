import cv2
import numpy as np
import math


class Drawer:
    def __init__(self):
        pass  # Initialize any required attributes if necessary

    def __getinstance(self, image, intype):
        if isinstance(image, intype):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        return image

    def draw_bbox(self, image, bbox, face_id=None, color=(255, 200, 20), thickness=2, draw_person=False,
                  draw_face=True):
        """
        Draws bounding boxes around detected objects (person and face).

        Parameters:
            image (ndarray): The image on which to draw.
            bbox (tuple): Bounding box coordinates (x1, y1, x2, y2).
            face_id (int, optional): Identifier for the face.
            color (tuple, optional): Color for the bounding box.
            thickness (int, optional): Thickness of the bounding box lines.
            draw_person (bool, optional): Flag to draw person bounding box.
            draw_face (bool, optional): Flag to draw face bounding box.

        Returns:
            image (ndarray): Image with bounding boxes drawn.
        """
        image = self.__getinstance(image, str)
        x1, y1, x2, y2 = bbox

        # Draw person bounding box if required
        if draw_person:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
            if face_id is not None:
                cv2.putText(image, f"Face {face_id}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

        # Draw face bounding box if required
        if draw_face:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
            if face_id is not None:
                cv2.putText(image, f"Face {face_id}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

        return image

    def draw_landmarks(self, image, landmarks, color=(255, 0, 0), radius=3, thickness=-1):
        """
        Draws landmarks (e.g., nose tip) on the image.

        Parameters:
            image (ndarray): The image on which to draw.
            landmarks (tuple): Landmark coordinates (x, y).
            color (tuple, optional): Color for the landmarks.
            radius (int, optional): Radius of the landmark circles.
            thickness (int, optional): Thickness of the landmark circles (-1 for filled).

        Returns:
            image (ndarray): Image with landmarks drawn.
        """
        image = self.__getinstance(image, str)
        x, y = landmarks
        cv2.circle(image, (int(x), int(y)), radius, color, thickness)
        return image

    def draw_axis(self, image, pose, landmarks, bbox, axis_size=50):
        """
        Draws 3D axes on the image based on head pose estimation.

        Parameters:
            image (ndarray): The image on which to draw.
            pose (tuple): Head pose angles (yaw, pitch, roll) in degrees.
            landmarks (tuple): Landmark coordinates (x, y).
            bbox (tuple): Bounding box coordinates (x1, y1, x2, y2).
            axis_size (int, optional): Size of the axes to draw.

        Returns:
            image (ndarray): Image with axes drawn.
        """
        image = self.__getinstance(image, str)
        yaw, pitch, roll = pose
        x, y = landmarks

        # Convert angles from degrees to radians for rotation matrix
        yaw_rad = math.radians(yaw)
        pitch_rad = math.radians(pitch)
        roll_rad = math.radians(roll)

        # Rotation matrices around the X, Y, and Z axis
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(pitch_rad), -math.sin(pitch_rad)],
            [0, math.sin(pitch_rad), math.cos(pitch_rad)]
        ])

        Ry = np.array([
            [math.cos(yaw_rad), 0, math.sin(yaw_rad)],
            [0, 1, 0],
            [-math.sin(yaw_rad), 0, math.cos(yaw_rad)]
        ])

        Rz = np.array([
            [math.cos(roll_rad), -math.sin(roll_rad), 0],
            [math.sin(roll_rad), math.cos(roll_rad), 0],
            [0, 0, 1]
        ])

        # Combined rotation matrix
        R = Rz @ Ry @ Rx

        # Define the axes
        axes = np.float32([
            [axis_size, 0, 0],
            [0, axis_size, 0],
            [0, 0, axis_size]
        ]).reshape(-1, 3)

        # Project the 3D axes to 2D image plane
        nose_end_point3D = R @ axes.T
        nose_end_point2D = nose_end_point3D[:2, :] + np.array([[x], [y]])

        # Draw the axes
        cv2.line(image, (int(x), int(y)), (int(nose_end_point2D[0][0]), int(nose_end_point2D[1][0])), (0, 0, 255),
                 3)  # X-axis in red
        cv2.line(image, (int(x), int(y)), (int(nose_end_point2D[0][1]), int(nose_end_point2D[1][1])), (0, 255, 0),
                 3)  # Y-axis in green
        cv2.line(image, (int(x), int(y)), (int(nose_end_point2D[0][2]), int(nose_end_point2D[1][2])), (255, 0, 0),
                 3)  # Z-axis in blue

        return image

    def draw_pose_info(self, image, pose, landmarks, bbox):
        """
        Draws yaw, pitch, and roll information on the image with a fully fitting transparent background box.
        """
        image = self.__getinstance(image, str)
        yaw, pitch, roll = pose
        x1, y1, x2, y2 = bbox

        # Text for yaw, pitch, and roll
        texts = [
            f"Yaw: {yaw:.2f}",
            f"Pitch: {pitch:.2f}",
            f"Roll: {roll:.2f}"
        ]

        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        padding = 10  # Padding around the text

        # Calculate the max text width and total height for all lines
        text_sizes = [cv2.getTextSize(text, font, font_scale, font_thickness)[0] for text in texts]
        max_text_width = max(size[0] for size in text_sizes)
        total_text_height = sum(size[1] for size in text_sizes) + (len(texts) - 1) * 5

        # Define the background rectangle coordinates with adjusted width
        rect_x1 = x1
        rect_y1 = y2 + 25  # Position below the bounding box
        rect_x2 = rect_x1 + max_text_width + 2 * padding + 20  # Extra width adjustment
        rect_y2 = rect_y1 + total_text_height + 2 * padding

        # Ensure the rectangle fits within the image boundaries
        img_height, img_width = image.shape[:2]
        if rect_y2 > img_height:
            rect_y1 = y1 - total_text_height - 2 * padding - 10  # Position above if below is out of frame
            rect_y2 = rect_y1 + total_text_height + 2 * padding

        # Draw semi-transparent background rectangle
        overlay = image.copy()
        alpha = 0.4
        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        # Draw each line of text with padding
        for idx, text in enumerate(texts):
            text_y = rect_y1 + padding + (idx + 1) * text_sizes[0][1] + idx * 5
            cv2.putText(image, text, (rect_x1 + padding, text_y), font, font_scale, (255, 255, 255), font_thickness,
                        cv2.LINE_AA)

        return image

    def draw_engagement_status(self, image, bbox, face_id, engagement_status):
        """
        Draws engagement status and face ID on the image.

        Parameters:
            image (ndarray): The image on which to draw.
            bbox (tuple): Bounding box coordinates (x1, y1, x2, y2).
            face_id (int): Identifier for the face.
            engagement_status (str): Engagement status ('Engaged' or 'Not Engaged').

        Returns:
            image (ndarray): Image with engagement status drawn.
        """
        image = self.__getinstance(image, str)
        x1, y1, x2, y2 = bbox
        text = f"ID:{face_id} {engagement_status}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.75
        font_thickness = 2
        text_color = (255, 255, 255)
        bg_color = (255, 150, 0)
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        cv2.rectangle(image, (int(x1), int(y1) - 54),
                      (int(x1) + text_width + 10, int(y1) - 54 + text_height + 10), bg_color, -1)
        cv2.putText(image, text, (int(x1) + 5, int(y1) - 54 + text_height + 5),
                    font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        return image

    def draw_emotion_label(self, image, bbox, emotion_label):
        """
        Draws the emotion label on the image.

        Parameters:
            image (ndarray): The image on which to draw.
            bbox (tuple): Bounding box coordinates (x1, y1, x2, y2).
            emotion_label (str): Emotion category ('Positive', 'Neutral', 'Negative').

        Returns:
            image (ndarray): Image with emotion label drawn.
        """
        image = self.__getinstance(image, str)
        x1, y1, x2, y2 = bbox
        text = f"Emotion: {emotion_label}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        text_color = (255, 255, 255)
        # Define positions
        text_x = int(x1)
        text_y = int(y2) + 25
        # Adjust if text goes beyond image
        img_height, img_width = image.shape[:2]
        if text_y + 25 > img_height:
            text_y = int(y1) - 10  # Position above the bounding box
        # Draw background rectangle for better visibility
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        rect_x1 = text_x
        rect_y1 = text_y - text_height - 5
        rect_x2 = text_x + text_width + 10
        rect_y2 = text_y + 5
        cv2.rectangle(image, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
        # Draw the text
        cv2.putText(image, text, (text_x + 5, text_y),
                    font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        return image

    def draw_summary(self, image, engaged_count, discussion_count, total_faces, engaged_score, discussion_score,
                     total_engagement_score, mode):
        """
        Draws a summary of engagement statistics on the image.

        Parameters:
            image (ndarray): The image on which to draw.
            engaged_count (int): Number of engaged faces.
            discussion_count (int): Number of faces in discussion.
            total_faces (int): Total number of faces detected.
            engaged_score (float): Percentage of engaged faces.
            discussion_score (float): Percentage of faces in discussion.
            total_engagement_score (float): Overall engagement score.
            mode (str): Current classroom mode.

        Returns:
            image (ndarray): Image with summary drawn.
        """
        text_lines = [
            f'Engagement Score: {engaged_count}/{total_faces} ({engaged_score:.2f}%) engaged',
            f'Discussion Score: {discussion_count}/{total_faces} ({discussion_score:.2f}%) in discussion',
            f'Total Engagement Score: {total_engagement_score:.2f}%',
            f'Classroom Mode: {mode}'
        ]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        text_color = (255, 255, 255)
        bg_color = (0, 255, 0)
        text_heights = []
        text_widths = []
        for text in text_lines:
            (w, h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            text_widths.append(w)
            text_heights.append(h)
        padding = 10
        line_spacing = 5
        total_text_height = sum(text_heights) + (len(text_lines) - 1) * line_spacing
        max_text_width = max(text_widths)
        img_height, img_width = image.shape[:2]
        box_width = img_width // 2
        bg_x1 = 0
        bg_y1 = img_height - total_text_height - 2 * padding
        bg_x2 = box_width
        bg_y2 = img_height
        cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, cv2.FILLED)
        current_y = bg_y1 + padding + text_heights[0]
        for idx, text in enumerate(text_lines):
            text_x = bg_x1 + padding
            cv2.putText(image, text, (text_x, current_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
            current_y += text_heights[idx] + line_spacing
        return image
