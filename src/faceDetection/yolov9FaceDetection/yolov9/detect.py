import argparse
import platform
import sys
from pathlib import Path
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import pandas as pd  # For CSV export
import matplotlib.pyplot as plt  # For graph generation
import os

# Adjust import paths as needed
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Adjusted to current directory

# Add the main src directory to sys.path
SRC_DIR = ROOT / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# Import modules with updated paths
try:
    from headPoseDetection.deepHeadPoseLiteMaster import stable_hopenetlite
    from drawer.drawer import Drawer
    from faceTracker.sort import Sort
    from deepface import DeepFace
except ModuleNotFoundError as e:
    print("Error importing modules:", e)
    print("Ensure your paths and module names match the project structure.")
    sys.exit(1)

from faceDetection.yolov9FaceDetection.yolov9.models.common import DetectMultiBackend
from faceDetection.yolov9FaceDetection.yolov9.utils.dataloaders import LoadImages, LoadScreenshots, LoadStreams, \
    IMG_FORMATS, VID_FORMATS
from utils.general import (Profile, check_file, check_img_size, check_imshow, check_requirements,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

@smart_inference_mode()
def run(
        weights=ROOT / '../../../downloadedWeights/best.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        engagement_timeline_threshold=0.5,
        yaw_threshold=50,
        upwards_pitch_threshold=20,
        downwards_pitch_threshold=-30,
        emotion_threshold=90,
        scaling_factor=0.20,
):
    # Set half to False to use full precision
    half = False

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make directory

    # Load yolo model
    device = select_device(device)
    print(f"Using device: {device}")
    yolo_model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = yolo_model.stride, yolo_model.names, yolo_model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Load HopeNetLite model
    model_path = '../../../headPoseDetection/deepHeadPoseLiteMaster/model/shuff_epoch_120.pkl'
    pose_net = stable_hopenetlite.shufflenet_v2_x1_0()
    checkpoint = torch.load(model_path, map_location=device)
    pose_net.load_state_dict(checkpoint, strict=False)
    pose_net.eval().to(device)

    # Initialize the Real-ESRGAN model
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64,
        num_block=23, num_grow_ch=32, scale=4
    )

    # Create the upsampler
    upsampler = RealESRGANer(
        scale=4,
        model_path='../../../downloadedWeights/RealESRGAN_x4plus.pth',
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False  # Set half to False to use full precision
    )

    # Remove the half-precision model conversion
    # if half:
    #     yolo_model.model.half()
    #     pose_net.half()
    #     # Real-ESRGAN may not support half-precision; test and adjust as needed

    # Initialize the Drawer class for head pose visualization
    drawer = Drawer()

    # Define a function to pad the image to a square
    def pad_to_square(image):
        w, h = image.size
        max_side = max(w, h)
        padding = (
            (max_side - w) // 2,  # Left padding
            (max_side - h) // 2,  # Top padding
            (max_side - w) - (max_side - w) // 2,  # Right padding
            (max_side - h) - (max_side - h) // 2   # Bottom padding
        )
        return ImageOps.expand(image, padding, fill=0)

    # Define image preprocessing steps for HopeNetLite
    transform = transforms.Compose([
        transforms.Lambda(pad_to_square),  # Pad image to square
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Emotion categories
    emotion_mapping = {
        'angry': 'Negative',
        'disgust': 'Negative',
        'fear': 'Negative',
        'sad': 'Negative',
        'happy': 'Positive',
        'surprise': 'Positive',
        'neutral': 'Neutral'
    }

    # Initialize the SORT tracker
    tracker = Sort(max_age=1, min_hits=3, iou_threshold=0.3)
    track_poses = {}
    track_bboxes = {}
    N = 5  # Number of frames for smoothing

    # Initialize face engagement tracking dictionary
    face_engagement = {}  # key: face_id, value: {'state': 'engaged', 'duration': 0.0}

    # Dataloader
    bs = 2048  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = None, None  # Initialize video writer variables

    # Initialize fps
    fps = 30  # Default fps

    # Run inference
    yolo_model.warmup(imgsz=(1 if pt or yolo_model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    # Initialize a list to store engagement data per frame
    engagement_data = []

    # Function to save engagement results
    def save_engagement_results(engagement_data, save_dir):
        if engagement_data:
            # Convert engagement data to a DataFrame
            engagement_df = pd.DataFrame(engagement_data)

            # Save DataFrame to CSV
            csv_save_path = save_dir / 'engagement_data.csv'
            engagement_df.to_csv(csv_save_path, index=False)
            print(f"Engagement data saved to {csv_save_path}")

            # Plot engagement scores over time
            plt.figure(figsize=(12, 6))
            plt.plot(engagement_df['time'], engagement_df['engaged_score'], label='Engaged Score (%)', color='blue')

            # Adding labels and title
            plt.xlabel('Time (seconds)')
            plt.ylabel('Score (%)')
            plt.title('Engagement Scores Over Time')
            plt.legend()
            plt.grid(True)

            # Save the plot
            graph_save_path = save_dir / 'engagement_graph.png'
            plt.savefig(graph_save_path)
            plt.close()
            print(f"Engagement graph saved to {graph_save_path}")

            # Identify the maximum number of faces detected in any frame
            max_faces = engagement_df['total_faces'].max()
            print(f"Maximum number of faces detected in any frame: {max_faces}")

            # Filter the DataFrame to include only frames with the maximum number of faces
            filtered_engagement = engagement_df[engagement_df['total_faces'] == max_faces]

            if not filtered_engagement.empty:
                # Sort the filtered DataFrame by 'total_engagement_score' in ascending order
                sorted_engagement = filtered_engagement.sort_values(by='total_engagement_score')

                # Initialize list to store selected lowest engagement points
                selected_engagement = []
                selected_times = []

                for idx, row in sorted_engagement.iterrows():
                    time_stamp = row['time']

                    # If no times have been selected yet, add the first one
                    if not selected_times:
                        selected_engagement.append(row)
                        selected_times.append(time_stamp)
                    else:
                        # Check if this time is at least 10 seconds apart from all previously selected times
                        if all(abs(time_stamp - t) >= 10 for t in selected_times):
                            selected_engagement.append(row)
                            selected_times.append(time_stamp)

                    # If we have already selected 3 points, break
                    if len(selected_engagement) >= 3:
                        break

                if selected_engagement:
                    # Display the selected lowest engagement points
                    print(
                        "\nThree Lowest Points of Engagement (with maximum number of faces, at least 10 seconds apart):")
                    for row in selected_engagement:
                        time_stamp = row['time']
                        score = row['total_engagement_score']
                        engaged = row['engaged_faces']
                        total_faces = row['total_faces']
                        mode = row['mode']
                        print(
                            f"Time: {time_stamp:.2f}s, Total Engagement Score: {score:.2f}%, Mode: {mode}, Engaged: {engaged}, Total Faces: {total_faces}")
                else:
                    print("No frames with the required separation were found.")
            else:
                print("No frames with the maximum number of faces were found.")

        else:
            print("No engagement data to save.")

    # Initialize a dictionary to keep track of saved faces
    saved_faces = {}  # key: face_id, value: True

    try:
        for path, im, im0s, vid_cap, s in dataset:
            if vid_cap and hasattr(vid_cap, 'get'):
                fps = vid_cap.get(cv2.CAP_PROP_FPS) or fps

            # Compute frame duration
            frame_duration = 1 / fps

            with dt[0]:
                im = torch.from_numpy(im).to(device)
                im = im.float()  # Always convert to full precision (float32)
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                visualize_path = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = yolo_model(im, augment=augment, visualize=visualize_path)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            # Initialize sets to track unique engaged and discussion participants
            engaged_faces_frame = set()
            discussion_faces_frame = set()
            total_faces_frame = 0

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                # Compute time in seconds
                time_in_sec = frame / fps

                # Create a copy of im0 for face crop extraction
                im0_copy = im0.copy()

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # e.g., 'runs/detect/exp/image.jpg'
                txt_path = str(save_dir / 'labels' / p.stem) + (
                    f'_{frame}' if dataset.mode != 'image' else '')  # e.g., 'runs/detect/exp/labels/image.txt'
                s += '%gx%g ' % im.shape[2:]  # print string
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))

                if len(det):
                    # Rescale boxes from img_size to original image size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results (number of detections per class)
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Prepare detections for SORT (x1, y1, x2, y2, score)
                    detections = []
                    for *xyxy, conf, cls in det:
                        x1, y1, x2, y2 = map(int, xyxy)
                        score = conf.item()
                        detections.append([x1, y1, x2, y2, score])

                        # Draw bounding box and label on the image
                        label = f"{names[int(cls)]} {conf:.2f}"
                        annotator.box_label(xyxy, label, color=colors(int(cls), True))

                    detections = np.array(detections)

                    # Update tracker
                    tracks = tracker.update(detections)

                    # Process each track
                    for track in tracks:
                        x1, y1, x2, y2, track_id = track.astype(int)

                        # Assign face_id
                        face_id = track_id  # Using track_id as face_id for consistency

                        # Calculate center coordinates and square bounding box
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        width = x2 - x1
                        height = y2 - y1
                        side_length = max(width, height)
                        side_length = int(side_length * (1 + scaling_factor))
                        half_side = side_length // 2

                        x1_face = max(0, center_x - half_side)
                        y1_face = max(0, center_y - half_side)
                        x2_face = min(im0.shape[1], center_x + half_side)
                        y2_face = min(im0.shape[0], center_y + half_side)

                        # Extract the face crop from im0_copy (original image without annotations)
                        face_crop = im0_copy[y1_face:y2_face, x1_face:x2_face]

                        # Save the original face image only once per face ID
                        if face_id not in saved_faces:
                            # Save the cropped face image with the face ID
                            face_save_path = save_dir / 'faces_original' / f'face_{face_id}.jpg'
                            face_save_path.parent.mkdir(parents=True, exist_ok=True)
                            cv2.imwrite(str(face_save_path), face_crop)
                            print(f"Original face saved to {face_save_path}")
                            saved_faces[face_id] = True  # Mark as saved

                        # Upscale the face crop using Real-ESRGAN
                        try:
                            # Real-ESRGAN expects images in BGR format
                            restored_face, _ = upsampler.enhance(face_crop, outscale=4)
                            restored_face_bgr = restored_face  # Output is in BGR format
                        except Exception as e:
                            print(f"Error during Real-ESRGAN upscaling: {e}")
                            print(f"Using original face crop for face ID {face_id}.")
                            restored_face_bgr = face_crop  # Use the original face crop

                        # Save the restored face image
                        face_save_path_restored = save_dir / 'faces_restored' / f'face_{face_id}_restored.jpg'
                        face_save_path_restored.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(face_save_path_restored), restored_face_bgr)
                        print(f"Restored face saved to {face_save_path_restored}")

                        # Convert restored face to PIL Image for further processing
                        face_crop_pil = Image.fromarray(cv2.cvtColor(restored_face_bgr, cv2.COLOR_BGR2RGB))

                        # Apply transformations
                        face_tensor = transform(face_crop_pil).unsqueeze(0).to(device, non_blocking=True)

                        # Smooth bounding boxes
                        if track_id not in track_bboxes:
                            track_bboxes[track_id] = []
                        track_bboxes[track_id].append([x1, y1, x2, y2])
                        track_bboxes[track_id] = track_bboxes[track_id][-N:]  # Keep last N

                        # Apply smoothing to bounding boxes
                        avg_x1 = int(np.mean([b[0] for b in track_bboxes[track_id]]))
                        avg_y1 = int(np.mean([b[1] for b in track_bboxes[track_id]]))
                        avg_x2 = int(np.mean([b[2] for b in track_bboxes[track_id]]))
                        avg_y2 = int(np.mean([b[3] for b in track_bboxes[track_id]]))
                        x1, y1, x2, y2 = avg_x1, avg_y1, avg_x2, avg_y2

                        # **Store original bounding box coordinates for annotations**
                        orig_x1, orig_y1, orig_x2, orig_y2 = x1, y1, x2, y2

                        # Run the model to get yaw, pitch, and roll
                        with torch.no_grad():
                            yaw_logits, pitch_logits, roll_logits = pose_net(face_tensor)

                            # Define idx_tensor and adjusted angles
                            idx_tensor = torch.arange(66, dtype=torch.float32).to(device)
                            angles = idx_tensor * 3 - 99 + 1.5  # Shift to bin midpoints

                            # Softmax to get probabilities
                            yaw_probs = torch.softmax(yaw_logits, dim=1)
                            pitch_probs = torch.softmax(pitch_logits, dim=1)
                            roll_probs = torch.softmax(roll_logits, dim=1)

                            # Predicted angles
                            yaw_predicted = torch.sum(yaw_probs * angles, dim=1)
                            pitch_predicted = torch.sum(pitch_probs * angles, dim=1)
                            roll_predicted = torch.sum(roll_probs * angles, dim=1)

                            # Convert to float
                            yaw_predicted = yaw_predicted.item()
                            pitch_predicted = pitch_predicted.item()
                            roll_predicted = roll_predicted.item()

                        # Print head pose values for debugging
                        print(f"Face {face_id} - Yaw: {yaw_predicted:.2f}, Pitch: {pitch_predicted:.2f}, Roll: {roll_predicted:.2f}")

                        # Smooth head pose estimations
                        if track_id not in track_poses:
                            track_poses[track_id] = []
                        track_poses[track_id].append((yaw_predicted, pitch_predicted, roll_predicted))
                        track_poses[track_id] = track_poses[track_id][-N:]  # Keep last N

                        # Apply smoothing
                        avg_yaw = np.mean([p[0] for p in track_poses[track_id]])
                        avg_pitch = np.mean([p[1] for p in track_poses[track_id]])
                        avg_roll = np.mean([p[2] for p in track_poses[track_id]])

                        # Use the smoothed head pose
                        pose_smoothed = (avg_yaw, avg_pitch, avg_roll)

                        # Prepare landmarks (nose tip) for drawing
                        landmarks = ((orig_x1 + orig_x2) // 2, (orig_y1 + orig_y2) // 2)

                        # Prepare bbox_dict for drawing functions
                        bbox = (orig_x1, orig_y1, orig_x2, orig_y2)

                        # Draw the axis
                        im0 = drawer.draw_axis(im0, pose_smoothed, landmarks, bbox, axis_size=50)

                        # Determine if the person is looking forward
                        is_looking_forward = (
                                abs(avg_yaw) <= yaw_threshold and
                                downwards_pitch_threshold <= avg_pitch <= upwards_pitch_threshold
                        )

                        # Initialize engagement status
                        if is_looking_forward:
                            # Proceed with emotion detection
                            try:
                                # Run DeepFace emotion analysis
                                emotion_analysis = DeepFace.analyze(
                                    np.array(restored_face_bgr),
                                    actions=['emotion'],
                                    enforce_detection=False,
                                    detector_backend='mtcnn'
                                )

                                if isinstance(emotion_analysis, list):
                                    if len(emotion_analysis) > 0:
                                        emotion_analysis = emotion_analysis[0]
                                    else:
                                        raise ValueError("DeepFace.analyze returned an empty list.")

                                # Extract dominant emotion and emotion scores
                                emotion_data = emotion_analysis.get('emotion', {})
                                dominant_emotion = emotion_analysis.get('dominant_emotion', 'neutral').lower()
                                dominant_emotion_score = emotion_data.get(dominant_emotion, 0.0)

                                # Determine if emotion_category should be 'Negative' based on score
                                if dominant_emotion in ['angry', 'disgust', 'fear', 'sad'] and dominant_emotion_score > emotion_threshold:
                                    emotion_category = 'Negative'
                                else:
                                    emotion_category = emotion_mapping.get(dominant_emotion, 'Neutral')

                                # Log the detected emotion details for each face
                                print(f"Face {face_id} - Dominant Emotion: {dominant_emotion.capitalize()} ({dominant_emotion_score:.2f}%)")
                                print(f"Face {face_id} - Emotion Category: {emotion_category}")

                            except Exception as e:
                                print(f"Warning: Face {face_id} - Emotion Analysis Error: {e}")
                                dominant_emotion = 'neutral'
                                dominant_emotion_score = 0.0
                                emotion_category = 'Neutral'

                            # Determine engagement based on emotion category
                            if emotion_category == 'Negative':
                                is_engaged = False  # Not Engaged
                                print(f"Face {face_id} classified as 'Not Engaged' due to negative emotion.")
                            else:
                                is_engaged = True
                                print(f"Face {face_id} classified as 'Engaged'.")
                        else:
                            # Person is not looking forward
                            is_engaged = False
                            print(f"Face {face_id} classified as 'Not Engaged' due to not looking forward (head pose).")

                        # *** Start of engagement smoothing logic ***

                        # Update face_engagement for the face_id
                        if face_id not in face_engagement:
                            # Initialize engagement state, duration, and overall_status
                            face_engagement[face_id] = {'state': 'engaged', 'duration': 0.0, 'overall_status': True}

                        current_state = face_engagement[face_id]['state']
                        current_duration = face_engagement[face_id]['duration']
                        previous_overall_status = face_engagement[face_id]['overall_status']

                        if is_engaged:
                            if current_state == 'engaged':
                                # Continue in engaged state, increment duration
                                face_engagement[face_id]['duration'] += frame_duration
                            else:
                                # State change from not engaged to engaged, reset duration
                                face_engagement[face_id]['state'] = 'engaged'
                                face_engagement[face_id]['duration'] = frame_duration
                        else:
                            if current_state == 'not_engaged':
                                # Continue in not engaged state, increment duration
                                face_engagement[face_id]['duration'] += frame_duration
                            else:
                                # State change from engaged to not engaged, reset duration
                                face_engagement[face_id]['state'] = 'not_engaged'
                                face_engagement[face_id]['duration'] = frame_duration

                        # Determine overall engagement status based on duration
                        if face_engagement[face_id]['state'] == 'not_engaged':
                            if face_engagement[face_id]['duration'] >= engagement_timeline_threshold:
                                # Consider the person as not engaged
                                overall_is_engaged = False
                            else:
                                # Maintain previous overall engagement status
                                overall_is_engaged = previous_overall_status
                        else:  # state == 'engaged'
                            if face_engagement[face_id]['duration'] >= engagement_timeline_threshold:
                                # Consider the person as engaged
                                overall_is_engaged = True
                            else:
                                # Maintain previous overall engagement status
                                overall_is_engaged = previous_overall_status

                        # Update the overall_status in face_engagement
                        face_engagement[face_id]['overall_status'] = overall_is_engaged

                        # Update engaged or discussion faces based on overall engagement status
                        if overall_is_engaged:
                            engaged_faces_frame.add(face_id)
                            print(
                                f"Face {face_id} is considered 'Engaged' (Duration in current state: {face_engagement[face_id]['duration']:.2f}s)")
                        else:
                            discussion_faces_frame.add(face_id)
                            print(
                                f"Face {face_id} is considered 'Not Engaged' after {face_engagement[face_id]['duration']:.2f}s in non-engagement.")

                        # Update the engagement status label for drawing
                        engagement_status = 'Engaged' if overall_is_engaged else 'Not Engaged'

                        # Draw engagement status and face ID
                        im0 = drawer.draw_engagement_status(im0, bbox, face_id, engagement_status)

                        # *** End of engagement smoothing logic ***

                        # Draw pose information (yaw, pitch, roll)
                        im0 = drawer.draw_pose_info(im0, pose_smoothed, landmarks, bbox)

                        # Draw emotion label (Positive, Neutral, Negative)
                        if is_looking_forward:
                            im0 = drawer.draw_emotion_label(im0, bbox, emotion_category)
                        else:
                            # If not looking forward, you can label emotion as 'Unknown' or skip
                            pass

                        if save_crop:
                            save_one_box([x1, y1, x2, y2], im0_copy, file=save_dir / 'crops' / f'Face_{face_id}.jpg', BGR=True)

                        total_faces_frame += 1  # Count each face once

                if total_faces_frame > 0:
                    engaged_score = (len(engaged_faces_frame) / total_faces_frame) * 100
                    discussion_score = (len(discussion_faces_frame) / total_faces_frame) * 100
                    total_engagement_score = engaged_score
                else:
                    engaged_score = 0
                    discussion_score = 0
                    total_engagement_score = 0

                # Determine classroom mode based on engaged_score and discussion_score
                if engaged_score > discussion_score:
                    mode = "Lecture Mode"
                elif engaged_score < discussion_score:
                    mode = "Peer Discussion Mode"
                else:
                    mode = "Mixed Engagement"

                # Log engagement and mode
                print(f"Engagement Score: {len(engaged_faces_frame)}/{total_faces_frame} ({engaged_score:.2f}%) engaged")
                print(f"Discussion Score: {len(discussion_faces_frame)}/{total_faces_frame} ({discussion_score:.2f}%) in discussion")
                print(f"Total Engagement Score (Engaged): {total_engagement_score:.2f}%")
                print(f"Current Classroom Mode: {mode}")

                # Store engagement data for this frame
                engagement_data.append({
                    'time': time_in_sec,
                    'engaged_faces': len(engaged_faces_frame),
                    'discussion_faces': len(discussion_faces_frame),
                    'total_faces': total_faces_frame,
                    'engaged_score': engaged_score,
                    'discussion_score': discussion_score,
                    'total_engagement_score': total_engagement_score,
                    'mode': mode
                })

                # Overlay the summary information on the frame
                im0 = drawer.draw_summary(im0, len(engaged_faces_frame), len(discussion_faces_frame), total_faces_frame,
                                          engaged_score, discussion_score, total_engagement_score, mode)

            else:
                print('No detections')

            # Stream results
            im0 = annotator.result()

            if view_img:
                if platform.system() == 'Linux':
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # Press 'q' to quit
                    raise StopIteration

            # Save results (video with detections)
            if save_img:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS) or fps
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream or images
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # enforce .mp4 extension
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

            # Print time (inference-only)
            print(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    except KeyboardInterrupt:
        print("Warning: Process interrupted by user. Saving engagement data...")
        save_engagement_results(engagement_data, save_dir)
        # Release video writer if it's open
        if save_img and isinstance(vid_writer, cv2.VideoWriter):
            vid_writer.release()
        sys.exit(0)  # Exit the program gracefully

    # After all frames are processed
    save_engagement_results(engagement_data, save_dir)

    # Release video writer if it's open
    if save_img and isinstance(vid_writer, cv2.VideoWriter):
        vid_writer.release()

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\nResults saved to {save_dir}"
        print(s)
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / '../../../downloadedWeights/best.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images',
                        help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                        help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=500, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    # Remove the '--half' argument
    # parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--engagement-timeline-threshold', type=float, default=0.5,
                        help='Engagement duration threshold in seconds')
    parser.add_argument('--yaw-threshold', type=float, default=50,
                        help='Yaw threshold in degrees')
    parser.add_argument('--upwards-pitch-threshold', type=float, default=20,
                        help='Upwards pitch threshold in degrees')
    parser.add_argument('--downwards-pitch-threshold', type=float, default=-30,
                        help='Downwards pitch threshold in degrees')
    parser.add_argument('--emotion-threshold', type=float, default=90,
                        help='Emotion threshold in percentage')
    parser.add_argument('--scaling-factor', type=float, default=0.20,
                        help='Scaling factor for bounding boxes')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
