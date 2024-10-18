import argparse
import os
import platform
import sys
from pathlib import Path
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image

# Set up file paths
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory

# Adding the models directory to the path
MODELS_DIR = ROOT.parents[2] / 'models'  # Adjust the path as necessary
if str(MODELS_DIR) not in sys.path:
    sys.path.append(str(MODELS_DIR))  # add MODELS_DIR to PATH

# Add the trackers directory to the path (since it's at the same level as models)
TRACKERS_DIR = ROOT.parents[2] / 'trackers'  # Adjust the path correctly
if str(TRACKERS_DIR) not in sys.path:
    sys.path.append(str(TRACKERS_DIR))  # add TRACKERS_DIR to PATH

# Import the required components for head pose estimation
from deepHeadPoseLiteMaster import stable_hopenetlite
from deepHeadPoseLiteMaster.drawer import Drawer
from deepface import DeepFace

# Import the SORT tracker
from sort import Sort

# Other necessary imports...
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

@smart_inference_mode()
def run(
        weights=ROOT / 'yolo.pt',  # model path or triton URL
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
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
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

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Load HopeNetLite model
    model_path = '../../../models/deepHeadPoseLiteMaster/model/shuff_epoch_120.pkl'
    pose_net = stable_hopenetlite.shufflenet_v2_x1_0()
    checkpoint = torch.load(model_path, map_location='cpu')
    pose_net.load_state_dict(checkpoint, strict=False)
    pose_net.eval()

    # Initialize the Drawer class for head pose visualization
    drawer = Drawer()

    # Define image preprocessing steps for HopeNetLite
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    def map_emotion_to_category(emotion):
        positive_emotions = ['happy', 'surprise']
        negative_emotions = ['angry', 'disgust', 'fear', 'sad']
        neutral_emotions = ['neutral']

        if emotion in positive_emotions:
            return 'Positive'
        elif emotion in negative_emotions:
            return 'Negative'
        elif emotion in neutral_emotions:
            return 'Neutral'
        else:
            return 'Neutral'  # Default to Neutral for any undefined emotions

    # Initialize the SORT tracker
    tracker = Sort(max_age=1, min_hits=3, iou_threshold=0.3)
    track_poses = {}
    track_bboxes = {}
    N = 5  # Number of frames for smoothing

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = None, None  # Initialize video writer variables

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    engaged_faces = set()  # Initialize set for engaged students
    discussion_faces = set()  # Initialize set for students in discussion
    negative_faces = set()  # Initialize set for negative emotions
    total_faces = 0  # Total faces detected

    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize_path = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize_path)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

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

                # Initialize sets to track unique engaged and discussion participants
                engaged_faces_frame = set()
                discussion_faces_frame = set()
                total_faces_frame = 0

                # Process each track
                for track in tracks:
                    x1, y1, x2, y2, track_id = track.astype(int)

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

                    # Increase the size of the bounding box by 20% for better head pose estimation
                    width_increase = int(0.2 * (x2 - x1))
                    height_increase = int(0.2 * (y2 - y1))
                    x1 = max(0, x1 - width_increase)
                    y1 = max(0, y1 - height_increase)
                    x2 = min(im0.shape[1], x2 + width_increase)
                    y2 = min(im0.shape[0], y2 + height_increase)

                    face_crop = im0[y1:y2, x1:x2]
                    face_crop_pil = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    face_tensor = transform(Image.fromarray(face_crop_pil)).unsqueeze(0)

                    # Run the model to get yaw, pitch, and roll
                    with torch.no_grad():
                        yaw, pitch, roll = pose_net(face_tensor)

                        # Convert to predicted angles similar to standalone script
                        yaw_predicted = torch.argmax(yaw, dim=1).item()
                        pitch_predicted = torch.argmax(pitch, dim=1).item()
                        roll_predicted = torch.argmax(roll, dim=1).item()

                        # Adjust the scale
                        yaw_predicted = float((yaw_predicted - 33) * 3)
                        pitch_predicted = float((pitch_predicted - 33) * 3)
                        roll_predicted = float((roll_predicted - 33) * 3)

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
                    pose_smoothed = [(avg_yaw, avg_pitch, avg_roll)]

                    # Assign face_id before the try-except block
                    face_id = track_id  # Using track_id as face_id for consistency

                    # Run DeepFace emotion analysis on the cropped face
                    try:
                        # Run DeepFace emotion analysis
                        emotion_analysis = DeepFace.analyze(
                            face_crop_pil,
                            actions=['emotion'],
                            enforce_detection=True,  # Set to True for debugging
                            detector_backend='mtcnn'
                            # You can experiment with other backends like 'opencv', 'dlib', etc.
                        )

                        # Handle cases where emotion_analysis is a list
                        if isinstance(emotion_analysis, list):
                            if len(emotion_analysis) > 0:
                                emotion_analysis = emotion_analysis[0]
                            else:
                                raise ValueError("DeepFace.analyze returned an empty list.")

                        # Extract dominant emotion and emotion scores
                        dominant_emotion = emotion_analysis.get('dominant_emotion', 'N/A')
                        emotion_data = emotion_analysis.get('emotion', {})

                        # Find the highest confidence emotion
                        if emotion_data:
                            dominant_emotion_conf = max(emotion_data, key=emotion_data.get)
                            confidence = emotion_data[dominant_emotion_conf]
                            if confidence >= 80:
                                emotion_category = map_emotion_to_category(dominant_emotion_conf)
                            else:
                                emotion_category = 'Neutral'
                        else:
                            emotion_category = 'Neutral'

                        LOGGER.info(f"Face {face_id} - Dominant Emotion: {dominant_emotion}")
                        LOGGER.info(f"Face {face_id} - Emotion Category: {emotion_category}")
                        LOGGER.info(f"Face {face_id} - Emotion Scores: {emotion_data}")

                        # Determine engagement based on head pose and emotion category
                        if emotion_category in ['Positive', 'Neutral']:
                            engaged_faces_frame.add(face_id)  # Add to engaged set
                        elif emotion_category == 'Negative':
                            negative_faces.add(face_id)  # Add to negative set

                    except Exception as e:
                        LOGGER.error(f"Face {face_id} - Emotion Analysis Error: {e}")
                        emotion_category = 'Neutral'

                    # Determine peer discussion based on yaw angle
                    is_peer_discussion = abs(avg_yaw) > 30

                    if is_peer_discussion:
                        discussion_faces_frame.add(face_id)  # Add to discussion set

                    if emotion_category in ['Positive', 'Neutral'] or is_peer_discussion:
                        total_faces_frame += 1  # Count only engaged or in discussion

                    # Draw the pose estimation and labels
                    landmarks = [[((x1 + x2) // 2, (y1 + y2) // 2)]]
                    bbox_dict = {"face": [x1, y1, x2, y2]}

                    drawer.draw_axis(im0, pose_smoothed, landmarks, bbox_dict, emotion_category, face_id,
                                     axis_size=100)  # Adjust axis_size as needed

                    # Log head pose and emotion information to terminal
                    LOGGER.info(
                        f"Face {face_id}: Yaw: {avg_yaw:.2f}, Pitch: {avg_pitch:.2f}, Roll: {avg_roll:.2f}, Detected Emotion: {emotion_category}")

                    # Log detailed emotion data for each face
                    for emotion, confidence in emotion_data.items():
                        LOGGER.info(f"Face {face_id} - Emotion: {emotion}, Confidence: {confidence:.2f}")

                    if save_crop:
                        save_one_box([x1, y1, x2, y2], im0, file=save_dir / 'crops' / f'Face_{face_id}.jpg', BGR=True)

            # **Calculate engagement scores**
            if total_faces_frame > 0:
                engaged_score = (len(engaged_faces_frame) / total_faces_frame) * 100
                discussion_score = (len(discussion_faces_frame) / total_faces_frame) * 100
                total_engagement_score = ((len(engaged_faces_frame) + len(
                    discussion_faces_frame)) / total_faces_frame) * 100
                total_engagement_score = min(total_engagement_score, 100.0)
            else:
                engaged_score = 0
                discussion_score = 0
                total_engagement_score = 0

            # Determine classroom mode
            if engaged_score > discussion_score:
                mode = "Lecture Mode"
                total_engagement_score = engaged_score
            elif engaged_score < discussion_score:
                mode = "Peer Discussion Mode"
                total_engagement_score = discussion_score
            else:
                mode = "Mixed Engagement"
                total_engagement_score = (engaged_score + discussion_score) / 2

            # Log engagement and mode
            LOGGER.info(
                f"Engagement Score: {len(engaged_faces_frame)}/{total_faces_frame} ({engaged_score:.2f}%) engaged")
            LOGGER.info(
                f"Discussion Score: {len(discussion_faces_frame)}/{total_faces_frame} ({discussion_score:.2f}%) in discussion")
            LOGGER.info(f"Total Engagement Score: {total_engagement_score:.2f}%")
            LOGGER.info(f"Current Classroom Mode: {mode}")

            # **Overlay the summary information on the frame**
            drawer.draw_summary(im0, len(engaged_faces_frame), len(discussion_faces_frame), total_faces_frame,
                                engaged_score, discussion_score, total_engagement_score, mode)

        else:
            LOGGER.info('No detections')

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
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream or images
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                save_path = str(Path(save_path).with_suffix('.mp4'))  # enforce .mp4 extension
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            vid_writer.write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # After all frames are processed
    # Release video writer
    if save_img and isinstance(vid_writer, cv2.VideoWriter):
        vid_writer.release()

    # Calculate final engagement scores after all frames are processed
    if total_faces_frame > 0:
        engaged_score = (len(engaged_faces_frame) / total_faces_frame) * 100
        discussion_score = (len(discussion_faces_frame) / total_faces_frame) * 100
        total_engagement_score = ((len(engaged_faces_frame) + len(discussion_faces_frame)) / total_faces_frame) * 100
        total_engagement_score = min(total_engagement_score, 100.0)
    else:
        engaged_score = 0
        discussion_score = 0
        total_engagement_score = 0

    # Determine final classroom mode
    if engaged_score > discussion_score:
        mode = "Lecture Mode"
        total_engagement_score = engaged_score
    elif engaged_score < discussion_score:
        mode = "Peer Discussion Mode"
        total_engagement_score = discussion_score
    else:
        mode = "Mixed Engagement"
        total_engagement_score = (engaged_score + discussion_score) / 2

    # Log final engagement and mode
    LOGGER.info(
        f"Final Engagement Score: {len(engaged_faces_frame)}/{total_faces_frame} ({engaged_score:.2f}%) engaged")
    LOGGER.info(
        f"Final Discussion Score: {len(discussion_faces_frame)}/{total_faces_frame} ({discussion_score:.2f}%) in discussion")
    LOGGER.info(f"Final Total Engagement Score: {total_engagement_score:.2f}%")
    LOGGER.info(f"Final Classroom Mode: {mode}")

    # **Overlay the final summary information on the last frame (optional)**
    # Uncomment the following line if you want to overlay the summary on the last frame
    # drawer.draw_summary(im0, len(engaged_faces_frame), len(discussion_faces_frame), total_faces_frame, engaged_score, discussion_score, total_engagement_score, mode)

    # Optionally, overlay the final total engagement score directly
    cv2.putText(im0, f'Final Total Engagement: {total_engagement_score:.2f}%',
                (10, im0.shape[0] - 10),  # Position at bottom-left corner
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2, cv2.LINE_AA)  # Yellow color

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\nResults saved to {save_dir}"
        LOGGER.info(s)
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolo.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images',
                        help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                        help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
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
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
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
