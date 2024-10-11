import argparse
import os
import platform
import sys
from pathlib import Path
import torchvision.transforms as transforms
import numpy as np
import cv2
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

# Adding the models directory to the path
MODELS_DIR = ROOT.parents[2] / 'models'  # Assuming models is at root level
if str(MODELS_DIR) not in sys.path:
    sys.path.append(str(MODELS_DIR))  # add MODELS_DIR to PATH

# Import the required components for head pose estimation
from deepHeadPoseLiteMaster import stable_hopenetlite
from deepHeadPoseLiteMaster.drawer import Drawer
from PIL import Image
from deepface import DeepFace

# Other necessary imports...
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
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
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

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
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Process predictions
                engaged_count = 0  # Initialize count for engaged students
                total_faces = 0  # Total faces detected
                peer_discussion_count = 0  # Number of students chatting with someone else

                # Write results
                for idx, (*xyxy, conf, cls) in enumerate(reversed(det)):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))

                    # Crop face and run head pose estimation
                    x1, y1, x2, y2 = map(int, xyxy)

                    # Increase the size of the bounding box by 10% for better head pose estimation
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

                    # Run DeepFace emotion analysis on the cropped face
                    try:
                        # Run DeepFace analysis on the cropped face with "emotion" action
                        emotion_analysis = DeepFace.analyze(face_crop_pil, actions=['emotion'], enforce_detection=False)

                        # Extract the dominant emotion
                        dominant_emotion = emotion_analysis[0]['dominant_emotion'] if isinstance(emotion_analysis,
                                                                                                 list) else \
                        emotion_analysis['dominant_emotion']
                        emotion_data = emotion_analysis[0]['emotion'] if isinstance(emotion_analysis, list) else \
                        emotion_analysis['emotion']

                        # Format emotion data with each emotion and value on a new line, rounded to 2 decimal places
                        formatted_emotion_data = '\n'.join(
                            [f"{emotion}: {confidence:.2f}" for emotion, confidence in emotion_data.items()])
                    except Exception as e:
                        dominant_emotion = "N/A"  # If detection fails, set to not available
                        formatted_emotion_data = "Emotion detection failed"

                    # Draw the detected emotion and the detailed analysis on the image
                    emotion_label_position_y = y1 - 17 if y1 - 17 > 0 else y1 + 15
                    cv2.putText(im0, f"Dominant Emotion: {dominant_emotion}", (x1, emotion_label_position_y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.75, (0, 255, 0), 2)

                    # Determine engagement type based on yaw and pitch
                    is_engaged_forward = (30 > yaw_predicted > -30) and (30 > pitch_predicted > -50)
                    is_peer_discussion = abs(yaw_predicted) > 30

                    if is_engaged_forward:
                        engaged_count += 1
                    elif is_peer_discussion:
                        peer_discussion_count += 1

                    total_faces += 1

                    # Draw the engagement label above the confidence score
                    label_position_y = y1 - 60 if y1 - 60 > 0 else y1 + 15
                    face_id = f'Face {idx + 1}'
                    cv2.putText(im0, face_id, (x1, label_position_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

                    # Draw the pose estimation on the image
                    pose = [(yaw_predicted, pitch_predicted, roll_predicted)]
                    landmarks = [[((x1 + x2) // 2, (y1 + y2) // 2)]]
                    bbox = {0: {"face": [x1, y1, x2, y2]}}

                    drawer.draw_axis(im0, pose, landmarks, bbox)

                    # Print the head pose information for each face
                    LOGGER.info(f"{face_id}: Yaw: {yaw_predicted:.2f}, Pitch: {pitch_predicted:.2f}, Roll: {roll_predicted:.2f}, Detected Emotion: {dominant_emotion}")

                    # Log detailed emotion data for each face
                    for emotion, confidence in emotion_data.items():
                        LOGGER.info(f"{face_id} - Emotion: {emotion}, Confidence: {confidence:.2f}")

                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == 27:  # ESC key to stop
                    break

            # Calculate engagement score if faces were detected
            if total_faces > 0:
                engaged_score = (engaged_count / total_faces) * 100
                discussion_score = (peer_discussion_count / total_faces) * 100

            # Determine classroom mode
            if engaged_score > discussion_score:
                mode =  "Lecture Mode"
            elif engaged_score < discussion_score:
                mode = "Peer Discussion Mode"
            else:
                mode = "Mixed Engagement"

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

        # Draw engagement score and mode on the image
        engagement_text = f"Engagement Score: {engaged_count}/{total_faces} ({engaged_score:.2f}%)"
        discussion_text = f"Discussion Score: {peer_discussion_count}/{total_faces} ({discussion_score:.2f}%)"
        mode_text = f"Mode: {mode}"

        # Log engagement and mode
        LOGGER.info(f"Engagement Score: {engaged_count}/{total_faces} ({engaged_score:.2f}%) engaged")
        LOGGER.info(f"Discussion Score: {peer_discussion_count}/{total_faces} ({discussion_score:.2f}%) in discussion")
        LOGGER.info(f"Current Classroom Mode: {mode}")

    # Draw engagement score and mode on the image
    #cv2.putText(im0s, "engage2 test", (10, im0s.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
    #            (0, 255, 0), 2)
    #cv2.putText(im, "discussion2 test", (10, im.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0),
    #            2)
    #cv2.putText(im0, "mode2 test", (10, im0.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0),
    #            2)

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolo.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
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