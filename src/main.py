import subprocess
import sys
from pathlib import Path

# --- User-defined video file path ---
SOURCE = Path('assets/video/Zoom.mp4')

# --- Threshold values ---
engagement_timeline_threshold = 1.0  # Engagement duration threshold (in seconds)
yaw_threshold = 35  # Left or right rotation beyond this angle (abs value)
upwards_pitch_threshold = 20
downwards_pitch_threshold = -30
emotion_threshold = 90  # Dominant emotion threshold in percentage
scaling_factor = 0.30  # Bounding box scaling factor

# Get the path to the directory where headPoseMain.py is located
FILE = Path(__file__).resolve()
ROOT = FILE.parent  # Root directory of the project

# Paths to detect.py, weights, and default project folder
DETECT_PY = ROOT / 'faceDetection' / 'yolov9FaceDetection' / 'yolov9' / 'detect.py'
WEIGHTS = ROOT / 'faceDetection' / 'yolov9FaceDetection' / 'yolov9' / 'best.pt'
PROJECT = ROOT / 'runs' / 'detect'

# Ensure detect.py, weights, and source video exist
for path, name in [(DETECT_PY, "detect.py"), (WEIGHTS, "Weights file")]:
    if not path.exists():
        print(f"{name} not found at {path}")
        sys.exit(1)

# Resolve SOURCE to an absolute path if it’s relative to ensure compatibility
SOURCE = SOURCE if SOURCE.is_absolute() else (ROOT / SOURCE)

# Check if the source video exists
if not SOURCE.exists():
    print(f"Source video not found at {SOURCE}")
    sys.exit(1)


# Function to get an incremented experiment directory
def get_incremented_exp_dir(base_dir, exp_name='exp'):
    exp_dir = base_dir / exp_name
    counter = 1
    while exp_dir.exists():  # Increment until we find a non-existing directory
        exp_dir = base_dir / f"{exp_name}{counter}"
        counter += 1
    return exp_dir


# Get the next available experiment directory
exp_dir = get_incremented_exp_dir(PROJECT)

# Build the command to run detect.py
command = [
    sys.executable,  # Use the same Python interpreter
    str(DETECT_PY),
    '--weights', str(WEIGHTS),
    '--source', str(SOURCE),
    '--project', str(PROJECT),
    '--name', exp_dir.name,
    '--exist-ok',
    '--engagement-timeline-threshold', str(engagement_timeline_threshold),
    '--yaw-threshold', str(yaw_threshold),
    '--upwards-pitch-threshold', str(upwards_pitch_threshold),
    '--downwards-pitch-threshold', str(downwards_pitch_threshold),
    '--emotion-threshold', str(emotion_threshold),
    '--scaling-factor', str(scaling_factor),
]

# Run the detect.py script
try:
    result = subprocess.run(command, cwd=str(DETECT_PY.parent))
    # Check if the subprocess exited with an error
    if result.returncode != 0:
        print("Error running detect.py")
        sys.exit(1)
    else:
        print(f"Detection completed successfully. Results saved to {exp_dir}")
except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)
