import subprocess
import sys
from pathlib import Path

# Get the path to the directory where main.py is located
FILE = Path(__file__).resolve()
ROOT = FILE.parent  # Root directory of the project

# --- User-defined video file path ---
SOURCE = Path('assets/video/classLow.mp4')  # Replace with an absolute or relative path as needed

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
    '--name', exp_dir.name,  # Save results in incremented exp directory
    '--exist-ok',  # Overwrite existing results if necessary
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
