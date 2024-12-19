# Student Engagement

SDEM (Student Engagement Detection Model) is a tool that leverages computer vision and deep learning to analyze student engagement in a classroom setting. By detecting and analyzing students' facial expressions, this tool generates reports on engagement levels throughout a class session, providing valuable insights for educators.

Created alongside professors Dr. Adballah Mohamed & Dr. Mohamed Shehata from the University of British Columbia Okanagan

## Getting Started

### Prerequisites

1. Clone the repository to your local machine by running the following command:
   ```bash
   git clone https://github.com/Jared-Waldroff/StudentEngagement.git
2. Ensure you have **Python 3.10** installed.
3. Install all required dependencies by running:
   ```bash
   pip install -r requirements.txt

### Download Weights
To run the model, you will need to download the pre-trained weights. Place the downloaded weights into the downloadedWeights folder:

https://drive.google.com/drive/folders/1bKONWeT_4BdGuKLBnXTAoTKvW0rv7lt5?usp=sharing

### Running the Code

1. **Set the Video File Path**:
   - Open the `main.py` file in the project root directory.
   - Locate the line defining `SOURCE` near the top of the file:
     ```python
     SOURCE = Path('your/video/photo/path/here')
     ```
   - Replace `'your/video/photo/path/here'` with the path to your desired video or photo file. You can use either an absolute path (e.g., `C:\Users\user\videos\myfile.mp4` on Windows) or a relative path from the project root (e.g., `assets/video/myfile.mp4`).

2. **Run the Detection Script**:
   - Navigate to the project root directory:
     ```bash
     cd src
     ```
   - Run the following command to start the engagement detection process:
     ```bash
     python headPoseMain.py
     ```

### Output

After running the code, the analysis output will be saved in the `runs/detect` directory in the src directory. Each run will create a new subfolder, such as `exp`, `exp2`, `exp3`, etc., incrementing automatically to keep each experiment’s output organized. Each folder will contain the processed files from the corresponding run.
