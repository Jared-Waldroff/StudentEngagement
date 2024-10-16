# Student Engagement

Student Engagement is a tool that leverages computer vision and deep learning to analyze student engagement in a classroom setting. By detecting and analyzing students' facial expressions, this tool generates reports on engagement levels throughout a class session, providing valuable insights for educators.

## Getting Started

### Prerequisites

1. Clone the repository to your local machine.
2. Ensure you have **Python 3.10** installed.
3. Install all required dependencies by running:
   ```bash
   pip install -r requirements.txt

### Download Weights
To run the model, you will need to download the pre-trained weights. Place the downloaded weights file in the yolov9 directory:

https://drive.google.com/file/d/1PhOHWpWj37cYcSnRzdYnElHbUrFRBSHT/view

### Running the Code
Open a terminal and navigate to the yolov9 directory:
bash
Copy code
cd yolov9
Run the following command to start the engagement detection process:
bash
Copy code
python detect.py --weights best.pt --source "your/image/video/path/here"
    
Replace "your/image/video/path/here" with the path to your input image or video file.
### Output
After running the code, the analysis output will be saved in the runs/detect directory within subfolders like exp, exp2, etc., depending on the number of runs. Each folder contains the processed file.