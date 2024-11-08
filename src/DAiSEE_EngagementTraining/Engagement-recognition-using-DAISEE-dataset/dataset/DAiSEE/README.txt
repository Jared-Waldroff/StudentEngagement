Hi,

Thank you for Downloading DAiSEE.

1) In the folder nameed "Dataset", you will find three sub-folders which are Test, Train and Validation. These have been already created as per the required criteria mentioned.

2) Data-Set Structure
	-Let us take the Test folder as an example.
	-Each folder inside Test contains all the videos of a particular individual.
	-The folder names is the unique anonymous identification given to the users who contributed to DAiSEE.
	-Once you open the videos of a particular user, you will have all the videos pertaining to him/her. These have also been made into folder for ease of frame extraction and further processing you may want to perform on individual videos.
	- Hence the Structure is : Test ---> Subjects ---> Videos of the Subject ---> 10 Sec Video Clip

3) You will find these text files, Test.txt, Train.txt and Validation.txt in their respective folders. These files contain all the clips of the given distribution.

3) Labels
	- All the required labels are present in the Labels folder.

4) Gender 
	-In the Gender Folder you will find the videos clips divided into the Gender of our subjects. You may use this to perform gender specific studies. 
	-For our work, we have considered two Gender categories, Male and Female. Every subject has identified themselves with one of these categories.

5) Scripts
	- We have written some helper scripts to get you started.
	- extractFrames.py : Script to extract all the frames from the video clips. Note that these frames will be stored in the same folder as the Video Clip.
	- hog.py : Script to extract HOG feature of the frames. 

NOTE : You may find holes in the number of the videos for any subject. The has arised due to the data cleaning that was perform. Kindly refer to the paper for more information


SAMPLE STRUCTURE
dataset/
  Training/
    person1/
        video1/
          video1.avi/mp4
        video2/
          video2.avi/mp4
        .
        .
    person2/
        video1/
          video1.avi/mp4
        video2/
          video2.avi/mp4
        .
        .
    .
    .
  Testing/
    person1/
        video1/
          video1.avi/mp4
        video2/
          video2.avi/mp4
        .
        .
    person2/
        video1/
          video1.avi/mp4
        video2/
          video2.avi/mp4
        .
        .
    .
    .
  Validation/
    person1/
        video1/
          video1.avi/mp4
        video2/
          video2.avi/mp4
        .
        .
    person2/
        video1/
          video1.avi/mp4
        video2/
          video2.avi/mp4
        .
        .
    .
    .