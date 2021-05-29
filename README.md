# Real-time-Driver-Distraction-Detection-System-with-Continuous-Face-Tracking
INTRODUCTION TO THE PROJECT-

This project detects driver's face and tracks continuously to check for driver's distraction. This also make sure if the same driver is driving the vehicle who gave the alcohol breath alcohol testing. The code is designed in such a way that when the person is giving breath alcohol testing the camera starts and takes 100-200 photos of the person. Then a model is trained on these photos to recoginze that person. Then camera takes real time pictures of the driver to recognize if the same person is driving. If the face is not matched it alerts by raising voice alert. The code also checks for yawning and eye closure state of the driver and raises voice alerts if it finds driver yawning more than three times or if it finds that eye is near to close. If the driver don't look at the front then also voice alert is raised.

PREREQUISITE-

opencv-contrib-python (Because we are going to use LBPH which is not found in opencv-python)

STEPS OF THE PROJECT - 

The code for the steps given below can be found in face_drowsiness_yawn2.py.
1. Collection of Driver's photos - First a webcam is opened face of the person is detected using harcaascade classifier. Haarcascade classifier is an object detection algorithm which is used to detect face. 200 photos of the person's face is taken who is blowing into alcohol breath-testing pipe. After taking photos camera stops.

2. Training of face recogintion model - A very fast algorithm is used here for training the model which is Local Binary Patterns Histogram(LBPH). This can be found in opencv package. So all the photos are fed to the LBPH model and now the model is ready to recogize the person based on the confidence number.

3. Functions for yawn and drowsiness detection - For detecting yawn and drowsiness of a person facial landmarks are used. The pre-trained facial landmark detector inside the dlib library is used to estimate the location of 68-(x,y) coordinates that map to the facial structure of the face. These 68-(x,y) coordinates represent the important regions of the face like mouth, left eyebrow, right eyebrow, left eye, right eye, nose, and jaw. Of these, the (x,y) coordinates of the left eye, right eye, and mouth are needed. Based on these landmarks we calculate eye aspect ratio and mouth aspect ratio. Mouth aspect ratio is distance between upper lip and lower lip. If this goes above a threshold it is detected a yawn status. Eye aspect ratio measures the closure of eyes. If it goes below a threshold it is detected that driver is feeling drowsy.

4. Final Face Tracking of Driver- Now the camera starts and the real time pictures are continuously clicked. The pictures are first passed to haarcascade classifier to locate the face and then it is cropped and sent to face recogintion model to check for the same person's face and also to functions of yawn and drowsiness. If the person's face does not match then continuous voice alerts are raised, if the person is not looking in the front then a voice alert is raised warning him to look at the front. If the person is found yawning more than three times or if he/she is found as therir eyes closed they are warned. The camera stops on interrupting it with keyboard and the program stops.

FUTURE SCOPE OF THE PROJECT- 

1. We can also detect the presence of breath testing pipe while clicking photos of driver. This will make sure that the driver is blowing into the same pipe and not pretending. I am working on this.

2. We can merge cellphone use alert system also in this project using somw less computational expensive way. Then this system will become a more efficient driver distraction detection system. 

