
import cv2

# Path configuration
preTrainedFaceDetectorPath = "" # pre Trained Face detector path
imagePath = "" # image path


# Load the image and convert it to grayscale
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load the pre trained face detector
detector = cv2.CascadeClassifier(preTrainedFaceDetectorPath)

# Detect faces in the image
faceRects = detector.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=5,
		                              minSize=(80, 80), flags=cv2.CASCADE_SCALE_IMAGE)

# Loop over the faces and draw a rectangle for each
for (x, y, w, h) in faceRects:
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image with faces detected
cv2.imshow("Detected Faces", image)
cv2.waitKey(0)