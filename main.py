import cv2
import people_recognition

image = cv2.imread("data/people_pics/party1.jpg")

recognizer = people_recognition.PeopleRecognition()
people = recognizer.crop_people(image)

print(len(people))

for i, person in enumerate(people):
    cv2.namedWindow("img_" + str(i), cv2.WINDOW_NORMAL)
    cv2.imshow("img_" + str(i), person)

cv2.waitKey(0)
cv2.destroyAllWindows()
