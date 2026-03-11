import face_recognition
import cv2
import pickle

# 저장된 학습 데이터 불러오기
with open("~/saved_model/face_data.pkl", "rb") as f:
    data = pickle.load(f)

# 단체사진 불러오기
group_image = face_recognition.load_image_file("group.jpg")
locations = face_recognition.face_locations(group_image)
encodings = face_recognition.face_encodings(group_image, locations)

group_image_cv = cv2.cvtColor(group_image, cv2.COLOR_RGB2BGR)

for (top, right, bottom, left), face_encoding in zip(locations, encodings):
    matches = face_recognition.compare_faces(data["encodings"], face_encoding)
    name = "Unknown"

    if True in matches:
        idx = matches.index(True)
        name = data["names"][idx]

    cv2.rectangle(group_image_cv, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(group_image_cv, name, (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

cv2.imshow("Result", group_image_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()