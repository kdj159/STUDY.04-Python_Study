## 증명사진으로 이름 매칭 학습 시켜서 러닝 파일 저장


import face_recognition
import pickle

known_encodings = []
known_names = []

for name, file in [("홍길동", "~/dataset/hong.jpg"), ("김철수", "~/dataset/kim.jpg")]:
    image = face_recognition.load_image_file(file)
    encoding = face_recognition.face_encodings(image)[0]
    known_encodings.append(encoding)
    known_names.append(name)

# 학습 데이터 저장
data = {"encodings": known_encodings, "names": known_names}
with open("~/saved_model/face_data.pkl", "wb") as f:
    pickle.dump(data, f)
