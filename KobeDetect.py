import cv2
import numpy as np
import face_recognition

# 初始化摄像头并设置分辨率
video_capture = cv2.VideoCapture(2)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 加载并编码已知人脸
known_face_encodings = []
known_face_names = ['Kobe']  # 这里可以根据实际情况添加多个已知人脸的名字
image_paths = ['imageBasic/kobe1.jpg', 'imageBasic/kobe2.jpg', 'imageBasic/kobe3.jpg', 'imageBasic/kobe4.jpg', 'imageBasic/kobe5.jpg', 'imageBasic/kobe6.jpg', 'imageBasic/kobe7.jpg', 'imageBasic/kobe8.jpg', 'imageBasic/kobe9.jpg', 'imageBasic/kobe10.jpg', 'imageBasic/kobe11.jpg', 'imageBasic/kobe12.jpg', 'imageBasic/kobe13.jpg', 'imageBasic/kobe14.jpg']  # 同一个人的不同表情或环境下的照片

for path in image_paths:
    img = face_recognition.load_image_file(path)
    img_encodings = face_recognition.face_encodings(img)
    if img_encodings:  # 检查是否成功生成了编码
        known_face_encodings.append(img_encodings[0])
    else:
        print(f"Warning: No faces found in {path}.")

if not known_face_encodings:
    print("No known face encodings. Exiting...")
    exit()

while True:
    ret, frame = video_capture.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # 当没有检测到脸时跳过
        if not face_encodings:
            continue

        distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if distances.size > 0:  # 检查距离列表是否为空
            best_match_index = np.argmin(distances)
            distance = distances[best_match_index]

            if distance < 0.45:
                name = known_face_names[0]  # 由于所有编码都代表同一个人，因此总是使用第一个名字
            else:
                name = "NOT Kobe"

            # 绘制框和标签，包括距离值
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            label = f"{name}-{1 - distance:.2f}"  # 显示名字和距离
            font = cv2.FONT_HERSHEY_DUPLEX
            # 首先，绘制黑色边框（背景）
            thickness = 3  # 文本边框的粗细
            cv2.putText(frame, label, (left + 6, bottom - 6), font, 0.5, (0, 0, 0), thickness)

            # 然后，绘制白色文本（前景）
            thickness = 1  # 实际文本的粗细
            cv2.putText(frame, label, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), thickness)

    cv2.imshow('Kobe detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

