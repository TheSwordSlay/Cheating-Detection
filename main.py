import cv2
import numpy as np 
import dlib
from math import hypot

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def midpoint(p1, p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def berkedip_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = (midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2])))
    center_bottom = (midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4])))

    #garis hijau
    #hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    #ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1]-center_bottom[1]))

    ratio = hor_line_lenght/ver_line_lenght    
    return ratio

def get_ratio_arah_lihat(eye_points, facial_landmarks):
    area_mata_kiri = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
    (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
    (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
    (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
    (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
    (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)

    #garis merah
    #cv2.polylines(frame, [area_mata_kiri], True, (0, 0, 255,), 2)

    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)

    cv2.polylines(mask, [area_mata_kiri], True, 255, 2)
    cv2.fillPoly(mask, [area_mata_kiri], 255)
    mata = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(area_mata_kiri[:, 0])
    max_x = np.max(area_mata_kiri[:, 0])
    min_y = np.min(area_mata_kiri[:, 1])
    max_y = np.max(area_mata_kiri[:, 1])

    gray_eye = mata[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    threshold_kiri = threshold_eye[0: height, 0: int(width/2)]
    putih_kiri = cv2.countNonZero(threshold_kiri)

    threshold_kanan = threshold_eye[0: height, int(width/2): width]
    putih_kanan = cv2.countNonZero(threshold_kanan)

    if putih_kanan == 0:
        ratio_arah_lihat = 1
    elif putih_kiri == 0:
        ratio_arah_lihat = 5
    else:
        ratio_arah_lihat = putih_kiri/putih_kanan
    return ratio_arah_lihat

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        # cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

        landmarks = predictor(gray, face)

        #deteksi berkedip
        ratio_mata_kiri = berkedip_ratio([36, 37, 38, 39, 40, 41], landmarks)
        ratio_mata_kanan = berkedip_ratio([42, 43, 44, 45, 46, 47], landmarks)
        ratio_berkedip = (ratio_mata_kiri + ratio_mata_kanan)/2

        if ratio_berkedip > 5.7:
            cv2.putText(frame, "Kedip", (50, 100), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 0, 0))

        #deteksi arah pandangan mata
        ratio_arah_lihat_mata_kiri = get_ratio_arah_lihat([36, 37, 38, 39, 40, 41], landmarks)
        ratio_arah_lihat_mata_kanan = get_ratio_arah_lihat([42, 43, 44, 45, 46, 47], landmarks)
        ratio_arah_lihat = (ratio_arah_lihat_mata_kanan + ratio_arah_lihat_mata_kiri)/2
        
        if ratio_arah_lihat <= 0.3:
            if ratio_berkedip < 5.7:
                cv2.putText(frame, "mencurigakan", (50, 100), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 3)
        elif 0.3 < ratio_arah_lihat < 3:
            if ratio_berkedip < 5.7:
                cv2.putText(frame, "tidak mencurigakan", (50, 100), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 3)
        else:
            if ratio_berkedip < 5.7:
                cv2.putText(frame, "mencurigakan", (50, 100), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 3)
        # presentasi
        # cv2.imshow("Mata", eye)
        # cv2.imshow("Mata thereshold", threshold_eye) 2
        # cv2.imshow("Mata kiri", mata_kiri)
        # cv2.imshow("Kiri", threshold_kiri) 2
        # cv2.imshow("Kanan", threshold_kanan) 2

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
