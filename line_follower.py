import cv2
import numpy as np

# Video dosyasını aç
cap = cv2.VideoCapture('line-yatay.mp4')

# Blur, kontrast, doygunluk ve parlaklık için ana değerler
blur_val = 1
contrast = 2.0
saturation = 0.0
brightness = 50

# HSV için sabit alt ve üst değerler
lower = np.array([0, 0, 0])
upper = np.array([30, 30, 220])

# VideoWriter ayarları
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output_segmented.mp4', fourcc, out_fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w = frame.shape[:2]

    # KARE KOORDİNATLARI
    red_box = (int(frame_w*0.35), int(frame_h*0.1), int(frame_w*0.3), int(frame_h*0.2))
    blue_box = (int(frame_w*0.1), int(frame_h*0.4), int(frame_w*0.2), int(frame_h*0.2))
    green_box = (int(frame_w*0.7), int(frame_h*0.4), int(frame_w*0.2), int(frame_h*0.2))

    # Tolerans karesi (kırmızı kutunun ortasında küçük bir kutu)
    tol_w, tol_h = int(red_box[2]*0.3), int(red_box[3]*0.5)
    tol_x = red_box[0] + (red_box[2] - tol_w)//2
    tol_y = red_box[1] + (red_box[3] - tol_h)//2
    tolerance_box = (tol_x, tol_y, tol_w, tol_h)

    # Görüntü işlemleri
    frame_adj = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
    hsv = cv2.cvtColor(frame_adj, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[...,1] = np.clip(hsv[...,1] * saturation, 0, 255)
    hsv = hsv.astype(np.uint8)
    frame_adj = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if blur_val > 1:
        frame_adj = cv2.GaussianBlur(frame_adj, (blur_val, blur_val), 0)
    hsv = cv2.cvtColor(frame_adj, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    # Kontur bul ve birleştir
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    combined_mask = np.zeros_like(mask)
    for cnt in contours:
        if cv2.contourArea(cnt) > 1000:
            cv2.drawContours(combined_mask, [cnt], -1, 255, -1)
    kernel = np.ones((15, 15), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = frame.copy()
    cv2.drawContours(output, combined_contours, -1, (0, 255, 0), 2)

    # KARELERİ ÇİZ
    cv2.rectangle(output, (red_box[0], red_box[1]), (red_box[0]+red_box[2], red_box[1]+red_box[3]), (0,0,255), 2)
    cv2.rectangle(output, (blue_box[0], blue_box[1]), (blue_box[0]+blue_box[2], blue_box[1]+blue_box[3]), (255,0,0), 2)
    cv2.rectangle(output, (green_box[0], green_box[1]), (green_box[0]+green_box[2], green_box[1]+green_box[3]), (0,255,0), 2)
    cv2.rectangle(output, (tolerance_box[0], tolerance_box[1]), (tolerance_box[0]+tolerance_box[2], tolerance_box[1]+tolerance_box[3]), (0,0,255), 1)

    # ALAN KONTROLÜ
    red_area = combined_mask[red_box[1]:red_box[1]+red_box[3], red_box[0]:red_box[0]+red_box[2]]
    blue_area = combined_mask[blue_box[1]:blue_box[1]+blue_box[3], blue_box[0]:blue_box[0]+blue_box[2]]
    green_area = combined_mask[green_box[1]:green_box[1]+green_box[3], green_box[0]:green_box[0]+green_box[2]]
    tol_area = combined_mask[tolerance_box[1]:tolerance_box[1]+tolerance_box[3], tolerance_box[0]:tolerance_box[0]+tolerance_box[2]]

    direction = ""

    # Kırmızı kutuda alan var mı?
    if cv2.countNonZero(red_area) > 0:
        # Kırmızı kutudaki segmentasyonun dikey orta noktası
        red_mask = np.zeros_like(combined_mask)
        red_mask[red_box[1]:red_box[1]+red_box[3], red_box[0]:red_box[0]+red_box[2]] = red_area
        moments = cv2.moments(red_mask)
        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            # Tolerans kutusunun x aralığı
            tol_left = tolerance_box[0]
            tol_right = tolerance_box[0] + tolerance_box[2]
            if tol_left <= cx <= tol_right and cv2.countNonZero(tol_area) > 0:
                direction = "ileri"
            elif cx > tol_right:
                direction = "sağa kay"
            elif cx < tol_left:
                direction = "sola kay"
        else:
            direction = "ileri"
    else:
        # Kırmızıda alan yoksa, yeşil ve maviye bak
        if cv2.countNonZero(green_area) > 0:
            direction = "sağ"
        elif cv2.countNonZero(blue_area) > 0:
            direction = "sol"

    if direction:
        cv2.putText(output, direction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)

    # Videoya yaz
    out.write(output)

    cv2.imshow('Segmentasyon', output)
    cv2.imshow('Birleşik Maske', combined_mask)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()