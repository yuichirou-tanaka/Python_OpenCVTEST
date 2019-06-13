import cv2

# Haar-like特徴分類器の読み込み
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

def capture_camera(mirror=True, size=None):
    """Capture video from camera"""
    # カメラをキャプチャする
    cap = cv2.VideoCapture(0) # 0はカメラのデバイス番号

    while True:
        # retは画像を取得成功フラグ
        ret, frame = cap.read()

        # 鏡のように映るか否か
        if mirror is True:
            frame = frame[:,::-1]

        # フレームをリサイズ
        # sizeは例えば(800, 600)
        #if size is not None and len(size) == 2:
        #    frame = cv2.resize(frame, size)

        # スクリーンショットを撮りたい関係で1/4サイズに縮小
        frame = cv2.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)))

        gray_frame =  cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 顔を検知
        faces = face_cascade.detectMultiScale(gray_frame)
        for(x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray_frame[y:y+h,x:x+w]
            roi_color = frame[y:y+h,x:x+w]

            eyes = eye_cascade.detectMultiScale(roi_gray)
            for(ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey*eh),(0,255,0),2)

        # 何か処理（ここでは文字列「hogehoge」を表示する）
        #edframe = frame
        #cv2.putText(edframe, 'hogehoge', (0,50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255,0), 3, cv2.LINE_AA)

        # 加工済の画像を表示する
        cv2.imshow('Capture Frame', frame)

        k = cv2.waitKey(1) # 1msec待つ
        if k == 27: # ESCキーで終了
            break

    # キャプチャを解放する
    cap.release()
    cv2.destroyAllWindows()

capture_camera()

