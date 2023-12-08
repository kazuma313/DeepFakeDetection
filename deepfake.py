import cv2 
import numpy as np 
import classify as clf


def clf_vidio(filepath, name_vidio='vidio'):
    ratioW = 10
    ratioH = 20
    cap = cv2.VideoCapture(filepath) 
    final_clf_result = 0
    detected_face = 0
    clf_result_count = 0
    result = ""
    
    if (cap.isOpened()== False): 
        print("Error opening video file") 
    else: 
        while(cap.isOpened()): 
            ret, frame = cap.read() 
            if ret == True: 
                # row, column, _ =  frame.shape
                gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_classifier = cv2.CascadeClassifier(
                                                        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                                                        )
                face = face_classifier.detectMultiScale(
                                                        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10)
                                                        )
                
                try:
                    if len(face[0]) == 4:     
                        detected_face += 1 
                        x, y, w, h = face[0] 
                          
                        offsetW = (ratioW / 100)*w
                        w = int(w + offsetW*1.5)
                        x = int(x - offsetW)
                               
                        offsetH = (ratioH / 100)*h
                        h = int(h + offsetH*3)
                        y = int(y - offsetH*2)
                        
                        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img_rgb = img_rgb[y:(h+y) , x:(w+x)]
                        resized_frame = cv2.resize(img_rgb, (256, 256)) 
                        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)/255. 
                        predict = clf.model.predict(np.expand_dims(resized_frame, axis=0))
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
                        cv2.putText(frame, "Scanning...",
                                    (x, y*2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        print(predict)
                        if predict > 0.90:
                            clf_result_count += 1
                        else:
                            pass
                except:
                    pass
                else:
                    pass  
                cv2.imshow(name_vidio, frame)
                if cv2.waitKey(25) & 0xFF == ord('q'): 
                    break
            else: 
                break   
        cap.release() 
        cv2.destroyAllWindows() 
        try:
            final_clf_result = clf_result_count/detected_face
            print("Final Result :",final_clf_result)
            if final_clf_result > 0.80:
                result = "Real"
            else:
                result = "Fake"
        except Exception as e:
            print(e)
        return result, final_clf_result


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=True,
        help="path to input video file")
    args = vars(ap.parse_args())
    result, final_clf_result = clf_vidio(filepath=args["video"], name_vidio=args["video"])
    hasil = {result : final_clf_result}
    print(hasil)


