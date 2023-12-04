import cv2 
import numpy as np 
import classify as clf
from threading import Thread
from queue import Queue

def clf_vidio(filepath, name_vidio='vidio'):
    # Tambahin Exception  
    cap = cv2.VideoCapture(filepath) 
    jumlah_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clf_result_count = 0
    result = ""
    count_frame = 0
    Q = Queue()
    
    if (cap.isOpened()== False): 
        print("Error opening video file") 
        
    # Read until video is completed 
    while(cap.isOpened()): 
        ret, frame = cap.read() 
        if ret == True: 
            # row, column, _ =  frame.shape
            resized_frame = cv2.resize(frame, (256, 256))/255.
            
            predict = clf.model.predict(np.expand_dims(resized_frame, axis=0))
            cv2.imshow(name_vidio, resized_frame)
            print(predict)
            if predict > 0.90:
                clf_result_count += 1
            else:
                pass
            
            if cv2.waitKey(25) & 0xFF == ord('q'): 
                break
        else: 
            break
        
        count_frame += 1
        
    cap.release() 
    cv2.destroyAllWindows() 
    
    try:
        final_clf_result = clf_result_count/jumlah_frame
        print("Final Result :",final_clf_result)
        if final_clf_result > 0.80:
            result = "Real"
        else:
            result = "Fake"
    except Exception as e:
        print(e)
        
    # print(f"====clf_result_count====")
    # print(f"===={clf_result_count}====")
    # print(f"====jumlah frame====")
    # print(f"===={jumlah_frame}====")
    # print(f"====final====")

    # print("result :", result)
    
    return result, final_clf_result

def main():
    list_deepfake_vidio = [
                        # "deepfake0.mp4", 
                        # "deepfake1.mp4", 
                        # "deepfake2.mp4", 
                        # "deepfake3.mp4", 
                        # "deepfake4.mp4",
                        # "deepfake5.mp4",
                        # "deepfake6.mp4",
                        # "deepfake7.mp4",
                        "deepfake0_360.mp4",
                        ]
    
    
    # list_deepfake_vidio = ["original0.MOV", 
    #                        "original1.MOV", 
    #                        "original2.MOV", 
    #                        "deepfake3.MP4"]
    list_result = []
    list_final_clf_result = []
    for name_file in list_deepfake_vidio:
        result, final_clf_result = clf_vidio(filepath=f"F:\python\MyProject\deepFake\data\{name_file}", name_vidio=name_file)
        list_result.append(result)
        list_final_clf_result.append(final_clf_result)
    print(list_result)
    print(list_final_clf_result)
    
    # name_file = "deepfake1.mp4"
    # result = clf_vidio(filepath=f"F:\python\MyProject\deepFake\data\{name_file}")
    # print(result)
    
if __name__ == "__main__":
    main()

