# importing libraries 
import cv2 
import numpy as np 
import tensorflow as tf
import classify as clf
import ffmpeg
# Create a VideoCapture object and read from input file 

def clf_vidio(filepath):
    # Tambahin Exception  
    cap = cv2.VideoCapture(filepath) 
    jumlah_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clf_result_count = 0
    result = ""
    
    # Check if camera opened successfully 
    if (cap.isOpened()== False): 
        print("Error opening video file") 
        
    # Read until video is completed 
    while(cap.isOpened()): 
    # Capture frame-by-frame 
        ret, frame = cap.read() 
        if ret == True: 
            _, column, _ = frame.shape
            column = int(column/2)
            # Ubah ukuran frame
            # frame = frame[:, :column, :]
            resized_frame = cv2.resize(frame, (256, 256))/255.
            predict = clf.model.predict(np.expand_dims(resized_frame, axis=0))
            print(predict)
            if predict > 0.5:
                clf_result_count += 1
                print(clf_result_count)
            else:
                pass
            
            # Tampilkan frame yang telah diubah ukurannya
            
            cv2.imshow('Video', resized_frame)
            
        # Press Q on keyboard to exit 
            if cv2.waitKey(25) & 0xFF == ord('q'): 
                break
    
    # Break the loop 
        else: 
            break
    
    # When everything done, release 
    # the video capture object 
    cap.release() 
    
    # Closes all the frames 
    cv2.destroyAllWindows() 
    
    try:
        final_clf_result = clf_result_count/jumlah_frame
        print(final_clf_result)
        if final_clf_result > 85:
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
    
    return result

def main():
    result = clf_vidio(filepath="F:\python\MyProject\deepFake\data\tomCrush.mp4")
    print(result)
    
if __name__ == "__main__":
    main()

