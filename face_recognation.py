import face_recognition

image = face_recognition.load_image_file("./IMG_6753.JPG")
face_locations = face_recognition.face_locations(image)