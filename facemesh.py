import time
import mediapipe as mp
import cv2

print('setup complete!! good to go!')

cap = cv2.VideoCapture(0)
ptime = 0
mpMesh = mp.solutions.face_mesh
mpDraw = mp.solutions.drawing_utils
mesh = mpMesh.FaceMesh(max_num_faces=1)
drawSpec = mpDraw.DrawingSpec(thickness=1,circle_radius=2)

while 1:
    s,img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    res = mesh.process(imgRGB)
    meshlist =[]
    if res.multi_face_landmarks:
        for facelm in res.multi_face_landmarks:
            #h,w,c=img.shape
            mpDraw.draw_landmarks(img,facelm,mpMesh.FACEMESH_CONTOURS,drawSpec,drawSpec)
            #total points on the face mesh - 468 points...!!!
            for id,lm in enumerate(facelm.landmark):
                h,w,c = img.shape
                x,y = int(lm.x*w) , int(lm.y*h)
                meshlist.append([id,x,y])

    ctime = time.time()
    fps=1/(ctime-ptime)
    ptime=ctime
    cv2.putText(img,f'FPS:{int(fps)}',(50,50),3,cv2.FONT_HERSHEY_PLAIN,(0,0,255),2)
    cv2.imshow("Face mesh",img)
    cv2.waitKey(1)