import time
import mediapipe as mp
import cv2

print('setup complete!! good to go!')


class FaceMesh():
    def __init__(self,static_image_mode=False,
               max_num_faces=1,
               refine_landmarks=False,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5,
               thickness=1,
               circle_radius=1):
        self.mode = static_image_mode
        self.maxfaces = max_num_faces
        self.lm = refine_landmarks
        self.mincon = min_detection_confidence
        self.mintrack = min_tracking_confidence
        self.mpDraw = mp.solutions.drawing_utils
        self.mpMesh = mp.solutions.face_mesh
        self.Mesh = self.mpMesh.FaceMesh(self.mode,self.maxfaces,self.lm,self.mincon,self.mintrack)
        self.DrawSpec = self.mpDraw.DrawingSpec(thickness,circle_radius)

    def DrawMesh(self,img,draw=True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        meshList =[]
        self.results = self.Mesh.process(imgRGB)
        if self.results.multi_face_landmarks:
            for facelms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,facelms,self.mpMesh.FACEMESH_CONTOURS,self.DrawSpec,self.DrawSpec)
                for id,lm in enumerate(facelms.landmark):
                    h,w,c=img.shape
                    x , y = int(lm.x*w) , int(lm.y*h)
                    meshList.append([id,x,y])
                    #the below line will print the id of each point of the face mesh,
                    #that is a whooping total of 468 points from 0-467.!!!
                    #cv2.putText(img,f'{id}',(x,y),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),1)

        return img,meshList
