import os
import numpy as np
import cv2
from ultralytics import YOLO
# from tracker import Tracker

# from deepsort.tracker import DeepSortTracker
from deep_sort_realtime.deepsort_tracker import DeepSort

# On récupère le chemin relatif de notre vidéo test et on le place dans un objet du cv2 qui permet de lire la vidéo en
# frame
video_path = os.path.join("people.mp4")
cap = cv2.VideoCapture(video_path)

# la méthode cap.read() permet de lire les frames de la vidéo et retourne une valeur logique si oui ou non il y a une
# frame avec la variable ret puis la frame en question.
ret, frame = cap.read()

# Vu que le modèle pré-entrainé de YOLOv8n.pt pour la detection sera suffisant pour notre cas, nous l'avons déjà
# télechargée et donner dans le chemin suivant.
yolo_path = os.path.join("..", "Try_!", "yolov8n.pt")
model = YOLO(yolo_path)

tracker = DeepSort(max_age=5)
# tracker = DeepSortTracker()
# tracker = Tracker()
while ret:
    # Ici, cela nous permet de faire la detection de chaque personne sur la frame et nous retourne les boundings boxes
    # de chaque personne (on parle de personnes, car notre vidéo test contient que de personne) sous forme de listes que
    # nous allons utiliser pour l'entrée du suiveur.

    results = model(frame)

    # On crée une boucle qui va itérer sur la variable results pour récupérer chaque bounding boxe de chaque personne
    # qui va le gardait
    for result in results:
        boxes = result.boxes
        probs = result.probs
        xyxy = boxes.xyxy
        conf = boxes.conf
        xywh= boxes.xywh

    bboxes_xywh = xywh.cpu().numpy()
    bboxes_xywh = np.array(bboxes_xywh, dtype=float)

    tracker.update_tracks(bboxes_xywh, frame=frame)

    for track in tracker.tracks:
        bbox = track.bbox
        track_id = track.track_id

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)))

    # Par exemple, affichez la frame
    cv2.imshow('Frame', frame)
    cv2.waitKey(25)

cap.release()
cv2.destroyAllWindows()
