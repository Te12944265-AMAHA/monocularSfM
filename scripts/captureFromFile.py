import numpy as np
import cv2

cap = cv2.VideoCapture('capture/output.avi')
print(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('capture/output_cut.avi',fourcc, 20.0, (1024,512), 0)
start = False

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        out.write(gray)
    if cv2.waitKey(1) & 0xFF == ord('w'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()