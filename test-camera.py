import cv2

cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)

while (True):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow("FRAME", frame)
    #cv2.imshow("GRAY", gray)
    
    k = cv2.waitKey(0) & 0xff
    if k == 27:
        break
    
cap.release()
cv2.detroyAllWindows()