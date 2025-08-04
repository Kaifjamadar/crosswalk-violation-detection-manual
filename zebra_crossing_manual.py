

from ultralytics import YOLO
import cv2
import numpy as np

# YOLOv8 model
model = YOLO("yolov8n.pt")

# video
cap = cv2.VideoCapture(r'zebra_crossing.mp4')

# Zebra crossing polygon setup
zebra_zone = None
points = []

# zebra crossing
def draw_polygon(event, x, y, flags, param):
    global points, zebra_zone
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        zebra_zone = np.array(points, dtype=np.int32)

# Draw zone window
ret, frame = cap.read()
if not ret:
    print("Failed to read video.")
    exit()

cv2.namedWindow("Define Zebra Zone")
cv2.setMouseCallback("Define Zebra Zone", draw_polygon)

while zebra_zone is None:
    temp_frame = frame.copy()
    for pt in points:
        cv2.circle(temp_frame, pt, 5, (0, 255, 0), -1)
    if len(points) > 1:
        cv2.polylines(temp_frame, [np.array(points)], False, (255, 0, 0), 2)
    cv2.imshow("Define Zebra Zone", temp_frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break
cv2.destroyWindow("Define Zebra Zone")

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    for box in results.boxes:
        cls = int(box.cls[0])
        if cls != 0:  # Only person
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Approximate leg point (bottom center of the box)
        leg_x = int((x1 + x2) / 2)
        leg_y = int(y2)

        # Check if leg point is inside zebra zone
        inside = cv2.pointPolygonTest(zebra_zone, (leg_x, leg_y), False) >= 0

        # Draw bounding box
        color = (0, 255, 0) if inside else (0, 0, 255)
        label = "Legal" if inside else "Illegal"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.circle(frame, (leg_x, leg_y), 4, color, -1)

        # Optionally draw leg point
        cv2.circle(frame, (leg_x, leg_y), 4, color, -1)

    # Draw zebra zone
    if zebra_zone is not None:
        cv2.polylines(frame, [zebra_zone], True, (255, 255, 0), 2)

    cv2.imshow("Result", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()



