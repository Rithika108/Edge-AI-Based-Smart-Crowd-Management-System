"""
Phase 1 - Image/Video Acquisition, Room Size and Occupancy
Edge AI-Based Smart Crowd Capacity Monitoring and Alert System

Automatically detects colored floor boundaries when available. Otherwise, trace
the crowd-area fence with an edge-snapping polygon selector.
"""

import sys
import math
import json
import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DISPLAY_WIDTH = 960
CONF = 0.10
YOLO_IMAGE_SIZE = 1536
SAVE_OUTPUT = True
OUTPUT_FILE = os.path.join(BASE_DIR, "phase1_output.mp4")
LIVE_COUNT_FILE = os.path.join(BASE_DIR, "live_count.txt")
MAX_CAPACITY_FILE = os.path.join(BASE_DIR, "max_capacity.json")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def _write_dashboard_value(path, value):
    """Atomically publish a value for the dashboard to read."""
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as file:
        file.write(value)
    os.replace(temp_path, path)

def crowd_status(pct):
    if pct < 50: return "SAFE", (0, 180, 0)
    if pct < 75: return "MODERATE", (0, 200, 230)
    if pct < 100: return "NEAR CAPACITY", (0, 140, 255)
    return "OVERCROWDED", (0, 0, 255)

def choose_input():
    if len(sys.argv) > 1: return sys.argv[1]
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
        path = filedialog.askopenfilename(title="Select an image or video",
            filetypes=[
                ("Images and videos", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.mp4 *.avi *.mov *.mkv *.wmv"),
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("All files", "*.*")
            ])
        root.destroy(); return path
    except Exception:
        return input("Enter image or video file path: ").strip().strip('"')

def ask_number(prompt, default):
    try:
        import tkinter as tk
        from tkinter import simpledialog
        root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
        val = simpledialog.askfloat("Room details", prompt, initialvalue=default, minvalue=0.1)
        root.destroy(); return float(val) if val is not None else float(default)
    except Exception:
        txt = input(f"{prompt} [{default}]: ").strip()
        return float(txt) if txt else float(default)

def resize_frame(frame):
    h, w = frame.shape[:2]; scale = DISPLAY_WIDTH / float(w)
    return cv2.resize(frame, (DISPLAY_WIDTH, int(h * scale)))

def _segments_cross(a, b, c, d):
    def orientation(p, q, r):
        px, py = int(p[0]), int(p[1])
        qx, qy = int(q[0]), int(q[1])
        rx, ry = int(r[0]), int(r[1])
        return (qx - px) * (ry - py) - (qy - py) * (rx - px)

    return (orientation(a, b, c) * orientation(a, b, d) < 0
            and orientation(c, d, a) * orientation(c, d, b) < 0)

def room_geometry(points):
    points = np.asarray(points, dtype=np.int32).reshape((-1, 2))
    if len(points) < 3 or len(np.unique(points, axis=0)) < 3:
        raise ValueError("Select at least three different boundary points.")
    contour = points.reshape((-1, 1, 2))
    for i in range(len(points)):
        a, b = points[i], points[(i + 1) % len(points)]
        for j in range(i + 1, len(points)):
            if j == i or j == (i + 1) % len(points) or (j + 1) % len(points) == i:
                continue
            if _segments_cross(a, b, points[j], points[(j + 1) % len(points)]):
                raise ValueError("Boundary crosses itself; trace the fence in order.")
    area = abs(cv2.contourArea(contour))
    if area <= 0:
        raise ValueError("The selected crowd area has no measurable area.")
    width_px, height_px = cv2.minAreaRect(contour)[1]
    long_side_px = max(width_px, height_px)
    short_side_px = min(width_px, height_px)
    if short_side_px <= 0:
        raise ValueError("The selected crowd area has invalid dimensions.")
    return points, long_side_px, short_side_px, area

def _snap_to_edge(point, edge_map, radius=12):
    x, y = point
    height, width = edge_map.shape
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    ys, xs = np.where(edge_map[y0:y1, x0:x1] > 0)
    if not len(xs):
        return point
    xs += x0
    ys += y0
    distances = (xs - x) ** 2 + (ys - y) ** 2
    nearest = int(np.argmin(distances))
    if distances[nearest] <= radius ** 2:
        return int(xs[nearest]), int(ys[nearest])
    return point

def select_room_boundary(frame):
    """Trace a variable-shape fence polygon, snapping clicks to nearby edges."""
    window = "Trace crowd fence boundary"
    points = []
    selected = None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edge_map = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    edge_overlay = np.zeros_like(frame)
    edge_overlay[edge_map > 0] = (255, 120, 0)
    preview = cv2.addWeighted(frame, 0.9, edge_overlay, 0.35, 0)

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            snapped = _snap_to_edge((x, y), edge_map)
            if not points or np.linalg.norm(np.asarray(snapped) - points[-1]) >= 3:
                points.append(snapped)
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        display = preview.copy()
        instructions = "Click along fence/bends; points snap to edges. Enter=finish R=reset Right-click=undo Esc=cancel"
        cv2.rectangle(display, (0, 0), (display.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(display, instructions, (10, 23), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (255, 255, 255), 1)
        if len(points) > 1:
            cv2.polylines(display, [np.asarray(points, dtype=np.int32)],
                          len(points) >= 3,
                          (0, 255, 255), 2)
        for point in points:
            cv2.circle(display, point, 6, (0, 255, 255), -1)
        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("r"):
            points.clear()
        elif key in (10, 13):
            if len(points) >= 3:
                try:
                    selected = room_geometry(points)
                except ValueError as exc:
                    print(f"{exc} Add or remove points, then try again.")
                else:
                    break
        elif key == 27:
            cv2.destroyWindow(window)
            raise RuntimeError("Manual crowd-area selection was cancelled.")
    cv2.destroyWindow(window)
    return selected

def room_candidate(frame):
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    floor_mask = cv2.inRange(
        hsv, np.array([0, 60, 25], dtype=np.uint8),
        np.array([25, 255, 255], dtype=np.uint8)
    )
    floor_mask = cv2.morphologyEx(
        floor_mask, cv2.MORPH_OPEN, np.ones((5, 5), dtype=np.uint8)
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(floor_mask)
    center_label = labels[height // 2, width // 2]
    candidates = []
    if center_label:
        candidates.append(center_label)
    else:
        for label in range(1, count):
            area = stats[label, cv2.CC_STAT_AREA]
            center_x, center_y = stats[label, cv2.CC_STAT_LEFT] + stats[label, cv2.CC_STAT_WIDTH] / 2, \
                stats[label, cv2.CC_STAT_TOP] + stats[label, cv2.CC_STAT_HEIGHT] / 2
            if area > height * width * 0.2 and abs(center_x - width / 2) < width * 0.3 \
                    and abs(center_y - height / 2) < height * 0.35:
                candidates.append(label)
    if not candidates:
        return None

    label = max(candidates, key=lambda item: stats[item, cv2.CC_STAT_AREA])
    component = np.uint8(labels == label) * 255
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    for epsilon in (0.02, 0.03, 0.04, 0.05, 0.06):
        polygon = cv2.approxPolyDP(contour, epsilon * perimeter, True)
        if len(polygon) < 4 or len(polygon) > 100:
            continue
        corners = polygon.reshape(-1, 2)
        margin_x = max(10, int(width * 0.02))
        margin_y = max(10, int(height * 0.02))
        if (corners[:, 0].min() <= margin_x or corners[:, 0].max() >= width - margin_x
                or corners[:, 1].min() <= margin_y
                or corners[:, 1].max() >= height - max(30, int(height * 0.06))):
            continue
        area = abs(cv2.contourArea(corners))
        if area < height * width * 0.2:
            continue
        try:
            return room_geometry(corners)
        except ValueError:
            continue
    return None

def detect_room_boundary(cap, first_frame):
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    scan_frames = max(1, min(90, int(fps * 3)))
    best = None
    frame = first_frame
    for _ in range(scan_frames):
        if frame is None:
            ok, frame = cap.read()
            if not ok:
                break
            frame = resize_frame(frame)
        candidate = room_candidate(frame)
        if candidate is not None and (best is None or candidate[3] > best[3]):
            best = candidate
        frame = None
    if best is None:
        raise RuntimeError(
            "Could not automatically detect a rectangular room boundary."
        )
    return best

class PersonDetector:
    def __init__(self):
        self.model = None
        try:
            from ultralytics import YOLO
            self.model = YOLO("yolov8n.pt")
            print(f"Detector: YOLOv8n imgsz={YOLO_IMAGE_SIZE} CONF {CONF} + TOP-VIEW HEAD DETECTOR")
        except Exception as e:
            print(f"YOLO not available ({e}). Using HEAD detector only.")
            self.model=None

    def detect(self, frame):
        boxes=[]
        # 1) YOLO attempt
        if self.model is not None:
            try:
                res = self.model(
                    frame, classes=[0], conf=CONF, imgsz=YOLO_IMAGE_SIZE, verbose=False
                )[0]
                for b in res.boxes.xyxy.cpu().numpy():
                    boxes.append(tuple(int(v) for v in b))
            except Exception as exc:
                print(f"YOLO inference failed ({exc}); switching to the head detector.")
                self.model = None
        
        # 2) If YOLO fails (<3 people), use top-view head detection
        if len(boxes) < 5:
            # Detect small circular heads - blue/yellow caps in your video
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (7,7), 0)
            
            # Try HoughCircles for heads
            circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=12,
                                       param1=50, param2=18, minRadius=4, maxRadius=18)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for (x,y,r) in circles[0,:]:
                    # convert circle to box
                    x1,y1,x2,y2 = x-r*2, y-r*2, x+r*2, y+r*2
                    boxes.append((x1,y1,x2,y2))
            
            # Backup: blob/contour detection
            if len(boxes) < 10:
                _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                # Invert if needed
                if np.mean(thresh) > 127: thresh = cv2.bitwise_not(thresh)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if 15 < area < 500:  # head size
                        x,y,w,h = cv2.boundingRect(cnt)
                        if 5 < w < 40 and 5 < h < 40:
                            boxes.append((x,y,x+w,y+h))
        
        return boxes

def annotate_frame(frame, room_poly, detector, length, width, area,
                   per_person, capacity, detection_frame=None):
    inference_frame = detection_frame if detection_frame is not None else frame
    boxes = detector.detect(inference_frame)
    scale_x = frame.shape[1] / inference_frame.shape[1]
    scale_y = frame.shape[0] / inference_frame.shape[0]
    count = 0
    for (x1, y1, x2, y2) in boxes:
        x1 = max(0, min(frame.shape[1] - 1, int(round(x1 * scale_x))))
        y1 = max(0, min(frame.shape[0] - 1, int(round(y1 * scale_y))))
        x2 = max(0, min(frame.shape[1] - 1, int(round(x2 * scale_x))))
        y2 = max(0, min(frame.shape[0] - 1, int(round(y2 * scale_y))))
        foot = ((x1 + x2) // 2, y2)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        inside = (cv2.pointPolygonTest(room_poly, foot, False) >= 0
                  or cv2.pointPolygonTest(room_poly, center, False) >= 0)
        if inside:
            count += 1
        colour = (0, 255, 0) if inside else (150, 150, 150)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        cv2.circle(frame, center, 3, colour, -1)

    occupancy = count / capacity * 100
    remaining = max(0, capacity - count)
    status, status_colour = crowd_status(occupancy)
    cv2.polylines(frame, [room_poly], True, (0, 255, 0), 3)
    panel = [
        f"Room: {length:g} m x {width:g} m = {area:.1f} m2",
        f"Max capacity: {capacity} people ({per_person:g} m2 each)",
        f"People inside: {count}",
        f"Occupancy: {occupancy:.1f} %",
        f"Remaining: {remaining}"
    ]
    cv2.rectangle(frame, (0, 0), (430, 30 + 26 * (len(panel) + 1)), (0, 0, 0), -1)
    for i, line in enumerate(panel):
        cv2.putText(frame, line, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (255, 255, 255), 2)
    cv2.putText(frame, "Status: " + status, (10, 28 + 26 * len(panel)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_colour, 2)
    if occupancy >= 100:
        cv2.putText(frame, "ALERT: CAPACITY EXCEEDED!",
                    (frame.shape[1] // 2 - 230, frame.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    return count, occupancy

def main():
    path = choose_input()
    if not path:
        sys.exit("No input selected.")
    is_image = os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS
    cap = None
    if is_image:
        source_image = cv2.imread(path)
        if source_image is None:
            sys.exit("Could not open image: " + path)
        first = resize_frame(source_image)
    else:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            sys.exit("Could not open video: " + path)
        ok, first = cap.read()
        if not ok:
            cap.release()
            sys.exit("Video has no frames.")
        first = resize_frame(first)
    try:
        if is_image:
            detected = room_candidate(first)
            room_data = detected if detected is not None else select_room_boundary(first)
        else:
            try:
                room_data = detect_room_boundary(cap, first)
            except RuntimeError as exc:
                print(f"{exc} Trace the fence boundary manually; clicks snap to nearby edges.")
                room_data = select_room_boundary(first)
    except RuntimeError as e:
        if cap is not None:
            cap.release()
        sys.exit(str(e))
    room_poly, long_side_px, _, room_area_px = room_data
    known_long_side = ask_number("Known LONG room side in metres",10)
    length = known_long_side
    area = room_area_px * (length / long_side_px) ** 2
    width = area / length
    per_person=ask_number("Space needed per person (m2)",2)
    capacity=max(1,int(math.floor(area/per_person)))
    _write_dashboard_value(
        MAX_CAPACITY_FILE,
        json.dumps({"max_capacity": capacity})
    )
    _write_dashboard_value(LIVE_COUNT_FILE, "0\n")
    print("\n----- ROOM DETAILS -----")
    print(f"Length            : {length:.2f} m (known long side)\nWidth             : {width:.2f} m (estimated from video)\nRoom area         : {area:.1f} m2 (estimated)\nSpace per person  : {per_person} m2\nMaximum capacity  : {capacity} people\n")
    detector=PersonDetector()
    if is_image:
        count, _ = annotate_frame(first, room_poly, detector, length, width,
                                  area, per_person, capacity,
                                  detection_frame=source_image)
        _write_dashboard_value(LIVE_COUNT_FILE, f"{count}\n")
        image_output = os.path.splitext(path)[0] + "_phase1_output.jpg"
        if not cv2.imwrite(image_output, first):
            sys.exit("Could not save output image: " + image_output)
        print(f"Output image saved: {image_output}")
        cv2.imshow("Phase 1 - Image Occupancy", first)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(f"People counted   : {count}\nOccupancy        : {count / capacity * 100:.1f} %")
        return

    fps=cap.get(cv2.CAP_PROP_FPS) or 25
    frame_delay_ms=max(1, int(round(1000 / fps)))
    writer=None
    if SAVE_OUTPUT:
        h,w=first.shape[:2]
        writer=cv2.VideoWriter(OUTPUT_FILE,cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))
    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    peak=0; total=0; frames=0; paused=False
    win="Phase 1 - Room Size and Occupancy"
    while True:
        if not paused:
            ok,frame=cap.read()
            if not ok: break
            source_frame=frame
            frame=resize_frame(source_frame); frames+=1
            count, occupancy = annotate_frame(
                frame, room_poly, detector, length, width, area, per_person, capacity,
                detection_frame=source_frame
            )
            _write_dashboard_value(LIVE_COUNT_FILE, f"{count}\n")
            peak=max(peak,count); total+=count
            if writer is not None: writer.write(frame)
        cv2.imshow(win,frame)
        k=cv2.waitKey(frame_delay_ms if not paused else 50)&0xFF
        if k in (27,ord("q")): break
        if k==32: paused=not paused
    cap.release()
    if writer is not None: writer.release()
    cv2.destroyAllWindows()
    if frames:
        avg=total/frames
        print("----- SUMMARY -----")
        print(f"Frames processed  : {frames}\nPeak people count : {peak}\nAverage count     : {avg:.1f}\nPeak occupancy    : {peak/capacity*100:.1f} %\nAverage occupancy : {avg/capacity*100:.1f} %")
        if SAVE_OUTPUT: print(f"Output video saved: {OUTPUT_FILE}")

if __name__=="__main__": main()