import cv2
from ultralytics import YOLO
import time

# ── Configuration ──────────────────────────────────────────
MODEL_PATH = "../models/circuit_detector.pt"  # we'll put model here after training
CONFIDENCE = 0.5  # only show detections above 50% confidence

# Component info dictionary
COMPONENT_INFO = {
    "Battery":       "Stores electrical energy. Provides DC voltage.",
    "Capacitor":     "Stores charge. Filters noise in circuits.",
    "Diode":         "Allows current in one direction only.",
    "Fuse":          "Protects circuit from overcurrent.",
    "IC":            "Integrated Circuit. Complex logic on a chip.",
    "Inductor":      "Stores energy in magnetic field.",
    "LDR":           "Light Dependent Resistor. Resistance varies with light.",
    "LED":           "Light Emitting Diode. Emits light when current flows.",
    "Photodiode":    "Converts light into electrical current.",
    "Potentiometer": "Variable resistor. Adjusts voltage/current.",
    "Resistor":      "Limits current flow. Measured in Ohms.",
    "Speaker":       "Converts electrical signal to sound.",
    "Switch":        "Opens or closes a circuit.",
    "Thermistor":    "Resistance changes with temperature.",
    "Transformator": "Transfers energy between circuits via induction.",
    "Transistor":    "Amplifies or switches electronic signals.",
    "Trimpot":       "Small adjustable potentiometer.",
    "Varco":         "Variable capacitor. Tunes frequency.",
}

# ── Colors for bounding boxes (BGR format) ─────────────────
COLORS = [
    (255, 100, 100), (100, 255, 100), (100, 100, 255),
    (255, 255, 100), (255, 100, 255), (100, 255, 255),
    (255, 150, 50),  (50, 255, 150), (150, 50, 255),
    (200, 200, 50),  (50, 200, 200), (200, 50, 200),
    (255, 200, 150), (150, 255, 200), (200, 150, 255),
    (100, 200, 255), (255, 100, 200), (200, 255, 100),
]

def draw_detection(frame, box, label, confidence, color):
    """Draw bounding box and label on frame."""
    x1, y1, x2, y2 = map(int, box)

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Draw label background
    text = f"{label} {confidence:.0%}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)

    # Draw label text
    cv2.putText(frame, text, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame

def draw_info_panel(frame, detections):
    """Draw info panel on the right side of frame."""
    h, w = frame.shape[:2]
    panel_x = w - 300

    # Semi-transparent panel background
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, 0), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Title
    cv2.putText(frame, "DETECTED COMPONENTS", (panel_x + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1)
    cv2.line(frame, (panel_x + 10, 40), (w - 10, 40), (0, 255, 200), 1)

    # List unique detected components
    seen = set()
    y_offset = 60
    for label, conf in detections:
        if label not in seen and y_offset < h - 20:
            seen.add(label)
            # Component name
            cv2.putText(frame, f"• {label}", (panel_x + 10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
            # Component info (wrapped)
            info = COMPONENT_INFO.get(label, "Electronic component")
            words = info.split()
            line = ""
            for word in words:
                if len(line + word) < 30:
                    line += word + " "
                else:
                    cv2.putText(frame, line, (panel_x + 15, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
                    y_offset += 15
                    line = word + " "
            if line:
                cv2.putText(frame, line, (panel_x + 15, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
                y_offset += 20

    return frame

def main():
    print("🔍 Loading model...")
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded!")
    print("📷 Starting camera... Press 'Q' to quit")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    fps_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Camera error!")
            break

        # Run YOLO detection
        results = model(frame, conf=CONFIDENCE, verbose=False)

        detections = []
        for result in results:
            for box in result.boxes:
                label = model.names[int(box.cls)]
                conf = float(box.conf)
                color = COLORS[int(box.cls) % len(COLORS)]
                frame = draw_detection(frame, box.xyxy[0], label, conf, color)
                detections.append((label, conf))

        # Draw info panel
        if detections:
            frame = draw_info_panel(frame, detections)

        # FPS counter
        frame_count += 1
        if frame_count % 10 == 0:
            fps = 10 / (time.time() - fps_time)
            fps_time = time.time()
        cv2.putText(frame, f"FPS: {fps:.1f}" if frame_count > 10 else "FPS: --",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Component count
        cv2.putText(frame, f"Components: {len(set(d[0] for d in detections))}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("AR Circuit Diagnostics - Component Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Detection stopped.")

if __name__ == "__main__":
    main()