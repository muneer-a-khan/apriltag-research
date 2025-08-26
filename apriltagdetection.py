import cv2
import apriltag

# Define which AprilTag ID corresponds to which Snap Circuit part
part_map = {
    0: "Battery Pack",
    1: "LED",
    2: "Switch",
    3: "Resistor",
    4: "Speaker",
}

def main():
    # Initialize video capture (0 = default  webcam)
    cap = cv2.VideoCapture(0)

    # Initialize the AprilTag detector
    detector = apriltag.Detector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale (AprilTag detection works best on grayscale images)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect AprilTags
        detections = detector.detect(gray)

        for detection in detections:
            tag_id = detection.tag_id
            center = tuple(map(int, detection.center))

            # Draw circle at tag center
            cv2.circle(frame, center, 5, (0, 255, 0), -1)

            # Draw bounding box
            corners = detection.corners.astype(int)
            for i in range(4):
                cv2.line(frame,
                         tuple(corners[i]),
                         tuple(corners[(i+1) % 4]),
                         (255, 0, 0), 2)

            # Look up the Snap Circuit part name
            part_name = part_map.get(tag_id, f"Unknown (ID={tag_id})")

            # Display part name
            cv2.putText(frame, part_name, (center[0] + 10, center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Snap Circuits Part Recognition", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
