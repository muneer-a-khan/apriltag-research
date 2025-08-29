import cv2
from pupil_apriltags import Detector
import math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime
import os

# Define which AprilTag ID corresponds to which Snap Circuit part
part_map = {
    0: "Wire",
    1: "Music", 
    2: "Speaker",
    3: "Resistor",
    4: "Switch",
    5: "Button",
    6: "Capacitor",
    7: "LED",
    8: "Diode",
    9: "Anti-parallel Diode",
    10: "Battery"
}

# Connection distance threshold (pixels)
CONNECTION_THRESHOLD = 100

def create_graph_visualization(detections_left, detections_right, frame_shape):
    """Create a graph visualization of the AprilTag connections"""
    # Create matplotlib figure with only two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Left circuit board graph
    create_single_graph(ax1, detections_left, "Left Circuit Board", (255, 0, 0))
    
    # Right circuit board graph  
    create_single_graph(ax2, detections_right, "Right Circuit Board", (0, 0, 255))
    
    plt.tight_layout()
    
    # Save the visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists("graph_outputs"):
        os.makedirs("graph_outputs")
    
    filename = f"graph_outputs/circuit_graph_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename

def create_single_graph(ax, detections, title, color):
    """Create graph for a single circuit board"""
    G = nx.Graph()
    positions = {}
    labels = {}
    
    # Calculate validation score
    validation_score, score_details = calculate_validation_score(detections)
    
    # Add nodes
    for detection in detections:
        tag_id = detection.tag_id
        center = detection.center
        part_name = part_map.get(tag_id, f"Unknown({tag_id})")
        
        G.add_node(tag_id)
        positions[tag_id] = (center[0], -center[1])  # Flip Y for matplotlib
        labels[tag_id] = f"{tag_id}\n{part_name}"
    
    # Add edges based on proximity
    for i, det1 in enumerate(detections):
        for j, det2 in enumerate(detections[i+1:], i+1):
            dist = math.sqrt((det1.center[0] - det2.center[0])**2 + 
                           (det1.center[1] - det2.center[1])**2)
            if dist < CONNECTION_THRESHOLD:
                G.add_edge(det1.tag_id, det2.tag_id, weight=dist)
    
    # Set up the title with validation score
    score_color = 'green' if validation_score >= 80 else 'orange' if validation_score >= 60 else 'red'
    main_title = f"{title}\nValidation Score: {validation_score:.1f}%"
    ax.set_title(main_title, fontsize=12, fontweight='bold')
    
    if G.nodes():
        nx.draw(G, positions, ax=ax, with_labels=False, 
                node_color='lightblue', node_size=500, 
                edge_color='gray', width=2)
        
        # Add custom labels
        for node, (x, y) in positions.items():
            ax.text(x, y, labels[node], ha='center', va='center', 
                   fontsize=8, bbox=dict(boxstyle='round,pad=0.3', 
                   facecolor='white', alpha=0.8))
    
    # Add score breakdown text
    if detections:
        breakdown_text = (f"Orientation: {score_details['orientation_score']:.1f}% "
                         f"({score_details['properly_oriented']}/{score_details['total_components']})\n"
                         f"Connections: {score_details['connection_score']:.1f}% "
                         f"({score_details['connected_components']}/{score_details['total_components']} connected)\n"
                         f"Total Links: {score_details['total_connections']}")
        
        ax.text(0.02, 0.98, breakdown_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
                facecolor=score_color, alpha=0.3))
    
    ax.set_aspect('equal')

def calculate_validation_score(detections):
    """Calculate validation score based on connections and orientations"""
    if not detections:
        return 0.0, {"orientation_score": 0.0, "connection_score": 0.0, "details": {}}
    
    # Calculate orientation score
    orientation_score = 0.0
    orientation_details = {}
    
    for detection in detections:
        tag_id = detection.tag_id
        angle = math.degrees(math.atan2(detection.corners[1][1] - detection.corners[0][1],
                                       detection.corners[1][0] - detection.corners[0][0]))
        part_name = part_map.get(tag_id, f"Unknown({tag_id})")
        
        # Validation rules for different component types
        is_valid_orientation = True
        if part_name in ["LED", "Diode", "Battery"]:
            # These components should be oriented horizontally or vertically
            normalized_angle = angle % 90
            if normalized_angle > 45:
                normalized_angle = 90 - normalized_angle
            is_valid_orientation = normalized_angle < 15  # Within 15 degrees of cardinal direction
        elif part_name in ["Switch", "Button"]:
            # Switches and buttons can have more flexible orientation
            normalized_angle = angle % 45
            if normalized_angle > 22.5:
                normalized_angle = 45 - normalized_angle
            is_valid_orientation = normalized_angle < 22.5  # Within 22.5 degrees
        # Wire, Music, Speaker, Resistor, Capacitor can be in any orientation
        
        orientation_details[tag_id] = {
            'part_name': part_name,
            'angle': angle,
            'valid': is_valid_orientation
        }
        
        if is_valid_orientation:
            orientation_score += 1
    
    orientation_percentage = (orientation_score / len(detections)) * 100 if detections else 0
    
    # Calculate connection score
    total_possible_connections = len(detections) * (len(detections) - 1) // 2
    actual_connections = 0
    connected_components = set()
    
    for i, det1 in enumerate(detections):
        for j, det2 in enumerate(detections[i+1:], i+1):
            dist = math.sqrt((det1.center[0] - det2.center[0])**2 + 
                           (det1.center[1] - det2.center[1])**2)
            if dist < CONNECTION_THRESHOLD:
                actual_connections += 1
                connected_components.add(det1.tag_id)
                connected_components.add(det2.tag_id)
    
    # Connection score based on how many components are connected to the circuit
    connection_percentage = (len(connected_components) / len(detections)) * 100 if detections else 0
    
    # Overall score is weighted average: 60% orientation, 40% connections
    overall_score = (orientation_percentage * 0.6) + (connection_percentage * 0.4)
    
    details = {
        "orientation_score": orientation_percentage,
        "connection_score": connection_percentage,
        "total_components": len(detections),
        "properly_oriented": int(orientation_score),
        "connected_components": len(connected_components),
        "total_connections": actual_connections,
        "component_details": orientation_details
    }
    
    return overall_score, details

def validate_connections(detections):
    """Validate circuit connections and orientations (legacy function for warnings)"""
    validation_results = []
    
    for detection in detections:
        tag_id = detection.tag_id
        center = detection.center
        angle = math.degrees(math.atan2(detection.corners[1][1] - detection.corners[0][1],
                                       detection.corners[1][0] - detection.corners[0][0]))
        
        part_name = part_map.get(tag_id, f"Unknown({tag_id})")
        
        # Simple validation rules
        is_valid_orientation = True
        if part_name in ["LED", "Diode", "Battery"]:
            # These components should be oriented horizontally or vertically
            normalized_angle = angle % 90
            if normalized_angle > 45:
                normalized_angle = 90 - normalized_angle
            is_valid_orientation = normalized_angle < 15  # Within 15 degrees of cardinal direction
        
        validation_results.append({
            'tag_id': tag_id,
            'part_name': part_name,
            'center': center,
            'angle': angle,
            'valid_orientation': is_valid_orientation
        })
    
    return validation_results

def process_detections(frame, gray, detector, side_name, x_offset=0):
    """Process AprilTag detections for one side"""
    detections = detector.detect(gray)
    
    for detection in detections:
        tag_id = detection.tag_id
        # Adjust center coordinates for the original frame
        center = (int(detection.center[0] + x_offset), int(detection.center[1]))
        
        # Calculate orientation
        corners = detection.corners.astype(int)
        # Adjust corner coordinates for the original frame
        corners[:, 0] += x_offset
        dx = corners[1][0] - corners[0][0]
        dy = corners[1][1] - corners[0][1]
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        
        print(f"[{side_name}] Detected tag ID: {tag_id} at {center} with orientation {angle_deg:.2f} degrees")
        
        # Draw circle at tag center
        cv2.circle(frame, center, 5, (0, 255, 0), -1)
        
        # Draw bounding box
        for i in range(4):
            cv2.line(frame,
                     tuple(corners[i]),
                     tuple(corners[(i+1) % 4]),
                     (255, 0, 0), 2)
        
        # Look up the Snap Circuit part name
        part_name = part_map.get(tag_id, f"Unknown (ID={tag_id})")
        
        # Display part name with side indicator
        cv2.putText(frame, f"{side_name}: {part_name}", (center[0] + 10, center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return detections

def main():
    # Initialize video capture (0 = default webcam)
    cap = cv2.VideoCapture(0)
    
    # Set camera resolution for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Initialize the AprilTag detector
    detector = Detector(
        families="tag36h11",
        nthreads=3,
        quad_decimate=3.0,
        quad_sigma=0.8,
        refine_edges=1,
        decode_sharpening=0.5,
        debug=0
    )
    
    frame_count = 0
    last_graph_save = 0

    print("Dual Circuit Board Detection Started")
    print("Left side: Red indicators | Right side: Blue indicators")
    print("Press 'q' to quit, 's' to save current graph, 'g' to generate graph")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        frame_count += 1
        height, width = frame.shape[:2]
        mid_width = width // 2

        # Split frame into left and right halves
        left_frame = frame[:, :mid_width]
        right_frame = frame[:, mid_width:]

        # Convert to grayscale for AprilTag detection
        left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

        # Process detections for both sides
        detections_left = process_detections(frame, left_gray, detector, "LEFT", 0)
        detections_right = process_detections(frame, right_gray, detector, "RIGHT", mid_width)

        # Draw dividing line
        cv2.line(frame, (mid_width, 0), (mid_width, height), (0, 255, 255), 3)
        
        # Add side labels
        cv2.putText(frame, "LEFT CIRCUIT", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "RIGHT CIRCUIT", (mid_width + 50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Calculate and display validation scores
        left_score, left_details = calculate_validation_score(detections_left)
        right_score, right_details = calculate_validation_score(detections_right)
        
        # Display validation scores with color coding
        left_color = (0, 255, 0) if left_score >= 80 else (0, 165, 255) if left_score >= 60 else (0, 0, 255)
        right_color = (0, 255, 0) if right_score >= 80 else (0, 165, 255) if right_score >= 60 else (0, 0, 255)
        
        cv2.putText(frame, f"Score: {left_score:.1f}%", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, left_color, 2)
        cv2.putText(frame, f"Score: {right_score:.1f}%", (mid_width + 50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, right_color, 2)
        
        # Display detailed breakdown
        cv2.putText(frame, f"Orient: {left_details['orientation_score']:.0f}% | Conn: {left_details['connection_score']:.0f}%", 
                   (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 1)
        cv2.putText(frame, f"Orient: {right_details['orientation_score']:.0f}% | Conn: {right_details['connection_score']:.0f}%", 
                   (mid_width + 50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 1)

        # Validate connections and show warnings
        left_validation = validate_connections(detections_left)
        right_validation = validate_connections(detections_right)
        
        display_validation_warnings(frame, left_validation, 0, "LEFT")
        display_validation_warnings(frame, right_validation, mid_width, "RIGHT")

        # Auto-save graph every 30 frames (approximately every second at 30fps)
        if frame_count - last_graph_save > 30 and (detections_left or detections_right):
            try:
                filename = create_graph_visualization(detections_left, detections_right, frame.shape)
                print(f"Auto-saved graph: {filename}")
                last_graph_save = frame_count
            except Exception as e:
                print(f"Error creating graph: {e}")

        # Display the frame
        cv2.imshow("Dual Circuit Board AprilTag Detection", frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Manual save
            try:
                filename = create_graph_visualization(detections_left, detections_right, frame.shape)
                print(f"Manually saved graph: {filename}")
            except Exception as e:
                print(f"Error creating graph: {e}")
        elif key == ord('g'):
            # Generate and display detailed validation report
            print("\n=== VALIDATION REPORT ===")
            
            print("LEFT CIRCUIT:")
            print(f"  Overall Score: {left_score:.1f}%")
            print(f"  Orientation Score: {left_details['orientation_score']:.1f}% ({left_details['properly_oriented']}/{left_details['total_components']} components)")
            print(f"  Connection Score: {left_details['connection_score']:.1f}% ({left_details['connected_components']}/{left_details['total_components']} connected)")
            print(f"  Total Connections: {left_details['total_connections']}")
            print("  Component Details:")
            for tag_id, details in left_details.get('component_details', {}).items():
                status = "✓" if details['valid'] else "✗"
                print(f"    {status} Tag {tag_id} ({details['part_name']}): {details['angle']:.1f}°")
            
            print("\nRIGHT CIRCUIT:")
            print(f"  Overall Score: {right_score:.1f}%")
            print(f"  Orientation Score: {right_details['orientation_score']:.1f}% ({right_details['properly_oriented']}/{right_details['total_components']} components)")
            print(f"  Connection Score: {right_details['connection_score']:.1f}% ({right_details['connected_components']}/{right_details['total_components']} connected)")
            print(f"  Total Connections: {right_details['total_connections']}")
            print("  Component Details:")
            for tag_id, details in right_details.get('component_details', {}).items():
                status = "✓" if details['valid'] else "✗"
                print(f"    {status} Tag {tag_id} ({details['part_name']}): {details['angle']:.1f}°")
            
            print("========================\n")

    cap.release()
    cv2.destroyAllWindows()

def count_connections(detections):
    """Count the number of connections between nearby tags"""
    connections = 0
    for i, det1 in enumerate(detections):
        for j, det2 in enumerate(detections[i+1:], i+1):
            dist = math.sqrt((det1.center[0] - det2.center[0])**2 + 
                           (det1.center[1] - det2.center[1])**2)
            if dist < CONNECTION_THRESHOLD:
                connections += 1
    return connections

def display_validation_warnings(frame, validation_results, x_offset, side_name):
    """Display validation warnings on the frame"""
    y_pos = 150
    for result in validation_results:
        if not result['valid_orientation']:
            warning_text = f"⚠ {result['part_name']} orientation!"
            cv2.putText(frame, warning_text, (x_offset + 50, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            y_pos += 25

if __name__ == "__main__":
    main()
