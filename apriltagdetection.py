import cv2
from pupil_apriltags import Detector
import math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime
import os

# Define which AprilTag ID corresponds to which Snap Circuit part (legacy for display)
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

# Enhanced component definition: Maps tag_id to a dict with name and terminal info.
component_db = {
    # Wires are simple connectors. They have two interchangeable terminals.
    0: {"name": "Wire", "terminals": {"A": {"polarity": "neutral"}, "B": {"polarity": "neutral"}}},

    # Battery has fixed polarity. Terminal A is positive (+), B is negative (-).
    10: {"name": "Battery", "terminals": {"A": {"polarity": "positive"}, "B": {"polarity": "negative"}}},

    # LED has polarity. The anode (A) must be more positive than the cathode (K).
    7: {"name": "LED", "terminals": {"A": {"polarity": "anode"}, "K": {"polarity": "cathode"}}},

    # Diode has polarity like LED.
    8: {"name": "Diode", "terminals": {"A": {"polarity": "anode"}, "K": {"polarity": "cathode"}}},
    9: {"name": "Anti-parallel Diode", "terminals": {"A": {"polarity": "anode"}, "K": {"polarity": "cathode"}}},

    # Resistor, Switch, Button, Capacitor are non-polarized. Terminals are interchangeable.
    3: {"name": "Resistor", "terminals": {"A": {"polarity": "neutral"}, "B": {"polarity": "neutral"}}},
    4: {"name": "Switch", "terminals": {"A": {"polarity": "neutral"}, "B": {"polarity": "neutral"}}},
    5: {"name": "Button", "terminals": {"A": {"polarity": "neutral"}, "B": {"polarity": "neutral"}}},
    6: {"name": "Capacitor", "terminals": {"A": {"polarity": "neutral"}, "B": {"polarity": "neutral"}}},

    # Speaker and Music are typically non-polarized for simple circuits.
    2: {"name": "Speaker", "terminals": {"A": {"polarity": "neutral"}, "B": {"polarity": "neutral"}}},
    1: {"name": "Music", "terminals": {"A": {"polarity": "neutral"}, "B": {"polarity": "neutral"}}},
}

# Connection distance threshold (pixels)
CONNECTION_THRESHOLD = 100

def analyze_circuit_logic(detections, connection_threshold=100):
    """
    Analyzes the logical layout of the circuit from AprilTag detections.
    Returns a dictionary with the netlist, errors, and warnings.
    """
    if not detections:
        return {"nets": {}, "errors": [], "warnings": ["No components detected."], "components": {}, "is_valid": True}

    # 1. Create Component Objects with Terminal Info
    components = {}
    for det in detections:
        tag_id = det.tag_id
        comp_info = component_db.get(tag_id, {"name": f"Unknown_{tag_id}", "terminals": {"A": {"polarity": "neutral"}, "B": {"polarity": "neutral"}}})
        components[tag_id] = {
            "type": comp_info["name"],
            "position": det.center,
            "terminals": {term: {"net_id": None} for term in comp_info["terminals"]}, # Initialize net_id to None
            "terminal_polarity": comp_info["terminals"] # Store the polarity info for validation later
        }

    # 2. Find Physical Connections & Assign Nets (Graph Connected Components)
    # A "Net" is a unique electrical node. All terminals connected by wires are on the same net.
    net_id_counter = 0
    nets = {} # Format: net_id: {"component_terminals": [(tag_id, terminal_name), ...]}

    # Helper function to propagate a net_id to all connected components
    def flood_fill_net(start_tag_id, current_net_id):
        stack = [start_tag_id]
        processed = set()
        
        while stack:
            current_tag_id = stack.pop()
            if current_tag_id in processed:
                continue
            processed.add(current_tag_id)
            
            current_comp = components[current_tag_id]

            # For each terminal in the current component, assign the net_id
            for term_name in current_comp["terminals"]:
                if current_comp["terminals"][term_name]["net_id"] is None:
                    current_comp["terminals"][term_name]["net_id"] = current_net_id
                    # Add this terminal to the net's list
                    nets[current_net_id]["component_terminals"].append( (current_tag_id, term_name) )

            # Now, find all physically connected neighbors and add them to the stack
            for other_tag_id, other_comp in components.items():
                if other_tag_id == current_tag_id or other_tag_id in processed:
                    continue # Skip self or already processed
                # Check proximity (your existing logic)
                dist = math.sqrt((current_comp['position'][0] - other_comp['position'][0])**2 +
                               (current_comp['position'][1] - other_comp['position'][1])**2)
                if dist < connection_threshold:
                    # If this neighbor hasn't been assigned a net, assign it and process it
                    if all(term_info["net_id"] is None for term_info in other_comp["terminals"].values()):
                        stack.append(other_tag_id)

    # Iterate through all components. If a component has no net, start a new net.
    for tag_id, comp in components.items():
        if all(term_info["net_id"] is None for term_info in comp["terminals"].values()):
            net_id_counter += 1
            nets[net_id_counter] = {"component_terminals": []}
            flood_fill_net(tag_id, net_id_counter)

    # 3. Validate Circuit Rules
    errors = []
    warnings = []

    # Check for polarity conflicts on each net
    for net_id, net_data in nets.items():
        polarities_on_net = {}
        # Collect all polarities present on this net
        for (tag_id, term_name) in net_data["component_terminals"]:
            comp_polarity = components[tag_id]["terminal_polarity"][term_name]["polarity"]
            comp_type = components[tag_id]["type"]

            # Map polarity to a "direction". Positive forces high voltage, negative forces low.
            polarity_value = None
            if comp_polarity in ["positive", "anode"]:
                polarity_value = +1
            elif comp_polarity in ["negative", "cathode"]:
                polarity_value = -1
            else: # neutral
                polarity_value = 0

            if polarity_value != 0:
                polarities_on_net[(tag_id, term_name)] = (comp_type, polarity_value)

        # Check if this net has conflicting driving polarities (e.g., two positives is ok, +1 and -1 is bad)
        unique_driving_forces = set(polarity for (_, (_, polarity)) in polarities_on_net.items())
        if +1 in unique_driving_forces and -1 in unique_driving_forces:
            conflict_details = []
            for (tag_id, term_name), (comp_type, pol) in polarities_on_net.items():
                if pol != 0:
                    conflict_details.append(f"{comp_type}(ID:{tag_id}-{term_name})")
            errors.append(f"Polarity Conflict on Net {net_id}: {' and '.join(conflict_details)} are shorted together.")

    # 4. Return the logical analysis results
    # Format the nets for easier reading in the output
    formatted_nets = {}
    for net_id, net_data in nets.items():
        formatted_nets[net_id] = [f"Comp{tag_id}.{term}" for (tag_id, term) in net_data["component_terminals"]]

    return {
        "components": components, # The enhanced component data
        "nets": formatted_nets,   # A simple list of what's on each net
        "errors": errors,
        "warnings": warnings,
        "is_valid": len(errors) == 0 # Circuit is logically valid if no errors
    }

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
        
        # Functional orientation validation (electrical requirements)
        is_valid_orientation = True
        
        if part_name in ["LED", "Diode"]:
            # These components have polarity - need to check if they're oriented 
            # for proper current flow in the circuit context
            # For now, we'll check if they're reasonably aligned (not sideways)
            # In a real implementation, we'd check actual terminal connections
            normalized_angle = abs(angle % 180)  # 0-180 range
            if normalized_angle > 90:
                normalized_angle = 180 - normalized_angle
            # Allow more flexibility - just not completely sideways
            is_valid_orientation = normalized_angle < 45  # Within 45 degrees of forward/backward
            
        elif part_name == "Battery":
            # Battery polarity matters, but orientation is more flexible
            # Main concern is + and - terminals connecting to right places
            # For basic validation, just ensure it's not at a weird angle
            normalized_angle = abs(angle % 180)
            if normalized_angle > 90:
                normalized_angle = 180 - normalized_angle
            is_valid_orientation = normalized_angle < 60  # Pretty flexible
            
        # Components where orientation doesn't matter functionally:
        # Switch, Button - can be flipped 180° and work the same
        # Wire, Music, Speaker, Resistor, Capacitor - no directionality
        
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
        
        # Functional orientation validation (same as main validation)
        is_valid_orientation = True
        
        if part_name in ["LED", "Diode"]:
            # Components with polarity - check they're not sideways
            normalized_angle = abs(angle % 180)
            if normalized_angle > 90:
                normalized_angle = 180 - normalized_angle
            is_valid_orientation = normalized_angle < 45
            
        elif part_name == "Battery":
            # Battery polarity matters but more flexible
            normalized_angle = abs(angle % 180)
            if normalized_angle > 90:
                normalized_angle = 180 - normalized_angle
            is_valid_orientation = normalized_angle < 60
        
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

        # Analyze the logical circuit for each side
        left_analysis = analyze_circuit_logic(detections_left, CONNECTION_THRESHOLD)
        right_analysis = analyze_circuit_logic(detections_right, CONNECTION_THRESHOLD)

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

        # Display logical validation results
        left_logic_color = (0, 255, 0) if left_analysis['is_valid'] else (0, 0, 255)
        right_logic_color = (0, 255, 0) if right_analysis['is_valid'] else (0, 0, 255)

        logic_text_left = f"Logic: {'VALID' if left_analysis['is_valid'] else 'INVALID'}"
        logic_text_right = f"Logic: {'VALID' if right_analysis['is_valid'] else 'INVALID'}"

        cv2.putText(frame, logic_text_left, (50, 160),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, left_logic_color, 2)
        cv2.putText(frame, logic_text_right, (mid_width + 50, 160),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, right_logic_color, 2)

        # Display error count
        cv2.putText(frame, f"Errors: {len(left_analysis['errors'])}", (50, 185),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, left_logic_color, 1)
        cv2.putText(frame, f"Errors: {len(right_analysis['errors'])}", (mid_width + 50, 185),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, right_logic_color, 1)

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

            print("\n=== LOGICAL ANALYSIS REPORT ===")
            print("LEFT CIRCUIT:")
            print(f"  Valid: {left_analysis['is_valid']}")
            print("  Nets:", left_analysis['nets'])
            if left_analysis['errors']:
                print("  ERRORS:")
                for error in left_analysis['errors']:
                    print(f"    ! {error}")
            if left_analysis['warnings']:
                print("  Warnings:")
                for warning in left_analysis['warnings']:
                    print(f"    ? {warning}")

            print("\nRIGHT CIRCUIT:")
            print(f"  Valid: {right_analysis['is_valid']}")
            print("  Nets:", right_analysis['nets'])
            if right_analysis['errors']:
                print("  ERRORS:")
                for error in right_analysis['errors']:
                    print(f"    ! {error}")
            if right_analysis['warnings']:
                print("  Warnings:")
                for warning in right_analysis['warnings']:
                    print(f"    ? {warning}")
            
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
