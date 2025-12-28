import cv2
import numpy as np
import argparse
import sys

def detect_aruco_markers(image, dictionary_type=cv2.aruco.DICT_6X6_250):
    """
    Detect ArUco markers in an image.
    
    Args:
        image: Input image (numpy array)
        dictionary_type: ArUco dictionary type (default: DICT_6X6_250)
    
    Returns:
        corners: Detected marker corners
        ids: Detected marker IDs
        image_with_markers: Image with markers drawn
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Get the ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_type)
    
    # Create detector parameters
    parameters = cv2.aruco.DetectorParameters()
    
    # Detect markers
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected = detector.detectMarkers(gray)
    
    # Draw detected markers on the image
    image_with_markers = image.copy()
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(image_with_markers, corners, ids)
        
        # Print detected marker IDs
        print(f"Detected {len(ids)} ArUco marker(s):")
        for i, marker_id in enumerate(ids.flatten()):
            print(f"  Marker ID: {marker_id}")
            # Calculate center of marker
            center = np.mean(corners[i][0], axis=0).astype(int)
            cv2.putText(image_with_markers, f"ID: {marker_id}", 
                       (center[0] - 20, center[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        print("No ArUco markers detected.")
    
    return corners, ids, image_with_markers


def detect_from_camera(dictionary_type=cv2.aruco.DICT_6X6_250):
    """
    Detect ArUco markers from webcam feed.
    """
    
    for camera_index in [0, 1, 2]:
        print(f"Attempting to open camera index {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            print("Press 'q' to quit, 's' to save current frame")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame.")
                    break
                
                # Detect markers
                corners, ids, frame_with_markers = detect_aruco_markers(frame, dictionary_type)
                
                # Display the frame
                cv2.imshow('ArUco Detection', frame_with_markers)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    return
                elif key == ord('s'):
                    # Save current frame
                    filename = 'aruco_detection.png'
                    cv2.imwrite(filename, frame_with_markers)
                    print(f"Frame saved as {filename}")
            
            cap.release()
            cv2.destroyAllWindows()
        else:
            print(f"Could not open any camera_index{camera_index}")


def detect_from_image(image_path, dictionary_type=cv2.aruco.DICT_6X6_250):
    """
    Detect ArUco markers from an image file.
    """
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    print(f"Processing image: {image_path}")
    corners, ids, image_with_markers = detect_aruco_markers(image, dictionary_type)
    
    # Display the result
    cv2.imshow('ArUco Detection', image_with_markers)
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save the result
    output_path = image_path.replace('.', '_aruco.')
    cv2.imwrite(output_path, image_with_markers)
    print(f"Result saved as {output_path}")


def main():
    parser = argparse.ArgumentParser(description='ArUco Marker Detection')
    parser.add_argument('--image', type=str, help='Path to input image file')
    parser.add_argument('--camera', action='store_true', help='Use webcam for detection')
    parser.add_argument('--dictionary', type=str, default='DICT_6X6_250',
                       choices=['DICT_4X4_50', 'DICT_4X4_100', 'DICT_4X4_250', 'DICT_4X4_1000',
                               'DICT_5X5_50', 'DICT_5X5_100', 'DICT_5X5_250', 'DICT_5X5_1000',
                               'DICT_6X6_50', 'DICT_6X6_100', 'DICT_6X6_250', 'DICT_6X6_1000',
                               'DICT_7X7_50', 'DICT_7X7_100', 'DICT_7X7_250', 'DICT_7X7_1000'],
                       help='ArUco dictionary type')
    
    args = parser.parse_args()
    
    # Map dictionary string to cv2 constant
    dict_map = {
        'DICT_4X4_50': cv2.aruco.DICT_4X4_50,
        'DICT_4X4_100': cv2.aruco.DICT_4X4_100,
        'DICT_4X4_250': cv2.aruco.DICT_4X4_250,
        'DICT_4X4_1000': cv2.aruco.DICT_4X4_1000,
        'DICT_5X5_50': cv2.aruco.DICT_5X5_50,
        'DICT_5X5_100': cv2.aruco.DICT_5X5_100,
        'DICT_5X5_250': cv2.aruco.DICT_5X5_250,
        'DICT_5X5_1000': cv2.aruco.DICT_5X5_1000,
        'DICT_6X6_50': cv2.aruco.DICT_6X6_50,
        'DICT_6X6_100': cv2.aruco.DICT_6X6_100,
        'DICT_6X6_250': cv2.aruco.DICT_6X6_250,
        'DICT_6X6_1000': cv2.aruco.DICT_6X6_1000,
        'DICT_7X7_50': cv2.aruco.DICT_7X7_50,
        'DICT_7X7_100': cv2.aruco.DICT_7X7_100,
        'DICT_7X7_250': cv2.aruco.DICT_7X7_250,
        'DICT_7X7_1000': cv2.aruco.DICT_7X7_1000,
    }
    
    dictionary_type = dict_map[args.dictionary]
    
    if args.camera:
        detect_from_camera(dictionary_type)
    elif args.image:
        detect_from_image(args.image, dictionary_type)
    else:
        print("Please specify either --image <path> or --camera")
        print("\nUsage examples:")
        print("  python main.py --camera")
        print("  python main.py --image path/to/image.jpg")
        print("  python main.py --image path/to/image.jpg --dictionary DICT_4X4_50")
        sys.exit(1)


if __name__ == "__main__":
    main()
