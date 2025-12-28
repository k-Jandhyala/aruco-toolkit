from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import cv2
import numpy as np
import base64
import pickle
from pathlib import Path
from calibration import CameraCalibrator
import uuid
from fastapi.staticfiles import StaticFiles


app = FastAPI(title="ArUco Marker Detection")
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

# Initialize calibrator
calibrator = CameraCalibrator()

# Test calibration storage (for uploaded .pkl files)
test_calibration_data = {
    "camera_matrix": None,
    "dist_coeffs": None,
    "marker_size": None
}

# Dictionary mapping
DICT_MAP = {
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


def detect_aruco_markers(image, dictionary_type=cv2.aruco.DICT_6X6_250, marker_size=None, use_pose=False):
    """
    Detect ArUco markers in an image.
    
    Args:
        image: Input image (numpy array)
        dictionary_type: ArUco dictionary type (default: DICT_6X6_250)
        marker_size: Size of markers in meters (for pose estimation)
        use_pose: Whether to estimate pose (requires calibration)
    
    Returns:
        corners: Detected marker corners
        ids: Detected marker IDs
        image_with_markers: Image with markers drawn
        marker_ids: List of marker IDs
        poses: List of pose data (rvecs, tvecs) if use_pose is True
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
    marker_ids = []
    poses = []
    
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(image_with_markers, corners, ids)
        
        # Estimate pose if calibration is available and marker size is provided
        if use_pose and calibrator.is_calibrated() and marker_size is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, marker_size, 
                calibrator.camera_matrix, 
                calibrator.dist_coeffs
            )
            
            # Draw pose axes
            for i in range(len(ids)):
                cv2.drawFrameAxes(
                    image_with_markers,
                    calibrator.camera_matrix,
                    calibrator.dist_coeffs,
                    rvecs[i],
                    tvecs[i],
                    marker_size * 0.5
                )
                poses.append({
                    "rvec": rvecs[i].tolist(),
                    "tvec": tvecs[i].tolist()
                })
        
        # Extract marker IDs and draw text
        for i, marker_id in enumerate(ids.flatten()):
            marker_ids.append(int(marker_id))
            # Calculate center of marker
            center = np.mean(corners[i][0], axis=0).astype(int)
            cv2.putText(image_with_markers, f"ID: {marker_id}", 
                       (center[0] - 20, center[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return corners, ids, image_with_markers, marker_ids, poses


def image_to_base64(image):
    """Convert OpenCV image to base64 string for web display."""
    _, buffer = cv2.imencode('.png', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page."""
    html_file = Path("templates/index.html")
    if html_file.exists():
        return FileResponse(html_file)
    else:
        return HTMLResponse(content="<h1>Error: Template file not found</h1>", status_code=404)


@app.get("/style.css")
async def get_style():
    """Serve the CSS file."""
    css_file = Path("templates/style.css")
    if css_file.exists():
        return FileResponse(css_file, media_type="text/css")
    else:
        return JSONResponse(status_code=404, content={"error": "CSS file not found"})


@app.get("/script.js")
async def get_script():
    """Serve the JavaScript file."""
    js_file = Path("templates/script.js")
    if js_file.exists():
        return FileResponse(js_file, media_type="application/javascript")
    else:
        return JSONResponse(status_code=404, content={"error": "JavaScript file not found"})


@app.post("/detect")
async def detect_markers(
    file: UploadFile = File(...),
    dictionary: str = Form(default="DICT_6X6_250"),
    marker_size: float = Form(default=None),
    use_pose: bool = Form(default=False)
):
    """
    Detect ArUco markers in uploaded image.
    """
    try:
        # Read uploaded file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Could not decode image. Please upload a valid image file."}
            )
        
        # Get dictionary type
        dict_type = DICT_MAP.get(dictionary, cv2.aruco.DICT_6X6_250)
        
        # Use pose estimation if calibration is available and marker size is provided
        use_pose_estimation = use_pose and calibrator.is_calibrated() and marker_size is not None
        
        # Detect markers
        corners, ids, image_with_markers, marker_ids, poses = detect_aruco_markers(
            image, dict_type, marker_size if marker_size else None, use_pose_estimation
        )
        
        # Convert result image to base64
        result_image_base64 = image_to_base64(image_with_markers)
        
        # Prepare response
        response_data = {
            "success": True,
            "marker_count": len(marker_ids),
            "marker_ids": marker_ids,
            "result_image": result_image_base64,
            "dictionary": dictionary,
            "calibrated": calibrator.is_calibrated(),
            "poses": poses if use_pose_estimation else []
        }
        
        return JSONResponse(content=response_data)
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing image: {str(e)}"}
        )


@app.post("/calibration/capture")
async def capture_calibration_image(file: UploadFile = File(...)):
    """Capture and save a calibration image."""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Could not decode image."}
            )
        
        # Generate unique filename
        image_name = f"calibration_{uuid.uuid4().hex[:8]}.jpg"
        
        # Save calibration image
        success = calibrator.save_calibration_image(image, image_name)
        
        if success:
            image_count = len(calibrator.get_calibration_images())
            return JSONResponse(content={
                "success": True,
                "message": f"Calibration image saved. Total images: {image_count}",
                "image_count": image_count
            })
        else:
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to save calibration image"}
            )
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error capturing calibration image: {str(e)}"}
        )


@app.post("/calibration/calibrate")
async def perform_calibration(
    method: str = Form(default="aruco"),
    dictionary: str = Form(default="DICT_6X6_250"),
    marker_size: float = Form(default=0.05),
    square_size: float = Form(default=0.04)
):
    """Perform camera calibration."""
    try:
        if method == "aruco":
            dict_type = DICT_MAP.get(dictionary, cv2.aruco.DICT_6X6_250)
            success, error, message = calibrator.calibrate_with_aruco(
                dict_type, marker_size, square_size
            )
        else:
            # Chessboard calibration
            success, error, message = calibrator.calibrate_with_chessboard(
                board_size=(10, 7), square_size=square_size
            )
        
        if success:
            calib_info = calibrator.get_calibration_info()
            return JSONResponse(content={
                "success": True,
                "message": message,
                "reprojection_error": error,
                "calibration_info": calib_info
            })
        else:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": message}
            )
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error during calibration: {str(e)}"}
        )


@app.get("/calibration/status")
async def get_calibration_status():
    """Get calibration status and information."""
    return JSONResponse(content={
        "calibrated": calibrator.is_calibrated(),
        "image_count": len(calibrator.get_calibration_images()),
        "calibration_info": calibrator.get_calibration_info()
    })


@app.delete("/calibration/images")
async def clear_calibration_images():
    """Clear all calibration images."""
    success = calibrator.clear_calibration_images()
    if success:
        return JSONResponse(content={
            "success": True,
            "message": "All calibration images cleared"
        })
    else:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to clear calibration images"}
        )


@app.post("/calibration/upload-single")
async def upload_single_calibration_file(
    file: UploadFile = File(...),
    file_type: str = Form(...)
):
    """Upload a single calibration file (.pkl) for testing."""
    try:
        contents = await file.read()
        data = pickle.loads(contents)
        
        if file_type == "camera_matrix":
            test_calibration_data["camera_matrix"] = np.array(data)
            return JSONResponse(content={
                "success": True,
                "message": "Camera matrix uploaded successfully"
            })
        elif file_type == "dist_coeffs":
            test_calibration_data["dist_coeffs"] = np.array(data)
            return JSONResponse(content={
                "success": True,
                "message": "Distortion coefficients uploaded successfully"
            })
        else:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Invalid file_type. Must be 'camera_matrix' or 'dist_coeffs'"}
            )
    
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": f"Error loading calibration file: {str(e)}"}
        )


@app.post("/calibration/save-marker-size")
async def save_marker_size(
    marker_size: float = Form(...)
):
    """Save marker size for distance calculation."""
    try:
        if marker_size <= 0:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Marker size must be greater than 0"}
            )
        
        test_calibration_data["marker_size"] = marker_size
        
        return JSONResponse(content={
            "success": True,
            "message": f"Marker size saved: {marker_size} meters"
        })
    
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": f"Error saving marker size: {str(e)}"}
        )


@app.post("/detect/distance")
async def detect_markers_with_distance(
    file: UploadFile = File(...),
    dictionary: str = Form(default="DICT_6X6_250")
):
    """Detect ArUco markers and calculate distance using uploaded calibration."""
    try:
        # Check if calibration data is loaded
        if (test_calibration_data["camera_matrix"] is None or 
            test_calibration_data["dist_coeffs"] is None or
            test_calibration_data["marker_size"] is None):
            return JSONResponse(
                status_code=400,
                content={"error": "Calibration files not uploaded. Please upload calibration files first."}
            )
        
        # Read uploaded file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Could not decode image. Please upload a valid image file."}
            )
        
        # Get dictionary type
        dict_type = DICT_MAP.get(dictionary, cv2.aruco.DICT_6X6_250)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Get the ArUco dictionary
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        
        # Detect markers
        corners, ids, rejected = detector.detectMarkers(gray)
        
        # Draw markers on image
        image_with_markers = image.copy()
        markers_info = []
        
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(image_with_markers, corners, ids)
            
            # Get calibration data
            camera_matrix = test_calibration_data["camera_matrix"]
            dist_coeffs = test_calibration_data["dist_coeffs"]
            marker_size = test_calibration_data["marker_size"]
            
            # Estimate pose
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, marker_size, camera_matrix, dist_coeffs
            )
            
            # Draw pose axes and calculate distance
            for i in range(len(ids)):
                rvec = rvecs[i]
                tvec = tvecs[i]
                
                # Calculate distance (magnitude of translation vector)
                distance = np.linalg.norm(tvec)
                
                # Draw axes
                cv2.drawFrameAxes(
                    image_with_markers,
                    camera_matrix,
                    dist_coeffs,
                    rvec,
                    tvec,
                    marker_size * 0.5
                )
                
                # Draw distance text
                marker_id = int(ids[i][0])
                center = np.mean(corners[i][0], axis=0).astype(int)
                distance_text = f"ID:{marker_id} {distance*100:.1f}cm"
                cv2.putText(
                    image_with_markers, 
                    distance_text,
                    (center[0] - 40, center[1] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (0, 255, 0), 
                    2
                )
                
                markers_info.append({
                    "id": marker_id,
                    "distance": float(distance),
                    "tvec": tvec.tolist()
                })
        
        # Convert result image to base64
        result_image_base64 = image_to_base64(image_with_markers)
        
        # Prepare response
        response_data = {
            "success": True,
            "marker_count": len(markers_info),
            "markers": markers_info,
            "result_image": result_image_base64,
            "dictionary": dictionary
        }
        
        return JSONResponse(content=response_data)
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing image: {str(e)}"}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

