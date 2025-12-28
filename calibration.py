"""
Camera Calibration Module
Handles camera calibration using ArUco markers or chessboard patterns.
"""
import cv2
import numpy as np
import json
import pickle
import glob
import shutil
from pathlib import Path
from typing import Tuple, Optional, List


class CameraCalibrator:
    """Handles camera calibration operations."""
    
    def __init__(self, calibration_dir: str = "calibration_data"):
        self.calibration_dir = Path(calibration_dir)
        self.calibration_out = Path("calibration_output")
        self.calibration_dir.mkdir(exist_ok=True)
        self.images_dir = self.calibration_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
        self.calibration_out.mkdir(exist_ok=True)
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibration_file = self.calibration_dir / "calibration.json"
        
        # Load existing calibration if available
        self.load_calibration()
    
    def save_calibration_image(self, image: np.ndarray, image_name: str) -> bool:
        """Save a calibration image."""
        try:
            image_path = self.images_dir / image_name
            cv2.imwrite(str(image_path), image)
            return True
        except Exception as e:
            print(f"Error saving calibration image: {e}")
            return False
    
    def get_calibration_images(self) -> List[str]:
        """Get list of saved calibration image filenames."""
        if not self.images_dir.exists():
            return []
        jpg_files = list(self.images_dir.glob("*.jpg"))
        png_files = list(self.images_dir.glob("*.png"))
        return [f.name for f in jpg_files + png_files]
    
    def calibrate_with_aruco(
        self, 
        dictionary_type: int = cv2.aruco.DICT_6X6_250,
        marker_size: float = 0.05,
        square_size: float = 0.04
    ) -> Tuple[bool, Optional[float], str]:
        """
        Calibrate camera using ArUco markers.
        
        Args:
            dictionary_type: ArUco dictionary type
            marker_size: Size of ArUco markers in meters
            square_size: Size of squares in the ArUco board in meters
        
        Returns:
            Tuple of (success, reprojection_error, message)
        """
        # Load calibration images
        jpg_files = list(self.images_dir.glob("*.jpg"))
        png_files = list(self.images_dir.glob("*.png"))
        image_files = sorted(jpg_files + png_files)
        
        if len(image_files) < 3:
            return False, None, f"Need at least 3 calibration images. Found {len(image_files)}."
        
        # Create ArUco board for calibration
        aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_type)
        board = cv2.aruco.CharucoBoard(
            size=(7, 5),  # 7x5 board
            squareLength=square_size,
            markerLength=marker_size,
            dictionary=aruco_dict
        )
        
        all_corners = []
        all_ids = []
        image_points = []
        object_points = []
        
        detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
        charuco_detector = cv2.aruco.CharucoDetector(board)
        
        for img_file in image_files:
            image = cv2.imread(str(img_file))
            if image is None:
                continue
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect ArUco markers
            corners, ids, rejected = detector.detectMarkers(gray)
            
            if ids is not None and len(ids) > 0:
                # Detect Charuco board
                charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)
                
                if charuco_corners is not None and len(charuco_corners) > 0:
                    # Get object points for the board
                    obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
                    
                    if obj_points is not None and img_points is not None:
                        object_points.append(obj_points)
                        image_points.append(img_points)
        
        if len(object_points) < 3:
            return False, None, f"Need at least 3 valid calibration images. Found {len(object_points)}."
        
        # Perform calibration
        try:
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                object_points,
                image_points,
                gray.shape[::-1],
                None,
                None
            )
            
            # Calculate reprojection error
            total_error = 0
            for i in range(len(object_points)):
                img_points2, _ = cv2.projectPoints(
                    object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
                )
                error = cv2.norm(image_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
                total_error += error
            
            mean_error = total_error / len(object_points)
            
            # Save calibration
            self.camera_matrix = camera_matrix
            self.dist_coeffs = dist_coeffs
            self.save_calibration()
            
            return True, mean_error, f"Calibration successful! Mean reprojection error: {mean_error:.4f} pixels"
        
        except Exception as e:
            return False, None, f"Calibration failed: {str(e)}"
    
    def calibrate_with_chessboard(
        self,
        board_size: Tuple[int, int] = (10, 7),
        square_size: float = 0.015,
        create_output_folders: bool = True,
        save_pickle: bool = True,
        save_summary: bool = True,
        create_zip: bool = True
    ) -> Tuple[bool, Optional[float], str]:
        """
        Calibrate camera using chessboard pattern.
        
        Args:
            board_size: Number of inner corners (width, height) - default (10, 7) for 8x11 squares board
            square_size: Size of squares in meters - default 0.015 (15mm)
            create_output_folders: Create success/fail folders for organizing images
            save_pickle: Save calibration data as pickle files
            save_summary: Save summary.txt file with calibration results
            create_zip: Create zip archive of calibration output
        
        Returns:
            Tuple of (success, reprojection_error, message)
        """
        # Create output directories if requested
        success_dir = None
        fail_dir = None
        if create_output_folders:
            success_dir = self.calibration_out / "success"
            fail_dir = self.calibration_out / "fail"
            success_dir.mkdir(exist_ok=True)
            fail_dir.mkdir(exist_ok=True)
        
        # Load calibration images - support multiple formats
        images = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            images.extend(glob.glob(str(self.images_dir / ext)))
        image_files = sorted(images)
        
        print(f"Found {len(image_files)} images.")
        
        if len(image_files) < 3:
            return False, None, f"Need at least 3 calibration images. Found {len(image_files)}."
        
        # Prepare object points (3D points in real-world space)
        corners_x, corners_y = board_size
        objp = np.zeros((corners_y * corners_x, 3), np.float32)
        objp[:, :2] = np.mgrid[0:corners_x, 0:corners_y].T.reshape(-1, 2)
        objp *= square_size
        
        object_points = []  # 3D points
        image_points = []   # 2D points
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        valid = 0
        
        for fname in image_files:
            img = cv2.imread(fname)
            if img is None:
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find chessboard corners
            flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            ret, corners = cv2.findChessboardCorners(gray, board_size, flags)
            
            if ret:
                # Refine corners
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                object_points.append(objp)
                image_points.append(corners2)
                
                # Draw and save success visualization
                if create_output_folders and success_dir:
                    img_with_corners = img.copy()
                    cv2.drawChessboardCorners(img_with_corners, board_size, corners2, ret)
                    cv2.imwrite(str(success_dir / Path(fname).name), img_with_corners)
                
                valid += 1
            else:
                # Save failure image
                if create_output_folders and fail_dir:
                    cv2.imwrite(str(fail_dir / Path(fname).name), img)
        
        print(f"Detected chessboard in {valid}/{len(image_files)} images.")
        
        if len(object_points) < 3:
            return False, None, f"Need at least 3 valid calibration images. Found {len(object_points)}."
        
        # Perform calibration
        try:
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                object_points,
                image_points,
                gray.shape[::-1],
                None,
                None
            )
            
            # Calculate reprojection error
            total_error = 0
            for i in range(len(object_points)):
                img_points2, _ = cv2.projectPoints(
                    object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
                )
                error = cv2.norm(image_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
                total_error += error
            
            mean_error = total_error / len(object_points)
            
            # Save calibration
            self.camera_matrix = camera_matrix
            self.dist_coeffs = dist_coeffs
            self.save_calibration()
            
            # Save pickle files if requested
            if save_pickle:
                with open(self.calibration_out / "camera_matrix.pkl", "wb") as f:
                    pickle.dump(camera_matrix, f)
                with open(self.calibration_out / "dist_coeffs.pkl", "wb") as f:
                    pickle.dump(dist_coeffs, f)
            
            # Save summary.txt if requested
            if save_summary:
                summary_file = self.calibration_out / "summary.txt"
                with open(summary_file, 'w') as f:
                    f.write(f"Camera matrix:\n{camera_matrix}\n")
                    f.write(f"\nDistortion coefficients:\n{dist_coeffs}\n")
                    f.write(f"\nMean reprojection error:\n{mean_error}\n")
            
            # Create zip archive if requested
            if create_zip:
                assets_dir = Path("assets")
                assets_dir.mkdir(exist_ok=True)
                zip_path = assets_dir / f"{self.calibration_out.name}.zip"
                shutil.make_archive(
                    str(zip_path).replace('.zip', ''),
                    "zip",
                    str(self.calibration_out)
                )
            
            print("\nCalibration complete.")
            print("Camera matrix:\n", camera_matrix)
            print("\nDistortion coefficients:\n", dist_coeffs)
            print("\nMean reprojection error:", mean_error)
            
            return True, mean_error, f"Calibration successful! Mean reprojection error: {mean_error:.4f} pixels. Detected chessboard in {valid}/{len(image_files)} images."
        
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Calibration exception: {error_details}")
            return False, None, f"Calibration failed: {str(e)}"
    
    def save_calibration(self):
        """Save calibration data to JSON file."""
        if self.camera_matrix is None or self.dist_coeffs is None:
            return False
        
        calibration_data = {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist()
        }
        
        try:
            with open(self.calibration_file, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving calibration: {e}")
            return False
    
    def load_calibration(self) -> bool:
        """Load calibration data from JSON file."""
        if not self.calibration_file.exists():
            return False
        
        try:
            with open(self.calibration_file, 'r') as f:
                calibration_data = json.load(f)
            
            camera_matrix = np.array(calibration_data.get("camera_matrix"))
            dist_coeffs = np.array(calibration_data.get("dist_coeffs"))
            
            # Validate the loaded data before assigning
            if not self._validate_calibration_data(camera_matrix, dist_coeffs):
                print("Warning: Invalid calibration data found, ignoring it.")
                return False
            
            self.camera_matrix = camera_matrix
            self.dist_coeffs = dist_coeffs
            return True
        except Exception as e:
            print(f"Error loading calibration: {e}")
            # Reset to None on error
            self.camera_matrix = None
            self.dist_coeffs = None
            return False
    
    def _validate_calibration_data(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> bool:
        """Validate that calibration data is properly formatted and valid."""
        if camera_matrix is None or dist_coeffs is None:
            return False
        
        # Check camera matrix is 3x3
        if camera_matrix.shape != (3, 3):
            return False
        
        # Check that camera matrix values are reasonable (not all zeros)
        if np.allclose(camera_matrix, 0):
            return False
        
        # Check that focal length values are positive (diagonal elements)
        if camera_matrix[0, 0] <= 0 or camera_matrix[1, 1] <= 0:
            return False
        
        # Check dist_coeffs is a 1D array with at least 4 elements
        if len(dist_coeffs.shape) != 1 or len(dist_coeffs) < 4:
            return False
        
        return True
    
    def is_calibrated(self) -> bool:
        """Check if camera is calibrated with valid data."""
        if self.camera_matrix is None or self.dist_coeffs is None:
            return False
        
        # Validate the data is still valid
        return self._validate_calibration_data(self.camera_matrix, self.dist_coeffs)
    
    def get_calibration_info(self) -> dict:
        """Get calibration information."""
        if not self.is_calibrated():
            return {
                "calibrated": False,
                "message": "Camera not calibrated"
            }
        
        return {
            "calibrated": True,
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "focal_length": (float(self.camera_matrix[0, 0]), float(self.camera_matrix[1, 1])),
            "principal_point": (float(self.camera_matrix[0, 2]), float(self.camera_matrix[1, 2]))
        }
    
    def clear_calibration_images(self) -> bool:
        """Clear all calibration images."""
        try:
            for img_file in self.images_dir.glob("*"):
                if img_file.is_file():
                    img_file.unlink()
            return True
        except Exception as e:
            print(f"Error clearing images: {e}")
            return False

