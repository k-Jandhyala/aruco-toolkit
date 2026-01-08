const fileInput = document.getElementById('fileInput');
const uploadSection = document.getElementById('uploadSection');
const webcamSection = document.getElementById('webcamSection');
const detectButton = document.getElementById('detectButton');
const dictionarySelect = document.getElementById('dictionarySelect');
const loading = document.getElementById('loading');
const error = document.getElementById('error');
const resultsSection = document.getElementById('resultsSection');
const markerInfo = document.getElementById('markerInfo');
const resultImage = document.getElementById('resultImage');
const webcamVideo = document.getElementById('webcamVideo');
const webcamCanvas = document.getElementById('webcamCanvas');
const webcamPlaceholder = document.getElementById('webcamPlaceholder');
const startWebcamBtn = document.getElementById('startWebcamBtn');
const stopWebcamBtn = document.getElementById('stopWebcamBtn');
const captureBtn = document.getElementById('captureBtn');
const downloadCalibrationPatternBtn = document.getElementById('downloadCalibrationPatternBtn');
const downloadCalibrationResultsBtn = document.getElementById('downloadCalibrationResultsBtn');

let selectedFile = null;
let stream = null;
let calibrationStream = null;
let currentMode = 'upload';

// File input change
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        selectedFile = e.target.files[0];
        detectButton.disabled = false;
        hideError();
    }
});

// Drag and drop
uploadSection.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadSection.classList.add('dragover');
});

uploadSection.addEventListener('dragleave', () => {
    uploadSection.classList.remove('dragover');
});

uploadSection.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadSection.classList.remove('dragover');
    
    if (e.dataTransfer.files.length > 0) {
        selectedFile = e.dataTransfer.files[0];
        fileInput.files = e.dataTransfer.files;
        detectButton.disabled = false;
        hideError();
    }
});

// Click on upload section to trigger file input
uploadSection.addEventListener('click', (e) => {
    if (e.target !== fileInput) {
        fileInput.click();
    }
});

// Detect button click (for file upload mode)
detectButton.addEventListener('click', async () => {
    if (!selectedFile || currentMode !== 'upload') return;

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('dictionary', dictionarySelect.value);

    // Show loading, hide results and error
    loading.classList.add('show');
    resultsSection.classList.remove('show');
    hideError();
    detectButton.disabled = true;

    try {
        const response = await fetch('/detect', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Failed to process image');
        }

        // Display results
        displayResults(data);

    } catch (err) {
        showError(err.message);
    } finally {
        loading.classList.remove('show');
        detectButton.disabled = false;
    }
});

function displayResults(data) {
    // Display marker information
    if (data.marker_count > 0) {
        markerInfo.innerHTML = `
            <h2>✅ Detected ${data.marker_count} Marker(s)</h2>
            <p>Dictionary: ${data.dictionary}</p>
            <div class="marker-ids">
                ${data.marker_ids.map(id => 
                    `<span class="marker-id-badge">ID: ${id}</span>`
                ).join('')}
            </div>
        `;
    } else {
        markerInfo.innerHTML = `
            <div class="no-markers">
                <h2>⚠️ No Markers Detected</h2>
                <p>Try a different image or dictionary type.</p>
            </div>
        `;
    }

    // Display result image
    resultImage.src = `data:image/png;base64,${data.result_image}`;
    resultsSection.classList.add('show');
}

function showError(message, type = 'error') {
    if (type === 'success') {
        error.style.background = '#4caf50';
        error.textContent = message;
        error.classList.add('show');
        setTimeout(() => {
            error.classList.remove('show');
        }, 5000);
    } else {
        error.style.background = '#ff6b6b';
    error.textContent = `Error: ${message}`;
    error.classList.add('show');
    }
}

function hideError() {
    error.classList.remove('show');
}

// Mode switching
function switchMode(mode) {
    currentMode = mode;
    const uploadTab = document.getElementById('uploadTab');
    const webcamTab = document.getElementById('webcamTab');
    const calibrationTab = document.getElementById('calibrationTab');
    const testCalibrationTab = document.getElementById('testCalibrationTab');
    const calibrationSection = document.getElementById('calibrationSection');
    const testCalibrationSection = document.getElementById('testCalibrationSection');

    if (mode === 'upload') {
        uploadTab.classList.add('active');
        webcamTab.classList.remove('active');
        calibrationTab.classList.remove('active');
        testCalibrationTab.classList.remove('active');
        uploadSection.style.display = 'block';
        webcamSection.style.display = 'none';
        calibrationSection.style.display = 'none';
        testCalibrationSection.style.display = 'none';
        detectButton.style.display = 'block';
        // Stop webcams if running
        if (stream) {
            stopWebcam();
        }
        if (calibrationStream) {
            stopCalibrationWebcam();
        }
        if (testCalibrationStream) {
            stopTestCalibrationWebcam();
        }
    } else if (mode === 'webcam') {
        uploadTab.classList.remove('active');
        webcamTab.classList.add('active');
        calibrationTab.classList.remove('active');
        testCalibrationTab.classList.remove('active');
        uploadSection.style.display = 'none';
        webcamSection.style.display = 'block';
        calibrationSection.style.display = 'none';
        testCalibrationSection.style.display = 'none';
        detectButton.style.display = 'none';
        // Stop calibration webcams if running
        if (calibrationStream) {
            stopCalibrationWebcam();
        }
        if (testCalibrationStream) {
            stopTestCalibrationWebcam();
        }
    } else if (mode === 'calibration') {
        uploadTab.classList.remove('active');
        webcamTab.classList.remove('active');
        calibrationTab.classList.add('active');
        testCalibrationTab.classList.remove('active');
        uploadSection.style.display = 'none';
        webcamSection.style.display = 'none';
        calibrationSection.style.display = 'block';
        testCalibrationSection.style.display = 'none';
        detectButton.style.display = 'none';
        // Stop main webcam if running
        if (stream) {
            stopWebcam();
        }
        if (testCalibrationStream) {
            stopTestCalibrationWebcam();
        }
        // Load calibration status
        loadCalibrationStatus();
    } else if (mode === 'testCalibration') {
        uploadTab.classList.remove('active');
        webcamTab.classList.remove('active');
        calibrationTab.classList.remove('active');
        testCalibrationTab.classList.add('active');
        uploadSection.style.display = 'none';
        webcamSection.style.display = 'none';
        calibrationSection.style.display = 'none';
        testCalibrationSection.style.display = 'block';
        detectButton.style.display = 'none';
        // Stop other webcams if running
        if (stream) {
            stopWebcam();
        }
        if (calibrationStream) {
            stopCalibrationWebcam();
        }
    }
}

// Webcam functions
async function startWebcam() {
    try {
        // Check if getUserMedia is available
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error('getUserMedia is not supported in this browser. Please use a modern browser or ensure you are accessing the site via HTTPS (or localhost).');
        }
        
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 1280 },
                height: { ideal: 720 }
            } 
        });
        webcamVideo.srcObject = stream;
        webcamVideo.style.display = 'block';
        webcamPlaceholder.style.display = 'none';
        startWebcamBtn.style.display = 'none';
        stopWebcamBtn.style.display = 'inline-block';
        captureBtn.style.display = 'inline-block';
        hideError();
    } catch (err) {
        showError('Could not access webcam: ' + err.message);
        console.error('Error accessing webcam:', err);
    }
}

function stopWebcam() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    webcamVideo.srcObject = null;
    webcamVideo.style.display = 'none';
    webcamPlaceholder.style.display = 'flex';
    startWebcamBtn.style.display = 'inline-block';
    stopWebcamBtn.style.display = 'none';
    captureBtn.style.display = 'none';
}

async function captureAndDetect() {
    if (!stream || webcamVideo.readyState !== webcamVideo.HAVE_ENOUGH_DATA) {
        showError('Webcam not ready. Please wait a moment.');
        return;
    }

    // Set canvas dimensions to match video
    webcamCanvas.width = webcamVideo.videoWidth;
    webcamCanvas.height = webcamVideo.videoHeight;

    // Draw current frame to canvas
    const ctx = webcamCanvas.getContext('2d');
    ctx.drawImage(webcamVideo, 0, 0);

    // Convert canvas to blob
    webcamCanvas.toBlob(async (blob) => {
        if (!blob) {
            showError('Failed to capture frame');
            return;
        }

        // Create a File object from the blob
        const file = new File([blob], 'webcam-capture.png', { type: 'image/png' });
        
        // Process the captured image
        const formData = new FormData();
        formData.append('file', file);
        formData.append('dictionary', dictionarySelect.value);

        // Show loading, hide results and error
        loading.classList.add('show');
        resultsSection.classList.remove('show');
        hideError();
        captureBtn.disabled = true;

        try {
            const response = await fetch('/detect', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Failed to process image');
            }

            // Display results
            displayResults(data);

        } catch (err) {
            showError(err.message);
        } finally {
            loading.classList.remove('show');
            captureBtn.disabled = false;
        }
    }, 'image/png');
}

// Event listeners for webcam buttons
startWebcamBtn.addEventListener('click', startWebcam);
stopWebcamBtn.addEventListener('click', stopWebcam);
captureBtn.addEventListener('click', captureAndDetect);

// Test Calibration elements (check if they exist)
const cameraMatrixFile = document.getElementById('cameraMatrixFile');
const distCoeffsFile = document.getElementById('distCoeffsFile');
const testMarkerSize = document.getElementById('testMarkerSize');
const uploadCameraMatrixBtn = document.getElementById('uploadCameraMatrixBtn');
const uploadDistCoeffsBtn = document.getElementById('uploadDistCoeffsBtn');
const saveMarkerSizeBtn = document.getElementById('saveMarkerSizeBtn');
const calibrationUploadStatus = document.getElementById('calibrationUploadStatus');
const cameraMatrixStatus = document.getElementById('cameraMatrixStatus');
const distCoeffsStatus = document.getElementById('distCoeffsStatus');
const markerSizeStatus = document.getElementById('markerSizeStatus');

const testCalibrationWebcamVideo = document.getElementById('testCalibrationWebcamVideo');
const testCalibrationWebcamCanvas = document.getElementById('testCalibrationWebcamCanvas');
const testCalibrationWebcamPlaceholder = document.getElementById('testCalibrationWebcamPlaceholder');
const startTestCalibrationWebcamBtn = document.getElementById('startTestCalibrationWebcamBtn');
const stopTestCalibrationWebcamBtn = document.getElementById('stopTestCalibrationWebcamBtn');
const distanceInfo = document.getElementById('distanceInfo');
const distanceDetails = document.getElementById('distanceDetails');

let testCalibrationStream = null;
let testCalibrationInterval = null;
let testCalibrationLoaded = {
    cameraMatrix: false,
    distCoeffs: false,
    markerSize: false
};

// Check if all calibration data is loaded
function checkCalibrationReady() {
    if (testCalibrationLoaded.cameraMatrix && testCalibrationLoaded.distCoeffs && testCalibrationLoaded.markerSize) {
        startTestCalibrationWebcamBtn.disabled = false;
        calibrationUploadStatus.textContent = '✅ All calibration data loaded! You can start the webcam now.';
        calibrationUploadStatus.style.color = '#28a745';
    } else {
        startTestCalibrationWebcamBtn.disabled = true;
        const missing = [];
        if (!testCalibrationLoaded.cameraMatrix) missing.push('Camera Matrix');
        if (!testCalibrationLoaded.distCoeffs) missing.push('Distortion Coefficients');
        if (!testCalibrationLoaded.markerSize) missing.push('Marker Size');
        calibrationUploadStatus.textContent = `Please upload: ${missing.join(', ')}`;
        calibrationUploadStatus.style.color = '#666';
    }
}

// Helper function to upload camera matrix
async function uploadCameraMatrixFile() {
    const matrixFile = cameraMatrixFile.files[0];
    if (!matrixFile) return;

    uploadCameraMatrixBtn.disabled = true;
    cameraMatrixStatus.textContent = 'Uploading...';
    hideError();

    try {
        const formData = new FormData();
        formData.append('file', matrixFile);
        formData.append('file_type', 'camera_matrix');

        const response = await fetch('/calibration/upload-single', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok && data.success) {
            cameraMatrixStatus.textContent = '✅ ' + matrixFile.name;
            cameraMatrixStatus.style.color = '#28a745';
            testCalibrationLoaded.cameraMatrix = true;
            checkCalibrationReady();
            hideError();
        } else {
            cameraMatrixStatus.textContent = '❌ Upload failed';
            cameraMatrixStatus.style.color = '#dc3545';
            showError(data.error || 'Failed to upload camera matrix file');
        }
    } catch (err) {
        cameraMatrixStatus.textContent = '❌ Upload failed';
        cameraMatrixStatus.style.color = '#dc3545';
        showError('Error uploading file: ' + err.message);
    } finally {
        uploadCameraMatrixBtn.disabled = false;
    }
}

// Helper function to upload distortion coefficients
async function uploadDistCoeffsFile() {
    const distFile = distCoeffsFile.files[0];
    if (!distFile) return;

    uploadDistCoeffsBtn.disabled = true;
    distCoeffsStatus.textContent = 'Uploading...';
    hideError();

    try {
        const formData = new FormData();
        formData.append('file', distFile);
        formData.append('file_type', 'dist_coeffs');

        const response = await fetch('/calibration/upload-single', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok && data.success) {
            distCoeffsStatus.textContent = '✅ ' + distFile.name;
            distCoeffsStatus.style.color = '#28a745';
            testCalibrationLoaded.distCoeffs = true;
            checkCalibrationReady();
            hideError();
        } else {
            distCoeffsStatus.textContent = '❌ Upload failed';
            distCoeffsStatus.style.color = '#dc3545';
            showError(data.error || 'Failed to upload distortion coefficients file');
        }
    } catch (err) {
        distCoeffsStatus.textContent = '❌ Upload failed';
        distCoeffsStatus.style.color = '#dc3545';
        showError('Error uploading file: ' + err.message);
    } finally {
        uploadDistCoeffsBtn.disabled = false;
    }
}

// Upload camera matrix file
uploadCameraMatrixBtn.addEventListener('click', async () => {
    alert("button clicked");
    const matrixFile = cameraMatrixFile.files[0];

    if (!matrixFile) {
        // If no file selected, open the file selector
        cameraMatrixFile.click();
        return;
    }

    // If file is selected, upload it
    await uploadCameraMatrixFile();
});

// Auto-upload when camera matrix file is selected
if (cameraMatrixFile) {
    cameraMatrixFile.addEventListener('change', async () => {
        if (cameraMatrixFile.files[0]) {
            await uploadCameraMatrixFile();
        }
    });
}

// Upload distortion coefficients file
if (uploadDistCoeffsBtn) {
    uploadDistCoeffsBtn.addEventListener('click', async () => {
        const distFile = distCoeffsFile.files[0];

        if (!distFile) {
            // If no file selected, open the file selector
            distCoeffsFile.click();
            return;
        }

        // If file is selected, upload it
        await uploadDistCoeffsFile();
    });
}

// Auto-upload when distortion coefficients file is selected
if (distCoeffsFile) {
    distCoeffsFile.addEventListener('change', async () => {
        if (distCoeffsFile.files[0]) {
            await uploadDistCoeffsFile();
        }
    });
}

// Save marker size
if (saveMarkerSizeBtn) {
saveMarkerSizeBtn.addEventListener('click', async () => {
    const markerSize = parseFloat(testMarkerSize.value);

    if (!markerSize || markerSize <= 0) {
        showError('Please enter a valid marker size');
        return;
    }

    saveMarkerSizeBtn.disabled = true;
    markerSizeStatus.textContent = 'Saving...';
    hideError();

    try {
        const formData = new FormData();
        formData.append('marker_size', markerSize);

        const response = await fetch('/calibration/save-marker-size', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok && data.success) {
            markerSizeStatus.textContent = '✅ Saved: ' + markerSize + 'm';
            markerSizeStatus.style.color = '#28a745';
            testCalibrationLoaded.markerSize = true;
            checkCalibrationReady();
            hideError();
        } else {
            markerSizeStatus.textContent = '❌ Save failed';
            markerSizeStatus.style.color = '#dc3545';
            showError(data.error || 'Failed to save marker size');
        }
    } catch (err) {
        markerSizeStatus.textContent = '❌ Save failed';
        markerSizeStatus.style.color = '#dc3545';
        showError('Error saving marker size: ' + err.message);
    } finally {
        saveMarkerSizeBtn.disabled = false;
    }
});
}

// Test calibration webcam functions
async function startTestCalibrationWebcam() {
    if (!testCalibrationLoaded.cameraMatrix || !testCalibrationLoaded.distCoeffs || !testCalibrationLoaded.markerSize) {
        showError('Please upload all calibration files first');
        return;
    }

    try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error('getUserMedia is not supported in this browser.');
        }

        // Stop other webcams
        if (stream) stopWebcam();
        if (calibrationStream) stopCalibrationWebcam();

        testCalibrationStream = await navigator.mediaDevices.getUserMedia({ video: true });
        testCalibrationWebcamVideo.srcObject = testCalibrationStream;
        testCalibrationWebcamVideo.style.display = 'block';
        testCalibrationWebcamPlaceholder.style.display = 'none';
        startTestCalibrationWebcamBtn.style.display = 'none';
        stopTestCalibrationWebcamBtn.style.display = 'inline-block';
        distanceInfo.style.display = 'block';

        // Start processing frames
        testCalibrationInterval = setInterval(processTestCalibrationFrame, 100); // Process every 100ms
    } catch (err) {
        showError('Error accessing webcam: ' + err.message);
    }
}

function stopTestCalibrationWebcam() {
    if (testCalibrationStream) {
        testCalibrationStream.getTracks().forEach(track => track.stop());
        testCalibrationStream = null;
    }
    if (testCalibrationInterval) {
        clearInterval(testCalibrationInterval);
        testCalibrationInterval = null;
    }
    testCalibrationWebcamVideo.style.display = 'none';
    testCalibrationWebcamPlaceholder.style.display = 'flex';
    startTestCalibrationWebcamBtn.style.display = 'inline-block';
    stopTestCalibrationWebcamBtn.style.display = 'none';
    distanceInfo.style.display = 'none';
}

if (startTestCalibrationWebcamBtn) {
    startTestCalibrationWebcamBtn.addEventListener('click', startTestCalibrationWebcam);
}
if (stopTestCalibrationWebcamBtn) {
    stopTestCalibrationWebcamBtn.addEventListener('click', stopTestCalibrationWebcam);
}

async function processTestCalibrationFrame() {
    if (!testCalibrationWebcamVideo || testCalibrationWebcamVideo.readyState !== 4) {
        return;
    }

    try {
        // Capture frame from video
        const canvas = testCalibrationWebcamCanvas;
        const video = testCalibrationWebcamVideo;
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0);

        // Convert to blob and send to server
        canvas.toBlob(async (blob) => {
            const formData = new FormData();
            formData.append('file', blob, 'frame.jpg');
            formData.append('dictionary', dictionarySelect.value);

            try {
                const response = await fetch('/detect/distance', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (response.ok && data.success) {
                    // Update video with annotated frame
                    if (data.result_image) {
                        const img = new Image();
                        img.onload = () => {
                            ctx.clearRect(0, 0, canvas.width, canvas.height);
                            ctx.drawImage(img, 0, 0);
                        };
                        img.src = 'data:image/png;base64,' + data.result_image;
                    }

                    // Update distance information
                    if (data.markers && data.markers.length > 0) {
                        let html = '';
                        data.markers.forEach(marker => {
                            html += `<div style="margin-bottom: 10px; padding: 10px; background: #f0f0f0; border-radius: 5px;">
                                <strong>Marker ID: ${marker.id}</strong><br>
                                Distance: ${marker.distance.toFixed(4)} meters (${(marker.distance * 100).toFixed(2)} cm)
                            </div>`;
                        });
                        distanceDetails.innerHTML = html;
                    } else {
                        distanceDetails.innerHTML = '<p style="color: #666;">No markers detected</p>';
                    }
                }
            } catch (err) {
                // Silently handle errors to avoid spamming console
                console.error('Error processing frame:', err);
            }
        }, 'image/jpeg', 0.8);
    } catch (err) {
        console.error('Error capturing frame:', err);
    }
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (stream) {
        stopWebcam();
    }
    if (calibrationStream) {
        stopCalibrationWebcam();
    }
    if (testCalibrationStream) {
        stopTestCalibrationWebcam();
    }
});

// Calibration functions
const calibrationImageCount = document.getElementById('calibrationImageCount');
const calibrateBtn = document.getElementById('calibrateBtn');
const clearCalibrationBtn = document.getElementById('clearCalibrationBtn');
const calibrationStatus = document.getElementById('calibrationStatus');
const calibrationMethod = document.getElementById('calibrationMethod');
const markerSizeContainer = document.getElementById('markerSizeContainer');

// Calibration webcam elements
const calibrationWebcamVideo = document.getElementById('calibrationWebcamVideo');
const calibrationWebcamCanvas = document.getElementById('calibrationWebcamCanvas');
const calibrationWebcamPlaceholder = document.getElementById('calibrationWebcamPlaceholder');
const startCalibrationWebcamBtn = document.getElementById('startCalibrationWebcamBtn');
const stopCalibrationWebcamBtn = document.getElementById('stopCalibrationWebcamBtn');
const captureCalibrationBtn = document.getElementById('captureCalibrationBtn');

// Show/hide marker size based on method
calibrationMethod.addEventListener('change', () => {
    if (calibrationMethod.value === 'aruco') {
        markerSizeContainer.style.display = 'block';
    } else {
        markerSizeContainer.style.display = 'none';
    }
});

// Calibration webcam functions
async function startCalibrationWebcam() {
    try {
        // Check if getUserMedia is available
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error('getUserMedia is not supported in this browser. Please use a modern browser or ensure you are accessing the site via HTTPS (or localhost).');
        }
        
        // Stop main webcam if running
        if (stream) {
            stopWebcam();
        }
        
        calibrationStream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 1280 },
                height: { ideal: 720 }
            } 
        });
        calibrationWebcamVideo.srcObject = calibrationStream;
        calibrationWebcamVideo.style.display = 'block';
        calibrationWebcamPlaceholder.style.display = 'none';
        startCalibrationWebcamBtn.style.display = 'none';
        stopCalibrationWebcamBtn.style.display = 'inline-block';
        captureCalibrationBtn.style.display = 'inline-block';
        hideError();
    } catch (err) {
        showError('Could not access webcam: ' + err.message);
        console.error('Error accessing webcam:', err);
    }
}

function stopCalibrationWebcam() {
    if (calibrationStream) {
        calibrationStream.getTracks().forEach(track => track.stop());
        calibrationStream = null;
    }
    calibrationWebcamVideo.srcObject = null;
    calibrationWebcamVideo.style.display = 'none';
    calibrationWebcamPlaceholder.style.display = 'flex';
    startCalibrationWebcamBtn.style.display = 'inline-block';
    stopCalibrationWebcamBtn.style.display = 'none';
    captureCalibrationBtn.style.display = 'none';
}

async function captureCalibrationImageFromWebcam() {
    if (!calibrationStream || calibrationWebcamVideo.readyState !== calibrationWebcamVideo.HAVE_ENOUGH_DATA) {
        showError('Webcam not ready. Please wait a moment.');
        return;
    }

    // Set canvas dimensions to match video
    calibrationWebcamCanvas.width = calibrationWebcamVideo.videoWidth;
    calibrationWebcamCanvas.height = calibrationWebcamVideo.videoHeight;

    // Draw current frame to canvas
    const ctx = calibrationWebcamCanvas.getContext('2d');
    ctx.drawImage(calibrationWebcamVideo, 0, 0);

    // Convert canvas to blob
    calibrationWebcamCanvas.toBlob(async (blob) => {
        if (!blob) {
            showError('Failed to capture frame');
            return;
        }

        // Create a File object from the blob
        const file = new File([blob], 'calibration-capture.jpg', { type: 'image/jpeg' });
        
        // Upload the captured image
        await captureCalibrationImage(file);
    }, 'image/jpeg', 0.95);
}

async function captureCalibrationImage(file) {
    const formData = new FormData();
    formData.append('file', file);

    // Show loading
    captureCalibrationBtn.disabled = true;
    loading.classList.add('show');

    try {
        const response = await fetch('/calibration/capture', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            calibrationImageCount.textContent = `Images captured: ${data.image_count}`;
            showError(`✅ Image captured! Total: ${data.image_count}`, 'success');
            await loadCalibrationStatus();
        } else {
            showError(data.error || 'Failed to capture calibration image');
        }
    } catch (err) {
        showError('Error capturing calibration image: ' + err.message);
    } finally {
        loading.classList.remove('show');
        captureCalibrationBtn.disabled = false;
    }
}

// Event listeners for calibration webcam
startCalibrationWebcamBtn.addEventListener('click', startCalibrationWebcam);
stopCalibrationWebcamBtn.addEventListener('click', stopCalibrationWebcam);
captureCalibrationBtn.addEventListener('click', captureCalibrationImageFromWebcam);

async function loadCalibrationStatus() {
    try {
        const response = await fetch('/calibration/status');
        const data = await response.json();
        
        console.log('Calibration status response:', data);

        if (!response.ok) {
            throw new Error(data.error || 'Failed to fetch calibration status');
        }

        // Check if calibrated - data.calibrated is the main flag
        if (data.calibrated === true) {
            const calibInfo = data.calibration_info || {};
            let statusHTML = `
                <p style="color: green; font-weight: 600;">✅ Camera is calibrated</p>
            `;
            
            // Add calibration details if available
            if (calibInfo.focal_length && Array.isArray(calibInfo.focal_length) && calibInfo.focal_length.length >= 2) {
                statusHTML += `
                    <p style="margin-top: 5px; color: #666;">Focal Length: (${calibInfo.focal_length[0].toFixed(2)}, ${calibInfo.focal_length[1].toFixed(2)})</p>
                `;
            }
            
            if (calibInfo.principal_point && Array.isArray(calibInfo.principal_point) && calibInfo.principal_point.length >= 2) {
                statusHTML += `
                    <p style="color: #666;">Principal Point: (${calibInfo.principal_point[0].toFixed(2)}, ${calibInfo.principal_point[1].toFixed(2)})</p>
                `;
            }
            
            statusHTML += `
                <button id="downloadCalibrationResultsBtn" class="webcam-button" style="margin-top: 10px;">Download Calibration Results (zip)</button>
            `;
            
            calibrationStatus.innerHTML = statusHTML;
            
            // Attach event listener to the download button
            const downloadBtn = document.getElementById('downloadCalibrationResultsBtn');
            if (downloadBtn) {
                // Remove any existing listeners first
                const newBtn = downloadBtn.cloneNode(true);
                downloadBtn.parentNode.replaceChild(newBtn, downloadBtn);
                newBtn.addEventListener('click', downloadCalibrationResults);
            } else {
                console.error('Download button not found after creating status HTML');
            }
        } else {
            calibrationStatus.innerHTML = `
                <p style="color: #ffa726; font-weight: 600;">⚠️ Camera not calibrated</p>
                <p style="margin-top: 5px; color: #666;">Capture at least 3 calibration images to begin calibration.</p>
            `;
        }
        
        if (data.image_count !== undefined) {
            calibrationImageCount.textContent = `Images captured: ${data.image_count}`;
        }
    } catch (err) {
        console.error('Error loading calibration status:', err);
        calibrationStatus.innerHTML = `<p style="color: red;">Error loading calibration status: ${err.message}</p>`;
    }
}

calibrateBtn.addEventListener('click', async () => {
    const method = 'chessboard';//calibrationMethod.value;
    const squareSize = 0.015;//parseFloat(document.getElementById('squareSize').value);
    const markerSize = 0.05;//parseFloat(document.getElementById('markerSize').value);
    const dictionary = dictionarySelect.value;

    calibrateBtn.disabled = true;
    loading.classList.add('show');
    hideError();

    try {
        const formData = new FormData();
        formData.append('method', method);
        formData.append('square_size', squareSize);
        formData.append('dictionary', dictionary);
        if (method === 'aruco') {
            formData.append('marker_size', markerSize);
        }

        const response = await fetch('/calibration/calibrate', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok && data.success) {
            showError(`✅ ${data.message}`, 'success');
            // Wait a moment for file system operations to complete
            await new Promise(resolve => setTimeout(resolve, 500));
            await loadCalibrationStatus();
        } else {
            showError(data.error || 'Calibration failed');
            // Still refresh status to show current state
            await loadCalibrationStatus();
        }
    } catch (err) {
        showError('Error during calibration: ' + err.message);
    } finally {
        loading.classList.remove('show');
        calibrateBtn.disabled = false;
    }
});

clearCalibrationBtn.addEventListener('click', async () => {
    if (!confirm('Are you sure you want to clear all calibration images?')) {
        return;
    }

    try {
        const response = await fetch('/calibration/images', {
            method: 'DELETE'
        });

        const data = await response.json();

        if (response.ok) {
            calibrationImageCount.textContent = 'Images captured: 0';
            await loadCalibrationStatus();
            hideError();
        } else {
            showError(data.error || 'Failed to clear images');
        }
    } catch (err) {
        showError('Error clearing images: ' + err.message);
    }
});

function downloadCalibrationPattern() {
    window.open('/assets/CameraCalibMatrix.pdf', '_blank');
}

// Make function globally accessible
async function downloadCalibrationResults() {
    try {
        const response = await fetch("/assets/calibration_output.zip");
        
        // ensure response ok
        if (!response.ok) {
            alert(`Download failed: ${response.status} ${response.statusText}. The calibration output may not have been created yet.`);
            console.error('Download failed:', response.status, response.statusText);
            return;
        }
        
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        
        const link = document.createElement("a");
        link.href = url;
        link.download = "calibration_output.zip";
        document.body.appendChild(link);
        link.click();
        
        // cleanup
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    } catch (err) {
        alert(`Error downloading calibration results: ${err.message}`);
        console.error('Error downloading calibration results:', err);
    }
}

// Make it globally accessible for onclick handlers
window.downloadCalibrationResults = downloadCalibrationResults;
    