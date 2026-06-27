document.addEventListener('DOMContentLoaded', () => {

    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const overlay = document.getElementById('overlay');
    const nameInput = document.getElementById('name');
    const regNoInput = document.getElementById('reg_no');
    const captureBtn = document.getElementById('captureBtn');
    const trainBtn = document.getElementById('trainBtn');

    const IMAGES_TO_CAPTURE = 20;

    // --- General Camera Setup ---
    async function setupCamera() {
        try {
            // Request camera access
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
            // Wait for the video metadata to load to get correct dimensions
            return new Promise((resolve) => {
                video.onloadedmetadata = () => resolve(video);
            });
        } catch (err) {
            console.error("Camera access error:", err);
            alert("Could not access the camera. Please ensure it's not in use by another application and that you have granted permissions in your browser.");
        }
    }

    // --- Enrollment Page Logic ---
    if (captureBtn) {
        setupCamera();
        const captureProgress = document.getElementById('captureProgress');
        const progressBar = document.getElementById('progressBar');
        const captureStatus = document.getElementById('captureStatus');

        captureBtn.addEventListener('click', async () => {
            const personName = nameInput.value.trim();
            const regNo = regNoInput.value.trim();

            if (!personName || !regNo) {
                alert('Please enter both name and registration number.');
                return;
            }

            captureBtn.disabled = true;
            captureProgress.style.display = 'block';
            captureStatus.innerHTML = 'Starting capture...';

            for (let i = 1; i <= IMAGES_TO_CAPTURE; i++) {
                // Draw current video frame to a hidden canvas
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
                // Convert canvas image to a data URL (base64 encoded JPEG)
                const imageDataURL = canvas.toDataURL('image/jpeg');

                try {
                    // Send the captured image and user data to the backend API
                    const response = await fetch('/api/capture', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ name: personName, reg_no: regNo, image: imageDataURL })
                    });
                    const data = await response.json();

                    if(data.status !== 'success') throw new Error(data.message);

                    // Update the progress bar and status message
                    let percent = Math.round((i / IMAGES_TO_CAPTURE) * 100);
                    progressBar.style.width = percent + '%';
                    progressBar.innerText = percent + '%';
                    captureStatus.innerHTML = `Captured image ${i}/${IMAGES_TO_CAPTURE}`;

                } catch (error) {
                    captureStatus.innerHTML = `<span class="text-danger">Error: ${error.message}</span>`;
                    break; // Stop capturing if an error occurs
                }
                // Wait briefly between captures
                await new Promise(resolve => setTimeout(resolve, 500));
            }

            captureStatus.innerHTML = `<span class="text-success">Capture complete for ${personName}! You can now train the model.</span>`;
            captureBtn.disabled = false;
        });
    }

    // --- Training Button Logic (Dashboard) ---
    if (trainBtn) {
        trainBtn.addEventListener('click', async () => {
            const trainingStatus = document.getElementById('trainingStatus');
            trainingStatus.innerHTML = '<div class="spinner-border spinner-border-sm"></div> Training... This may take several minutes.';
            trainBtn.disabled = true;

            try {
                // Send a request to the backend to start the training process
                const response = await fetch('/api/train', { method: 'POST' });
                const data = await response.json();
                if (data.status === 'success') {
                    trainingStatus.innerHTML = `<span class="text-success">${data.message}</span>`;
                    // Reload the page after 2 seconds to show the updated model status
                    setTimeout(() => window.location.reload(), 2000);
                } else {
                    throw new Error(data.message);
                }
            } catch (error) {
                trainingStatus.innerHTML = `<span class="text-danger">Training failed: ${error.message}</span>`;
            } finally {
                trainBtn.disabled = false;
            }
        });
    }

    // --- Live Attendance Page Logic ---
    if (overlay) {
        const statusDiv = document.getElementById('status');
        setupCamera().then(video => {
            // Match the drawing canvas size to the video stream size
            overlay.width = video.videoWidth;
            overlay.height = video.videoHeight;
            const context = overlay.getContext('2d');

            // Check if the model is ready on the server before starting
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    if (data.model_loaded) {
                        statusDiv.className = 'alert alert-success text-center';
                        statusDiv.innerText = 'Model loaded. Starting live recognition...';

                        // Start the recognition loop
                        setInterval(async () => {
                            const tempCanvas = document.createElement('canvas');
                            tempCanvas.width = video.videoWidth;
                            tempCanvas.height = video.videoHeight;
                            tempCanvas.getContext('2d').drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
                            const imageDataURL = tempCanvas.toDataURL('image/jpeg');

                            try {
                                // Send the current frame to the recognition API
                                const response = await fetch('/api/recognize', {
                                    method: 'POST',
                                    headers: { 'Content-Type': 'application/json' },
                                    body: JSON.stringify({ image: imageDataURL })
                                });

                                const data = await response.json();
                                // Clear previous drawings
                                context.clearRect(0, 0, overlay.width, overlay.height);

                                // Draw the new results
                                if (data.results) {
                                    data.results.forEach(face => {
                                        const { box, name, confidence } = face;
                                        const [x, y, w, h] = box;
                                        const color = name.includes('Unknown') ? '#dc3545' : '#198754'; // Red for Unknown, Green for recognized

                                        // Draw bounding box
                                        context.strokeStyle = color;
                                        context.lineWidth = 3;
                                        context.strokeRect(x, y, w, h);

                                        // Draw text label with background
                                        const text = `${name}`;
                                        const textWidth = context.measureText(text).width;
                                        context.fillStyle = color;
                                        context.fillRect(x, y > 20 ? y - 22 : y, textWidth + 10, 22);

                                        context.fillStyle = 'white';
                                        context.font = '16px Arial';
                                        context.fillText(text, x + 5, y > 20 ? y - 5 : y + 16);
                                    });
                                }
                            } catch (error) {
                                console.error("Recognition error:", error);
                            }
                        }, 500); // Send a frame for recognition twice per second
                    } else {
                        statusDiv.className = 'alert alert-danger text-center';
                        statusDiv.innerText = data.message || 'Model not loaded on server. Please train the model from the dashboard.';
                    }
                });
        });
    }

    // --- Manage Users Page Logic ---
    const saveButtons = document.querySelectorAll('.save-btn');
    if (saveButtons.length > 0) {
        const statusMessage = document.getElementById('status-message');

        saveButtons.forEach(button => {
            button.addEventListener('click', async () => {
                const regNo = button.getAttribute('data-reg-no');
                const nameInput = document.getElementById(`name-${regNo}`);
                const newName = nameInput.value.trim();

                if (!newName) {
                    alert('Name cannot be empty.');
                    return;
                }

                try {
                    // Send request to update the user's name in the registry
                    const response = await fetch('/api/update_registry', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ reg_no: regNo, name: newName })
                    });
                    const data = await response.json();

                    if (data.success) {
                        statusMessage.innerHTML = `<div class="alert alert-success">${data.message}</div>`;
                    } else {
                        throw new Error(data.message);
                    }
                } catch (error) {
                    statusMessage.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
                }
            });
        });
    }
});