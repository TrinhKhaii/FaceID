const DETECTOR_API_URL = 'http://localhost:8001/detect';
const PIPELINE_API_URL = 'http://localhost:8004/identify_crop';

const video = document.getElementById('video-feed');
const canvas = document.getElementById('video-canvas');
const ctx = canvas.getContext('2d', { alpha: false });
const statusDiv = document.getElementById('status');
const startCameraBtn = document.getElementById('start-camera-btn');


let latestDetections = [];
let facesData = [];
let isProcessing = false;
let isIdentifying = false;
let cameraStarted = false;
let animationFrameId = null;

const offCanvas = document.createElement('canvas');
const offCtx = offCanvas.getContext('2d');


async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1280 }, height: { ideal: 720 } },
            audio: false
        });
        window.localStream = stream; 
        video.srcObject = stream;
        await video.play();

        while (video.videoWidth === 0) {
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        const w = video.videoWidth;
        const h = video.videoHeight;

        canvas.width = w;
        canvas.height = h;
        offCanvas.width = w;
        offCanvas.height = h;
        cameraStarted = true;
        animationFrameId = requestAnimationFrame(gameLoop);

    } catch (err) {
        cameraStarted = false;
        console.error(err);
    }
}

function stopCamera() {
    cameraStarted = false;

    if (animationFrameId !== null) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }

    if (window.localStream) {
        window.localStream.getTracks().forEach(track => track.stop());
        window.localStream = null;
    }

    if (video) {
        video.pause();
        video.srcObject = null;
    }
    if (canvas && ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    latestDetections = [];
    facesData = [];
    isProcessing = false;
    isIdentifying = false;
}



function gameLoop(timestamp) {
    if (!cameraStarted || video.paused || video.ended) return;
    renderFrame();
    // DETECT
    if (!isProcessing && video.readyState === 4) {
        isProcessing = true; 
        processDetect().finally(() => {
            isProcessing = false; 
        });
    }
    // IDENTIFY
    if (!isIdentifying && facesData.length > 0) {
        isIdentifying = true;
        taskIdentify().finally(() => {
            isIdentifying = false;
        });
    }
    animationFrameId = requestAnimationFrame(gameLoop);
}


function renderFrame() {
    const w = canvas.width;
    const h = canvas.height;

    ctx.save();
    ctx.scale(-1, 1);
    ctx.drawImage(video, -w, 0, w, h);
    ctx.restore();

    if (latestDetections.length > 0) {
        ctx.lineWidth = 3;
        ctx.font = "bold 18px Arial";

        for (const det of latestDetections) {
            let { x, y, w: bw, h: bh } = det.bbox;
            let mirroredX = w - x - bw;

            let color = '#ffff00';
            if (det.name && det.name !== 'Unknown') {
                color = '#00ff00';
            } else if (det.name === 'Unknown') {
                color = '#ff6b6b';
            }

            ctx.strokeStyle = color;
            ctx.fillStyle = color;

            ctx.strokeRect(mirroredX, y, bw, bh);

            let label = 'Detecting...';
            if (det.name) {
                const scorePct = det.score ? (det.score * 100).toFixed(1) : '0.0';
                label = `${det.name} (${scorePct}%)`;
            }
            ctx.fillText(label, mirroredX, y - 10);
        }
    }
}

function iou(boxA, boxB) {
    const xA = Math.max(boxA.x, boxB.x);
    const yA = Math.max(boxA.y, boxB.y);
    const xB = Math.min(boxA.x + boxA.w, boxB.x + boxB.w);
    const yB = Math.min(boxA.y + boxA.h, boxB.y + boxB.h);

    const interW = Math.max(0, xB - xA);
    const interH = Math.max(0, yB - yA);
    const interArea = interW * interH;

    const boxAArea = boxA.w * boxA.h;
    const boxBArea = boxB.w * boxB.h;

    const union = boxAArea + boxBArea - interArea;
    if (union <= 0) return 0;

    return interArea / union;
}


function findMatchedPreviousDetection(newBox, iouThreshold = 0.5) {
    let bestMatch = null;
    let bestIoU = 0;

    for (const oldDet of latestDetections) {
        const currentIoU = iou(newBox, oldDet.bbox);
        if (currentIoU > bestIoU && currentIoU >= iouThreshold) {
            bestIoU = currentIoU;
            bestMatch = oldDet;
        }
    }
    return bestMatch;
}


async function processDetect() {
    offCtx.drawImage(video, 0, 0, offCanvas.width, offCanvas.height);

    const blob = await new Promise(resolve => offCanvas.toBlob(resolve, 'image/jpeg', 0.8));
    if (!blob) return;

    const formData = new FormData();
    formData.append('image', blob, 'frame.jpg');

    const t0 = performance.now();
    try {
        const res = await fetch(DETECTOR_API_URL, { method: 'POST', body: formData });
        if (!res.ok) {
            console.warn("Detect API error, status:", res.status);
            return;
        }
        const data = await res.json();

        const detections = data.detections || [];
        const newDetections = [];
        const newFacesForIdentify = [];

        for (const det of detections) {
            const matchedOld = findMatchedPreviousDetection(det.bbox);
            
            if (matchedOld && matchedOld.name) {
                newDetections.push({
                    ...det,
                    name: matchedOld.name,
                    score: matchedOld.score
                });
            } else {
                newDetections.push({
                    ...det,
                    name: null,
                    score: 0
                });
                newFacesForIdentify.push(det);
            }
        }
        latestDetections = newDetections;
        facesData = newFacesForIdentify;
    } catch (e) {
        console.warn("API Detect Error:", e);
    }
}


async function taskIdentify() {
    if (facesData.length === 0) return;

    const promises = facesData.map(async (det) => {
        try {
            const res = await fetch(PIPELINE_API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    cropped_face_base64: det.cropped_face_base64
                })
            });

            if (res.ok) {
                const result = await res.json();

                const matched = findMatchedPreviousDetection(det.bbox);
                if (matched) {
                    matched.name = result.name || 'Unknown';
                    matched.score = result.score || 0.0;
                }
            }
        } catch (err) {
            console.error("Identify error:", err);
        }
    });

    await Promise.all(promises);

}


startCameraBtn.addEventListener('click', async () => {
    if (!cameraStarted) {
        startCameraBtn.disabled = true;

        try {
            await startCamera();
            startCameraBtn.textContent = 'Turn off camera';
        } catch (err) {
            console.error('Start camera error:', err);
            startCameraBtn.textContent = 'Turn on camera';
        } finally {
            startCameraBtn.disabled = false;
        }

    } else {
        stopCamera();
        startCameraBtn.textContent = 'Turn on camera';
    }
});
