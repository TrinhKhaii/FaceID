const addUserForm = document.getElementById('add-user-form');
const addUserStatus = document.getElementById('add-user-status');

const startCamBtn = document.getElementById('start-capture-camera');
const captureBtn1 = document.getElementById('capture-shot-1');
const captureBtn2 = document.getElementById('capture-shot-2');

const video = document.getElementById('capture-video');
const canvas = document.getElementById('capture-canvas');
const preview1 = document.getElementById('preview-shot-1');
const preview2 = document.getElementById('preview-shot-2');

let captureStream = null;
let capturedBlob1 = null;
let capturedBlob2 = null;


const ADD_USER_API_URL = 'http://localhost:8004/add_user_form';


async function startCaptureCamera() {
    try {
        captureStream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 } },
            audio: false
        });

        video.srcObject = captureStream;
        await video.play();

        while (video.videoWidth === 0) {
            await new Promise(resolve => setTimeout(resolve, 50));
        }
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        captureBtn1.disabled = false;
        captureBtn2.disabled = false;

    } catch (err) {
        console.error('Error starting capture camera:', err);
    }
}

function stopCaptureCamera() {
    if (captureStream) {
        captureStream.getTracks().forEach(t => t.stop());
        captureStream = null;
    }
    if (video) {
        video.pause();
        video.srcObject = null;
    }
}

async function captureShot(target = 1) {
    if (!captureStream || video.readyState < 2) {
        return;
    }

    const ctx = canvas.getContext('2d', { alpha: false });
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise(resolve =>
        canvas.toBlob(resolve, 'image/jpeg', 0.9)
    );
    if (!blob) {
        return;
    }

    const url = URL.createObjectURL(blob);

    if (target === 1) {
        capturedBlob1 = blob;
        preview1.src = url;
    } else {
        capturedBlob2 = blob;
        preview2.src = url;
    }
}

startCamBtn.addEventListener('click', async () => {
    if (!captureStream) {
        startCamBtn.disabled = true;
        startCamBtn.textContent = 'Turning on camera...';
        await startCaptureCamera();
        startCamBtn.textContent = 'Turn off camera';
        startCamBtn.disabled = false;
    } else {
        stopCaptureCamera();
        startCamBtn.textContent = 'Turn on camera';
        captureBtn1.disabled = true;
        captureBtn2.disabled = true;
    }
});

captureBtn1.addEventListener('click', () => captureShot(1));
captureBtn2.addEventListener('click', () => captureShot(2));

// SUBMIT FORM: GỬI 2 BLOB NHƯ 2 FILE
addUserForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    const fullName = document.getElementById('full-name').value.trim();
    if (!fullName) {
        addUserStatus.style.color = '#dc2626';
        addUserStatus.textContent = 'Vui lòng nhập họ tên.';
        return;
    }

    if (!capturedBlob1 || !capturedBlob2) {
        addUserStatus.style.color = '#dc2626';
        addUserStatus.textContent = 'Vui lòng chụp đủ 2 ảnh khuôn mặt.';
        return;
    }

    const formData = new FormData();
    formData.append('full_name', fullName);
    formData.append('face_image_1', capturedBlob1, 'face1.jpg');
    formData.append('face_image_2', capturedBlob2, 'face2.jpg');

    addUserStatus.style.color = '#0369a1';
    addUserStatus.textContent = 'Đang gửi dữ liệu...';

    try {
        const res = await fetch(ADD_USER_API_URL, {
            method: 'POST',
            body: formData
        });

        const data = await res.json().catch(() => ({}));

        if (res.ok && data.success) {
            addUserStatus.style.color = '#16a34a';
            addUserStatus.textContent = `Đã thêm ${data.embeddings_added} embedding cho ${data.name}`;
            // Reset nếu muốn:
            // addUserForm.reset();
            // capturedBlob1 = capturedBlob2 = null;
            // preview1.src = preview2.src = '';
        } else {
            addUserStatus.style.color = '#dc2626';
            addUserStatus.textContent =
                data.error || (data.detail ? JSON.stringify(data.detail) : 'Lỗi lưu dữ liệu.');
        }
    } catch (err) {
        console.error('Add user error:', err);
        addUserStatus.style.color = '#dc2626';
        addUserStatus.textContent = 'Không thể kết nối server.';
    }
});

// DỌN DẸP KHI RỜI TRANG
window.addEventListener('beforeunload', () => {
    stopCaptureCamera();
});
