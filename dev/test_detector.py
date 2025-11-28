import requests
import cv2
import numpy as np
import base64


API_URL = "http://localhost:8001/detect"
IMAGE_PATH = "dev/tests/data/trump2.jpg"

def test_single_detect():
    try:
        with open(IMAGE_PATH, 'rb') as f:
            files = {'image': f}
            response = requests.post(API_URL, files=files)
    except FileNotFoundError:
        print("FileNotFoundError")
        return
    except Exception as e:
        print(f"Error: {e}")
        return

    if response.status_code != 200:
        print(f"API error {response.status_code}: {response.text}")
        return

    data = response.json()
    print("API response success!")
    
    if not data.get("success"):
        print("API cannot find any faces")
        return

    detections = data.get("detections", [])
    if not detections:
        print("Detections empty")
        return

    first_face = detections[0]
    bbox = first_face.get("bbox")
    
    print(f"\n--------------")
    print(f"BBox: {bbox}")
    print(f"Crop image size: {first_face.get('width')}x{first_face.get('height')}")

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("Cannot read by OpenCV")
        return

    if bbox:
        x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(img, f"Face: {w}x{h}", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow(img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_single_detect()

    