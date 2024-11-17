from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_cors import CORS
import os
import torch
import cv2
import time

app = Flask(__name__)
CORS(app)

# YOLOv9 저장소 경로 설정 (로컬 경로)
yolov9_local_path = os.path.abspath('./yolov9')
custom_weights = './pt/best.pt'

# YOLOv9 모델 불러오기
try:
    model = torch.hub.load(yolov9_local_path, 'custom', path=custom_weights, source='local')
    print("YOLOv9 모델이 성공적으로 로드되었습니다.")
except Exception as e:
    print(f"YOLOv9 모델 로드 중 오류 발생: {e}")
    exit(1)

# 파일 업로드 페이지 렌더링 (GET 요청 전용)
@app.route('/upload', methods=['GET'])
def upload_page():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>File Upload</title>
    </head>
    <body>
        <h2>Upload a file</h2>
        <form action="/upload_file" method="post" enctype="multipart/form-data">
            <label for="file">Choose file:</label>
            <input type="file" name="file" id="file" required>
            <button type="submit">Upload</button>
        </form>
    </body>
    </html>
    '''

# Flask 파일 업로드 및 YOLOv9 실행 (POST 요청 전용)
@app.route('/upload_file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"message": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"message": "No file selected for uploading"}), 400

    # 파일 저장 경로 설정
    file_path = os.path.join("./detect", file.filename)
    file.save(file_path)

    # 파일 형식에 따라 처리
    if file.filename.endswith(('.jpg', '.jpeg', '.png')):
        # 이미지 파일 처리
        img = cv2.imread(file_path)
        if img is None:
            return jsonify({"message": "Error: Could not read image."}), 500

        # YOLOv9 모델로 이미지 처리
        results = model(img)
        results.render()  # 이미지에 결과 렌더링

        # 감지된 객체 정보 출력 (디버깅용)
        detections = results.xyxy[0]
        if detections is None or len(detections) == 0:
            print("No objects detected in the image.")
        else:
            for det in detections:
                x1, y1, x2, y2, confidence, class_id = det.tolist()
                print(f"Detected object - Class ID: {class_id}, Confidence: {confidence}, BBox: ({x1}, {y1}), ({x2}, {y2})")

        # 처리된 이미지 저장
        result_image_path = os.path.join('./detect', f"output_{file.filename}")
        cv2.imwrite(result_image_path, img)

        return jsonify({"message": "Image processed successfully", "output": result_image_path}), 200

    elif file.filename.endswith('.mp4'):
        # 비디오 파일 처리
        def process_video(file_path):
            cap = cv2.VideoCapture(file_path)
            output_folder = './detect/'
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = model(frame)
                results.render()
                result_image_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
                cv2.imwrite(result_image_path, frame)
                frame_count += 1
            cap.release()
            return "Video processed successfully."

        video_result = process_video(file_path)
        return jsonify({"message": video_result, "output": "output path"}), 200

    else:
        return jsonify({"message": "Unsupported file format. Only JPG, PNG, and MP4 are supported."}), 400

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
