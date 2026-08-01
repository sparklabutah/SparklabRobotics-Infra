import cv2
import pyrealsense2 as rs
import numpy as np
from flask import Flask, render_template_string, Response

app = Flask(__name__)

# TODO: Replace with your actual 3 RealSense camera serial numbers
# You can find them using the "realsense-viewer" tool
serial_numbers = ["353322271147", "323622272781", "243322071190"]
pipelines = []

# Initialize all 3 cameras
for sn in serial_numbers:
    try:
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(sn)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipe.start(cfg)
        pipelines.append(pipe)
        print(f"Successfully started camera: {sn}")
    except Exception as e:
        print(f"Error starting camera {sn}: {e}")

def generate_frames(camera_index):
    """Generator function to fetch frames from a specific camera."""
    if camera_index >= len(pipelines):
        return
    
    pipe = pipelines[camera_index]
    while True:
        try:
            frames = pipe.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())

            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', color_image)
            frame_bytes = buffer.tobytes()

            # Yield the frame in byte format for MJPEG streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Error reading from camera {camera_index}: {e}")
            break

@app.route('/')
def index():
    """Main page layout grid displaying all 3 streams side-by-side."""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RealSense Multi-Camera Stream</title>
        <style>
            .container { display: flex; justify-content: center; gap: 20px; font-family: sans-serif; }
            .cam-box { text-align: center; border: 2px solid #ccc; padding: 10px; border-radius: 8px; }
            img { width: 100%; max-width: 400px; height: auto; background: #222; }
        </style>
    </head>
    <body>
        <h1 style="text-align:center;">Intel RealSense 3-Camera Dashboard</h1>
        <div class="container">
            <div class="cam-box"><h3>Camera 1</h3><img src="/video_feed/0"></div>
            <div class="cam-box"><h3>Camera 2</h3><img src="/video_feed/1"></div>
            <div class="cam-box"><h3>Camera 3</h3><img src="/video_feed/2"></div>
        </div>
    </body>
    </html>
    """
    return render_template_string(html_template)

@app.route('/video_feed/<int:cam_id>')
def video_feed(cam_id):
    """Dynamic route for each individual camera stream."""
    return Response(generate_frames(cam_id), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Run the Flask app on port 8080. 
    # threaded=True allows Flask to process multiple image streams concurrently.
    app.run(host='0.0.0.0', port=8080, threaded=True)
