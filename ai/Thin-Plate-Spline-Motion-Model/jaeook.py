import sys
import cv2
import torch
import numpy as np
from skimage.transform import resize
import warnings
from skimage import img_as_ubyte

from demo import load_checkpoints
from demo import make_animation
from demo import find_best_frame as _find

warnings.filterwarnings("ignore")


device = "cuda" if torch.cuda.is_available() else "cpu"
dataset_name = "vox"  # ['vox', 'taichi', 'ted', 'mgif'] 얼굴, 몸전체, 상반신, 말?
config_path = "config/vox-256.yaml"  # vox 모델은 256x256으로 합성됨
checkpoint_path = "checkpoints/vox.pth.tar"
predict_mode = "relative"  # ['standard', 'relative', 'avd']
find_best_frame = True  # when use the relative mode to animate a face, use 'find_best_frame=True' can get better quality result

pixel = 256  # for vox, taichi and mgif, the resolution is 256*256
if dataset_name == "ted":  # for ted, the resolution is 384*384
    pixel = 384

inpainting, kp_detector, dense_motion_network, avd_network = load_checkpoints(
    config_path=config_path, checkpoint_path=checkpoint_path, device=device
)


##### Preprocessing
source_image_path = "assets/source.png"
driving_video_path = "assets/driving.mp4"

source_image = cv2.imread(source_image_path)
source_image = resize(source_image, (pixel, pixel))[..., :3]


# cap = cv2.VideoCapture(driving_video_path)
cap = cv2.VideoCapture(0)
driving_video = []
if cap.isOpened():
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)
    print(f"초당 프레임(FPS) : {cap.get(cv2.CAP_PROP_FPS)}")
    print(f"총 프레임 : {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = resize(frame, (pixel, pixel))[..., :3]
        driving_video.append(frame)

        cv2.imshow("VideoFrame", frame)
        if cv2.waitKey(1) != -1:
            break
cap.release()
cv2.destroyAllWindows()

#### inference
print("Start Inference.....")
output_video_path = "assets/webcam2.mp4"

if predict_mode == "relative" and find_best_frame:
    i = _find(source_image, driving_video, device == "cpu")
    print("Best frame: " + str(i))
    driving_forward = driving_video[i:]
    driving_backward = driving_video[: (i + 1)][::-1]
    predictions_forward = make_animation(
        source_image,
        driving_forward,
        inpainting,
        kp_detector,
        dense_motion_network,
        avd_network,
        device=device,
        mode=predict_mode,
    )
    predictions_backward = make_animation(
        source_image,
        driving_backward,
        inpainting,
        kp_detector,
        dense_motion_network,
        avd_network,
        device=device,
        mode=predict_mode,
    )
    predictions = predictions_backward[::-1] + predictions_forward[1:]
else:
    predictions = make_animation(
        source_image,
        driving_video,
        inpainting,
        kp_detector,
        dense_motion_network,
        avd_network,
        device=device,
        mode=predict_mode,
    )
print("Finish Inference !!")

#### save result video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (pixel, pixel))
for frame in predictions:
    out.write(img_as_ubyte(frame))
out.release()
cv2.destroyAllWindows()
