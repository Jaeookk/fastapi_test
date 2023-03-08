import numpy as np
import imageio
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
import cv2
from demo import load_checkpoints, relative_kp

souce_image_path = "assets/result.jpg"
cap = cv2.VideoCapture(0)

img_shape = (256, 256)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("assets/result.mp4", fourcc, 30.0, (1024, 512))

device = torch.device("cuda")

inpainting, kp_detector, dense_motion_network, avd_network = load_checkpoints(
    config_path="config/vox-256.yaml",
    checkpoint_path="checkpoints/vox.pth.tar",
    device=device,
)

source_image = imageio.imread(souce_image_path)
source_image = resize(source_image, img_shape)[..., :3]
source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(
    0, 3, 1, 2
)
source = source.to(device)

kp_source = kp_detector(source)

kp_driving_initial = None

with torch.no_grad():
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        img = img[200 : 200 + 256, 200 : 200 + 256]
        img = cv2.flip(img, 1)

        input_img = (
            torch.tensor(img[:, :, ::-1].astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device)
        )

        kp_driving = kp_detector(input_img)

        if kp_driving_initial is None:
            kp_driving_initial = kp_driving

        kp_norm = relative_kp(
            kp_source=kp_source,
            kp_driving=kp_driving,
            kp_driving_initial=kp_driving_initial,
        )

        dense_motion = dense_motion_network(
            source_image=source,
            kp_driving=kp_norm,
            kp_source=kp_source,
            bg_param=None,
            dropout_flag=False,
        )
        inpaint_result = inpainting(source, dense_motion)

        prediction = np.transpose(
            inpaint_result["prediction"].data.cpu().numpy(), [0, 2, 3, 1]
        )[0]
        prediction = img_as_ubyte(prediction)[:, :, ::-1]

        img = cv2.resize(img, dsize=(512, 512))
        prediction = cv2.resize(prediction, dsize=(512, 512))

        result = np.hstack([img, prediction])

        out.write(result)

        cv2.imshow("result", result)
        if cv2.waitKey(1) == ord("q"):
            break
