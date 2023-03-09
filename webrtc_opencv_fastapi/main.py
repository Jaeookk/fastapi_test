import asyncio
import os
import cv2
import torch
import imageio
import numpy as np
import time
from skimage.transform import resize
from skimage import img_as_ubyte
from deepfake.demo import load_checkpoints, relative_kp

from av import VideoFrame

from imageai.Detection import VideoObjectDetection

from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer, MediaRelay, MediaBlackhole

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.templating import Jinja2Templates

from src.schemas import Offer

ROOT = os.path.dirname(__file__)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eyes = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
smiles = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")


class VideoTransformTrack(MediaStreamTrack):
    """
    A video stream track that transforms frames from an another track.
    """

    kind = "video"

    def __init__(self, track, transform, *args):
        super().__init__()
        self.track = track
        self.transform = transform
        if args:
            self.inpainting = args[0]
            self.kp_detector = args[1]
            self.dense_motion_network = args[2]
            self.avd_network = args[3]
            self.source = args[4]
            self.kp_source = args[5]

    async def recv(self):
        frame = await self.track.recv()

        if self.transform == "cartoon":
            start = time.time()
            print("cartoon time check")
            img = frame.to_ndarray(format="bgr24")

            # prepare color
            img_color = cv2.pyrDown(cv2.pyrDown(img))
            for _ in range(6):
                img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
            img_color = cv2.pyrUp(cv2.pyrUp(img_color))

            # prepare edges
            img_edges = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_edges = cv2.adaptiveThreshold(
                cv2.medianBlur(img_edges, 7),
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                9,
                2,
            )
            img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)

            # combine color and edges
            img = cv2.bitwise_and(img_color, img_edges)

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            print(f"cartoon : {time.time()-start}")
            print(new_frame.pts)
            print(new_frame.time_base)
            return new_frame
        elif self.transform == "edges":
            # perform edge detection
            img = frame.to_ndarray(format="bgr24")
            img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        elif self.transform == "rotate":
            # rotate image
            img = frame.to_ndarray(format="bgr24")
            rows, cols, _ = img.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), frame.time * 45, 1)
            img = cv2.warpAffine(img, M, (cols, rows))

            # rebuild a VideoFrame, preserving timing information
            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            return new_frame
        elif self.transform == "cv":
            start = time.time()
            img = frame.to_ndarray(format="bgr24")
            face = faces.detectMultiScale(img, 1.1, 19)
            for (x, y, w, h) in face:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            eye = eyes.detectMultiScale(img, 1.1, 19)
            for (x, y, w, h) in eye:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # smile = smiles.detectMultiScale(img, 1.1, 19)
            # for (x, y, w, h) in smile:
            #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 5), 2)

            new_frame = VideoFrame.from_ndarray(img, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            print(f"cv time : {time.time()-start}")
            print(new_frame.pts)
            print(new_frame.time_base)
            return new_frame
        elif self.transform == "otalk":
            print("start")
            start = time.time()
            # ----------------------------------------------------------------------------------------------------------------------
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            img = frame.to_ndarray(format="bgr24")
            print(f"img load : {time.time()-start}")

            with torch.no_grad():  # 시간 측정 해보자 몇초 걸리는지, 그리고 다른 것들과 비교해보자... 뭐가 문제인지 위의 모델 load 문제인지, 아니면 그냥 inference 속도 문제인지
                tmp = time.time()
                img = img[200 : 200 + 256, 200 : 200 + 256]
                img = cv2.flip(img, 1)
                print(f"img transform : {time.time()-tmp}")

                tmp = time.time()
                kp_driving_initial = None
                input_img = (
                    torch.tensor(img[:, :, ::-1].astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
                )
                print(f"img to cuda : {time.time()-tmp}")

                tmp = time.time()
                kp_driving = self.kp_detector(input_img)
                if kp_driving_initial is None:
                    kp_driving_initial = kp_driving
                print(f"kp_driving : {time.time()-tmp}")

                tmp = time.time()
                kp_norm = relative_kp(
                    kp_source=self.kp_source,
                    kp_driving=kp_driving,
                    kp_driving_initial=kp_driving_initial,
                )
                print(f"kp_norm : {time.time()-tmp}")

                tmp = time.time()
                dense_motion = self.dense_motion_network(
                    source_image=self.source,
                    kp_driving=kp_norm,
                    kp_source=self.kp_source,
                    bg_param=None,
                    dropout_flag=False,
                )
                print(f"dense_motion : {time.time()-tmp}")

                tmp = time.time()
                inpaint_result = self.inpainting(self.source, dense_motion)
                print("finish inference 1 frame")
                print(f"inpaint_result : {time.time()-tmp}")

                tmp = time.time()
                prediction = inpaint_result["prediction"]
                print(f"prediction : {time.time()-tmp}")
                prediction = prediction.detach()
                print(f"detach : {time.time()-tmp}")
                prediction = prediction.cpu()
                print(f"cpu : {time.time()-tmp}")
                prediction = prediction.numpy()
                print(f"numpy : {time.time()-tmp}")
                prediction = np.transpose(prediction, [0, 2, 3, 1])[0]
                print(f"np.transpose : {time.time()-tmp}")
                prediction = img_as_ubyte(prediction)[:, :, ::-1]
                print(f"img_as_ubyte : {time.time()-tmp}")
                # prediction = cv2.resize(prediction, dsize=(512, 512))
                # print(f"prediction : {time.time()-tmp}")

            tmp = time.time()
            new_frame = VideoFrame.from_ndarray(prediction, format="bgr24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            print(f"new_frame : {time.time()-tmp}")
            print(f"otalk : {time.time()-start}")
            print(new_frame.pts)
            print(new_frame.time_base)
            print("*" * 50)
            return new_frame
        else:
            return frame


def create_local_tracks(play_from=None):
    if play_from:
        player = MediaPlayer(play_from)
        return player.audio, player.video
    else:
        options = {"framerate": "30", "video_size": "1920x1080"}
        # if relay is None:
        # if platform.system() == "Darwin":
        # webcam = MediaPlayer(
        #     "default:none", format="avfoundation", options=options
        # )
        # elif platform.system() == "Windows":
        # webcam = MediaPlayer("video.mp4")
        webcam = MediaPlayer("video=FULL HD 1080P Webcam", format="dshow", options=options)

        # else:
        # webcam = MediaPlayer("/dev/video0", format="v4l2", options=options)
        # audio, video = VideoTransformTrack(webcam.video, transform="cv")
        relay = MediaRelay()
        return None, relay.subscribe(webcam.video)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/cv", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index_cv.html", {"request": request})


@app.post("/offer")
async def offer(params: Offer):
    offer = RTCSessionDescription(sdp=params.sdp, type=params.type)

    pc = RTCPeerConnection()
    pcs.add(pc)
    recorder = MediaBlackhole()

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    # open media source
    audio, video = create_local_tracks()

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()

    await pc.setRemoteDescription(offer)
    for t in pc.getTransceivers():
        if t.kind == "audio" and audio:
            pc.addTrack(audio)
        elif t.kind == "video" and video:
            pc.addTrack(video)

    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


@app.post("/offer_cv")
async def offer(params: Offer):
    offer = RTCSessionDescription(sdp=params.sdp, type=params.type)

    pc = RTCPeerConnection()
    pcs.add(pc)
    recorder = MediaBlackhole()

    relay = MediaRelay()

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    # open media source
    # audio, video = create_local_tracks()

    @pc.on("track")
    def on_track(track):

        # if track.kind == "audio":
        #     pc.addTrack(player.audio)
        #     recorder.addTrack(track)
        if track.kind == "video":
            start = time.time()
            if params.video_transform == "otalk":
                souce_image_path = "./deepfake/assets/result.jpg"

                img_shape = (256, 256)

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                inpainting, kp_detector, dense_motion_network, avd_network = load_checkpoints(
                    config_path="./deepfake/config/vox-256.yaml",
                    checkpoint_path="./deepfake/checkpoints/vox.pth.tar",
                    device=device,
                )

                source_image = imageio.imread(souce_image_path)
                source_image = resize(source_image, img_shape)[..., :3]
                source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
                source = source.to(device)

                kp_source = kp_detector(source)
                pc.addTrack(
                    VideoTransformTrack(
                        relay.subscribe(track),
                        "otalk",
                        inpainting,
                        kp_detector,
                        dense_motion_network,
                        avd_network,
                        source,
                        kp_source,
                    )
                )
                print(f"track : {time.time()-start}")
            else:
                pc.addTrack(VideoTransformTrack(relay.subscribe(track), transform=params.video_transform))
            # if args.record_to:
            #     recorder.addTrack(relay.subscribe(track))

        @track.on("ended")
        async def on_ended():
            await recorder.stop()

    # handle offer
    await pc.setRemoteDescription(offer)
    await recorder.start()

    # send answer
    answer = await pc.createAnswer()
    await pc.setRemoteDescription(offer)
    await pc.setLocalDescription(answer)

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


pcs = set()
args = ""


@app.on_event("shutdown")
async def on_shutdown():
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()
