import os
import torch
import argparse
import numpy as np

from app.model import Generator
from PIL import Image
from util import *


def toonify(latent1, latent2):
    with torch.no_grad():
        noise1 = [getattr(generator1.noises, f"noise_{i}") for i in range(generator1.num_layers)]
        noise2 = [getattr(generator2.noises, f"noise_{i}") for i in range(generator2.num_layers)]

        out1 = generator1.input(latent1[0])
        out2 = generator2.input(latent2[0])
        out = (1 - early_alpha) * out1 + early_alpha * out2

        out1, _ = generator1.conv1(out, latent1[0], noise=noise1[0])
        out2, _ = generator2.conv1(out, latent2[0], noise=noise2[0])
        out = (1 - early_alpha) * out1 + early_alpha * out2

        skip1 = generator1.to_rgb1(out, latent1[1])
        skip2 = generator2.to_rgb1(out, latent2[1])
        skip = (1 - early_alpha) * skip1 + early_alpha * skip2

        i = 2
        for conv1_1, conv1_2, noise1_1, noise1_2, to_rgb1, conv2_1, conv2_2, noise2_1, noise2_2, to_rgb2 in zip(
            generator1.convs[::2],
            generator1.convs[1::2],
            noise1[1::2],
            noise1[2::2],
            generator1.to_rgbs,
            generator2.convs[::2],
            generator2.convs[1::2],
            noise2[1::2],
            noise2[2::2],
            generator2.to_rgbs,
        ):

            conv_alpha = early_alpha if i < num_swap else alpha
            out1, _ = conv1_1(out, latent1[i], noise=noise1_1)
            out2, _ = conv2_1(out, latent2[i], noise=noise2_1)
            out = (1 - conv_alpha) * out1 + conv_alpha * out2
            i += 1

            conv_alpha = early_alpha if i < num_swap else alpha
            out1, _ = conv1_2(out, latent1[i], noise=noise1_2)
            out2, _ = conv2_2(out, latent2[i], noise=noise2_2)
            out = (1 - conv_alpha) * out1 + conv_alpha * out2
            i += 1

            conv_alpha = early_alpha if i < num_swap else alpha
            skip1 = to_rgb1(out, latent1[i], skip)
            skip2 = to_rgb2(out, latent2[i], skip)
            skip = (1 - conv_alpha) * skip1 + conv_alpha * skip2

            i += 1

    image = skip.clamp(-1, 1)

    return image


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--disney_seed", type=int, default=686868, help="디즈니 사진 번호")
    parser.add_argument("-n", "--num_swap", type=int, default=7, help="클수록 실사화")
    parser.add_argument("-a", "--alpha", type=float, default=0.4, help="클수록 만화화")
    parser.add_argument("--early_alpha", type=float, default=0)
    parser.add_argument("files", metavar="FILES", nargs="+")

    args = parser.parse_args()

    out_dir = "./toon_result"
    os.makedirs(out_dir, exist_ok=True)

    generator1 = Generator(256, 512, 8, channel_multiplier=2).eval().to(device)
    generator2 = Generator(256, 512, 8, channel_multiplier=2).to(device).eval()

    mean_latent1 = load_model(generator1, "face.pt")
    mean_latent2 = load_model(generator2, "disney.pt")

    truncation = 0.5
    disney_seed = args.disney_seed  # @param {type:"number"}

    with torch.no_grad():
        torch.manual_seed(disney_seed)
        reference_code = torch.randn([1, 512]).to(device)
        latent2 = generator2.get_latent(reference_code, truncation=truncation, mean_latent=mean_latent2)
        reference_im, _ = generator2(latent2)

    num_swap = args.num_swap  # default = 6
    alpha = args.alpha  # defaut = 0.5

    early_alpha = args.early_alpha

    ### Load real world latent
    Image.open(args.files[0])

    latent_path = f"./inversion_codes/{os.path.splitext(os.path.basename(args.files[0]))[0]}.pt"

    latent_real = torch.load(latent_path)["latent"]
    latent_real = style2list(latent_real)

    source_im, _ = generator1(latent_real)

    result = toonify(latent_real, latent2)

    if result.is_cuda:
        result = result.cpu()
    if result.dim() == 4:
        result = result[0]
    # tensor 로 변환된 이미지는 [C,H,W] 형태를 가지고 있으니 opencv나 plt로 보이게 하려면 numpy형태[H,W,C]로 바꾸어야한다.
    result = ((result.clamp(-1, 1) + 1) / 2).permute(1, 2, 0).detach().numpy()

    # PIL 은 [W,H,C]
    result = Image.fromarray((result * 255.0).astype(np.uint8))

    output_path = f"{out_dir}/{os.path.splitext(os.path.basename(args.files[0]))[0]}_toon.jpg"
    uniq = 1

    while os.path.exists(output_path):
        output_path = f"{out_dir}/{os.path.splitext(os.path.basename(args.files[0]))[0]}_toon({uniq}).jpg"
        uniq += 1
    result.save(output_path)
