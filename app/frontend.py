import io
import os
import base64
from pathlib import Path

import requests
from PIL import Image

import streamlit as st

# from confirm_button_hack import cache_on_button_press

# # SETTING PAGE CONFIG TO WIDE MODE
# ASSETS_DIR_PATH = os.path.join(Path(__file__).parent.parent.parent.parent, "assets")

# st.set_page_config(layout="wide")

# root_password = "password"


def main():
    st.title("Inversion Model")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))

        st.image(image, caption="Uploaded Image")
        st.write("Inversion...")

        # 기존 stremalit 코드
        # _, y_hat = get_prediction(model, image_bytes)
        # label = config['classes'][y_hat.item()]
        files = [("files", (uploaded_file.name, image_bytes, uploaded_file.type))]
        response = requests.post("http://localhost:8000/inversion", files=files)
        img_ar = response.json()["products"][0]["img_result"]
        # ASCII코드로 변환된 bytes 데이터(str) -> bytes로 변환 -> 이미지로 디코딩
        img_ar = Image.open(io.BytesIO(base64.b64decode(img_ar)))
        st.image(img_ar)

        response = requests.post("http://localhost:8000/toonify")
        toon_result = response.json()["products"][0]["img_result"]
        toon_result = Image.open(io.BytesIO(base64.b64decode(toon_result)))
        st.image(toon_result)


main()
# @cache_on_button_press("Authenticate")
# def authenticate(password) -> bool:
#     return password == root_password


# password = st.text_input("password", type="password")

# if authenticate(password):
#     st.success("You are authenticated!")
#     main()
# else:
#     st.error("The password is invalid.")
