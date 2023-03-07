# FastAPI_test

Deepfake 구현 중

---

<br/>

### 1) Directory 구조
```shell
   fastapi_test
   |
   ├── 📁 ai
   │    ├── 📁 SOAT ( Toonification )
   │    │    ├── 📁 imgs
   |    |    |    └──  ⋮  
   |    |    ├── 📁 op
   |    |    |    └──  ⋮
   |    |    ├── 💾 projector.py ( 생성모델을 통한 사용자 이미지 인버젼 )
   |    |    ├── 💾 toonify.py ( 만화캐릭터와 합성 )
   |    |    └──  ⋮
   |    |
   │    └── 📁 Thin-Plate-Spline-Motion-Model ( Deep-fake )
   |         ├── 📁 checkpoints
   │         ├── 💾 realtime.py
   │         └──  ⋮ 
   |
   ├── 📁 app
   |    ├── 💾 __main__.py
   |    ├── 💾 main.py
   |    ├── 💾 frontend.py
   │    └──  ⋮
   |
   ├── 💾 README.md
   └── 💾 requirements.txt

```
<br/>

### 2) Start Toonification


1. Python requirements  

    `Python` : 3.10.9

    <br/>

2. Installation  

    1. 가상 환경 설정  

    2. CUDA 설정  

        * [https://github.com/alsqja/O-Talk/wiki/CUDA](https://github.com/alsqja/O-Talk/wiki/CUDA)
        * 파이토치 설치 : `pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113`
    
    <br/>

    3. 프로젝트 의존성 설치.  
    
        * ```> pip install -r requirements.txt```
    
    <br/>

    4. 가중치 다운로드  

        모든 가중치는 `ai/SOAT`에 다운받기.
        * [face.pt](https://drive.google.com/file/d/1BmhpJkpunxGUFvJD4SefC6oNBswc9-TH/view?usp=sharing)
        * [disney.pt](https://drive.google.com/file/d/1ypADNNH0gTiPG-iJItapkqdBAIKNybtc/view?usp=sharing)
        * [inversion_stats.npz](https://drive.google.com/file/d/1dNXXFIn9-VRy1MyOe95hZxLEFdA5J8Rf/view?usp=sharing)

    <br/>

    5. Frontend(Streamlit)와 Server(FastAPI)를 같이 실행.  
    ```> make -j 2 run_app```

<br/>
<br/>





**참고**
* [https://mumin-blog.tistory.com/337](https://mumin-blog.tistory.com/337)
