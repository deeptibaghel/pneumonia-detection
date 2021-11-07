FROM python:3.8-slim-buster

RUN apt-get update -y

# gcc compiler and opencv prerequisites
RUN apt-get -y install git build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev
RUN apt-get -y install ffmpeg libsm6 libxext6 

# Detectron2 prerequisites
RUN pip install torch==1.9.0+cpu torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install cython
RUN pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# Detectron2 - CPU copy
RUN python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.9/index.html

# Development packages
RUN pip install flask flask-cors requests opencv-python

WORKDIR /app
COPY web_api.py web_api.py
COPY model_final.pth model_final.pth
COPY test_img.jpg test_img.jpg
ENTRYPOINT ["python", "/app/web_api.py"]