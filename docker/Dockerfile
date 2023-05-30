FROM malikitmo/uavid_semantic:latest

ENV UAVid /workspace

WORKDIR $UAVid

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \ 
	 apt-get install -y git ffmpeg libsm6 libxext6 
 
RUN git init .
RUN git remote add -t \* -f origin https://github.com/maliksyria/Semantic_Segmentation_UAVid.git
RUN git checkout -f master
RUN pip3 install --upgrade pip wheel setuptools requests
RUN pip3 install -r ./requirements.txt
