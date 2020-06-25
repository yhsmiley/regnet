xhost +local:docker
docker run -it --gpus all -v /media/data/regnet:/regnet -v /media/data/detection-dataset:/dataset --net=host --ipc host -e DISPLAY=unix$DISPLAY  --privileged -v /dev/:/dev/ regnet
