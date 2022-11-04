docker run \
       -it --privileged \
       --name hyperpcr-cnt \
       --user user \
       --net=host \
       -e DISPLAY=$DISPLAY -e NVIDIA_DRIVER_CAPABILITIES=all \
       -v /tmp/.X11-unix:/tmp/.X11-unix \
       -v /dev:/dev \
       --gpus all --runtime=nvidia \
       hyperpcr-img
