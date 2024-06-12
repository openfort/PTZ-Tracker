# NDI-PTZ-Tracker
A free PTZ tracker for any NDI-enabled PTZ camera. The software runs on a desktop and captures the NDI video feed from any PTZ camera on the network. Any detected head can be selected for tracking and the software will send NDI PTZ commands to the camera to keep the detected head in frame.

## tested hardware
- GTX1070 and Birddog P400

## dependencies
- yolov8
- scuthead dataset
- onnx engine
- ndi-python wrapper
- opencv

## installation windows
the inference uses windows direct-ml, only windows version is avaiable yet.
- download repo
- install python 3.10.11
- install requirements
- run in python or build with nuitka
