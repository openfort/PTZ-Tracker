# NDI PTZ Tracker
A free PTZ tracker for any NDI-enabled PTZ camera. The software runs on a desktop and captures the NDI video feed from any PTZ camera on the network. Any detected head can be selected for tracking and the software will send NDI PTZ commands to the camera to keep the detected head in frame.

## Demo
The tracker is in use at lifechurch wil. Here are some links to see tracking in action on youtube:
- [youtube - lifechurch wil](https://youtu.be/Er5B_IqR304?t=709)
- [youtube - lifechurch wil](https://youtu.be/-PTu4VsTdoA?t=1351)

## Tested hardware
- GTX1660 and Birddog P400

## Dependencies
- yolov8
- scuthead dataset
- onnx engine
- ndi-python wrapper
- opencv

## Installation windows
The inference runs on windows direct-ml, only windows version is avaiable yet.
- download repo
- install python 3.10.11
- install requirements
- run in python or build with nuitka
