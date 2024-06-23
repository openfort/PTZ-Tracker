# NDI PTZ Tracker

A free PTZ tracker for any NDI-enabled PTZ camera. The software runs on a desktop and captures the NDI video feed from any PTZ camera on the network. Any detected head can be selected for tracking and the software will send NDI PTZ commands to the camera to keep the detected head in frame.

## Demo

![](C:\Users\jansc\Documents\Software\GitHub\NDI-PTZ-Tracker\images\demo.jpg)

The tracker is in use at lifechurch wil. Here are some links to see tracking in action on youtube:

- [youtube - lifechurch wil](https://youtu.be/Er5B_IqR304?t=710)
- [youtube - lifechurch wil](https://youtu.be/-PTu4VsTdoA?t=1350)
- [youtube - lifechurch wil](https://youtu.be/pv6bBC2xMHI?t=1510)

## Usage

- Write the camera name and speed to the configx.json file, where x corresponds to the number of running instances.

- Use the keys w, a, s, d, e and c to move the camera

- Click on a green head to initiate tracking.

- Press x to stop tracking

- Press q to quit application

## Tested hardware

- GTX1660 and Birddog P400 (Up to 4 individual streams at 25 fps)

## Related projects

- Ultralytics YOLOv8 [GitHub](https://github.com/ultralytics/ultralytics)
- onnxruntime [GitHub](https://github.com/microsoft/onnxruntime)
- NDI Python wrapper [GitHub](https://github.com/buresu/ndi-python)

## Installation Windows

The inference runs on windows direct-ml, only windows version is avaiable yet.

1. Download Release (tracker.zip) file
2. Extract files
3. Run tracker.exe

## Build Windows

1. Download repo

2. Install python 3.9.7

3. Install requirements from requirements.txt
   
   `pip install -r requirements.txt`

4. Run in python or build with nuitka
   
   `py .\tracker.py`
   
   `nuitka .\tracker.py`
