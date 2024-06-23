# nuitka-project: --standalone
# nuitka-project: --disable-console
# nuitka-project: --include-data-files={MAIN_DIRECTORY}/model/head_s_640.onnx=model/head_s_640.onnx
# nuitka-project: --include-data-files={MAIN_DIRECTORY}/model/nose-pose19Ps.onnx=model/nose-pose19Ps.onnx
# nuitka-project: --include-data-files={MAIN_DIRECTORY}/images/cat.jpg=images/cat.jpg
# nuitka-project: --include-data-files={MAIN_DIRECTORY}/default_config.json=default_config.json

# updates:
    # fixed some performance issues and preview scaling
    # multi config support
    # integrated ndi find in get frame
    # show sources on startup
    # updated tensor conversion
    # resolution matched to input frame
    # changed model to width 640
    # speed correction from json
    # improved manual cam controls
    # individual axis lock

# issues:
    # no known issues

# comming soon features:
    # -

# author: openfort
# date: 22.02.24
# version: 3.10

from kivy.app import App
from kivy.uix.image import AsyncImage
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from threading import Thread
import NDIlib as ndi
import numpy as np
import onnxruntime as rt
import cv2
import json
import psutil

def intersec(single_box, bounding_boxes):
    """
    Calculate the intersection between a single bounding box and an array of bounding boxes.
    Parameters:
    - single_box: Single bounding box in the format (x, y, w, h).
    - bounding_boxes: Array of bounding boxes in the format (x, y, w, h).
    Returns:
    - Array of intersection values between the single box and each box in the array.
    """
    dx = np.abs(single_box[0] - bounding_boxes[:, 0])
    dy = np.abs(single_box[1] - bounding_boxes[:, 1])
    intersection = np.zeros(len(bounding_boxes))
    mask_x = dx <= (bounding_boxes[:, 2] + single_box[2]) / 2
    mask_y = dy <= (bounding_boxes[:, 3] + single_box[3]) / 2
    mask = mask_x & mask_y
    u = np.maximum(single_box[1] - single_box[3] / 2, bounding_boxes[:, 1] - bounding_boxes[:, 3] / 2)
    t = np.minimum(single_box[1] + single_box[3] / 2, bounding_boxes[:, 1] + bounding_boxes[:, 3] / 2)
    r = np.maximum(single_box[0] - single_box[2] / 2, bounding_boxes[:, 0] - bounding_boxes[:, 2] / 2)
    l = np.minimum(single_box[0] + single_box[2] / 2, bounding_boxes[:, 0] + bounding_boxes[:, 2] / 2)
    h = np.maximum(0, t - u)
    w = np.maximum(0, l - r)
    intersection[mask] = (2 * w[mask] * h[mask]) / (single_box[2] * single_box[3] + bounding_boxes[mask, 2] * bounding_boxes[mask, 3])
    return np.array(intersection)

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.speed = 15000
        self.prev_error = 0
        self.integral = 0
    
    def calc_Kp(self, object_shape):
        # big object Kd = 7 (130000), small object (10000) Kd = 2
        #size = object_shape[2] * object_shape[3]
        #self.Kp = (size-1000)/self.speed+1.5
        pass
        #print(self.Kp)
    
    def set_Kp(self, Kp):
        #self.Kp = Kp
        pass

    def calculate(self, setpoint, process_variable):
        error = setpoint - process_variable
        self.integral += error
        self.integral = np.clip(self.integral, -0.5, 0.5)
        derivative = error - self.prev_error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

class MOSSE_Tracker:
    def __init__(self, frame, tbbox):  # xywh
        #size = tbbox[2] if tbbox[2] < tbbox[3] else tbbox[3]
        self.tbbox = np.array((tbbox[0]-tbbox[2]//2, tbbox[1]-tbbox[3]//2, tbbox[2], tbbox[3])).astype(np.int32)    # tlwh
        self.tracker = cv2.legacy.TrackerMOSSE_create()
        self.tracker.init(cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR), self.tbbox)
        
    def update(self, frame):
        success, tbbox = self.tracker.update(cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR))
        if success:
            #self.bbox = [ bbox[0]+bbox[2]//2, bbox[1]+bbox[3]//2, bbox[2], bbox[3] ]
            self.tbbox = np.array(tbbox).astype(np.int32)
        return [self.tbbox[0]+self.tbbox[2]//2, self.tbbox[1]+self.tbbox[3]//2, self.tbbox[2], self.tbbox[3]]

class Webcam:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.frame = np.zeros(1)
        self.ptz_values = np.zeros(3)
        self.ndi_recv = None
        self.ptz_en = False

    def get_frame(self):
        r, frame = self.cap.read()
        if r:
            self.frame = frame
            return frame
        else:
            return np.zeros(1)
        
    def close(self):
        self.cap.release()

    def pan_tilt(self, pan_tilt):
        pass
    def zoom(self, zoom):
        pass

class NDIstream:
    def __init__(self, NDI_cam, resolution):
        self.ptz_en = False
        self.ptz_values = np.zeros(3)       # pan, tilt, zoom
        self.old_ptz_values = np.zeros(3)
        self.resolution = resolution
        self.name = NDI_cam
        self.frame = None
        self.ndi_recv = None
        self.ndi_find = None
        ndi.initialize()
        return

    def new_source(self, source):
        ndi_recv_create = ndi.RecvCreateV3()
        ndi_recv_create.color_format = ndi.RECV_COLOR_FORMAT_BGRX_BGRA
        ndi_recv_create.bandwidth = ndi.RECV_BANDWIDTH_LOWEST
        ndi_recv = ndi.recv_create_v3(ndi_recv_create)
        if ndi_recv is None:
            return
        ndi.recv_connect(ndi_recv, source)
        self.ndi_recv = ndi_recv

    def find_sources(self):
        if not self.ndi_find:
            self.ndi_find = ndi.find_create_v2()
        if self.ndi_find is None:
            return
        ndi.find_wait_for_sources(self.ndi_find, 100)
        sources = ndi.find_get_current_sources(self.ndi_find)
        sources_text = []
        sources_text.append('Looking for sources ...')
        for i, s in enumerate(sources):
            sources_text.append(f'   {i+1}. {s.ndi_name}')
        for s in sources:
            if self.name in s.ndi_name:
                self.name = s.ndi_name
                self.new_source(s)
                return
        return sources_text

    def get_frame(self):
        if self.ndi_recv == None:
            self.frame = cv2.resize(cv2.imread('images/cat.jpg'), self.resolution, interpolation=cv2.INTER_NEAREST)
            text = self.find_sources()
            text_position = [self.resolution[0]//3-30, 24]
            if text:
                for line in text:
                    text_position[1] += 20
                    cv2.putText(self.frame, line, org=text_position, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0,0,0), thickness=1)
            return self.frame
        else:
            while True:
                t, v, a, m = ndi.recv_capture_v3(self.ndi_recv, 100)
                if t == ndi.FRAME_TYPE_VIDEO:
                    # print('Video data received (%dx%d).' % (v.xres, v.yres))
                    if v.xres == 640:
                        self.frame = np.copy(v.data[:352,:,:3])     # crop to 640x352
                    else:
                        self.frame = cv2.resize(np.copy(v.data[:,:,0:3]), self.resolution, interpolation=cv2.INTER_NEAREST)     # with resizing to 640x352
                    ndi.recv_free_video_v2(self.ndi_recv, v)
                    return self.frame
                elif t == ndi.FRANE_TYPE_STATUS_CHANGE:
                    if ndi.recv_ptz_is_supported(self.ndi_recv):
                        self.ptz_en = True
                        print("ptz is available")
                elif t == ndi.FRAME_TYPE_AUDIO:
                    ndi.recv_free_audio_v3(self.ndi_recv, a)
                elif t == ndi.FRAME_TYPE_METADATA:
                    ndi.recv_free_metadata(self.ndi_recv, m)
    
    def pan_tilt(self, pan_tilt):
        self.ptz_values[0] += pan_tilt[0]
        self.ptz_values[1] += pan_tilt[1]
        self.ptz_values[0:2] = np.clip(self.ptz_values[0:2], -1, 1)

    def zoom(self, zoom):
        self.ptz_values[2] += zoom
        self.ptz_values[2] = np.clip(self.ptz_values[2], -1, 1)

    def send_ptz_values(self):
        if self.ptz_en and (self.ptz_values != self.old_ptz_values):
            ndi.recv_ptz_pan_tilt_speed(self.ndi_recv, -self.ptz_values[0], -self.ptz_values[1])
            ndi.recv_ptz_zoom_speed(self.ndi_recv, self.ptz_values[2])
        self.old_ptz_values = self.ptz_values
        self.ptz_values = np.zeros(3)
        return self.old_ptz_values

    def save_preset(self, number):
        if self.ptz_en:
            ndi.recv_ptz_store_preset(self.ndi_recv, number)

    def load_preset(self, number):
        if self.ptz_en:
            ndi.recv_ptz_recall_preset(self.ndi_recv, number, 1)

    def close(self):
        if self.ptz_en:
            ndi.recv_ptz_pan_tilt_speed(self.ndi_recv,0,0)
            ndi.recv_ptz_zoom_speed(self.ndi_recv,0)
        if self.ndi_recv:
            ndi.recv_destroy(self.ndi_recv)
        if self.ndi_find:
            ndi.find_destroy(self.ndi_find)
        ndi.destroy()
        print('ndi close')

class YOLOv8:
    def __init__(self, model_path, CONFIDENCE_THRESHOLD=0.5, NMS_THRESHOLD=0.5):
        self.CONFIDENCE_THRESHOLD = CONFIDENCE_THRESHOLD # min required confidence from ml result
        self.NMS_THRESHOLD = NMS_THRESHOLD # max allowed overlap between results
        self.model_path = model_path
        #### Load Model
        opt_session = rt.SessionOptions()       # onnx options
        opt_session.execution_mode = rt.ExecutionMode.ORT_PARALLEL
        opt_session.graph_optimization_level = rt.GraphOptimizationLevel.ORT_DISABLE_ALL
        EP_list = ['DmlExecutionProvider', 'CPUExecutionProvider']
        self.ort_session = rt.InferenceSession(model_path, sess_options=opt_session, providers=EP_list)      # onnx runtime
        model_inputs = self.ort_session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        model_output = self.ort_session.get_outputs()
        self.output_names = [model_output[i].name for i in range(len(model_output))]

    def conv_input_tensor(self, img):
        if img.any():
            if not np.array_equal((img.shape[1], img.shape[0]), (self.input_shape[3], self.input_shape[2])):              # skip resize if shape already matches
                img = cv2.resize(img, (self.input_shape[3], self.input_shape[2]), interpolation=cv2.INTER_NEAREST)        # resize, input_shape[b, c, y, x]
                #print(f'resize to: {(self.input_shape[3], self.input_shape[2])}')
            img = img.transpose(2,0,1)                       # change dimension to (channel, width, height)
            img = img[[2, 1, 0], :, :].astype(np.float32)    # convert BGRX to RGB, but faster
            #img = cv2.cvtColor(img,cv2.COLOR_BGRA2RGB)      # convert to RGB
            img /= 256                                       # normalize
            img = img[np.newaxis, :, :, :]                   # add a batch dimension
            return img
        return np.zeros(1)
    
    def detection(self, imageBGRX):
        image_shape = imageBGRX.shape
        input_tensor = self.conv_input_tensor(imageBGRX)
        outputs = self.ort_session.run(self.output_names, {self.input_names[0]: input_tensor})[0]      # inference
        predictions = np.squeeze(outputs).T
        # Filter out object confidence scores below threshold
        predictions = predictions[predictions[:, 4] > self.CONFIDENCE_THRESHOLD, :]
        # get scores
        scores = predictions[:, 4]
        # Get bounding boxes for each object
        results = np.delete(predictions, 4, axis=1)
        #rescale results
        if scores.any():
            input_height, input_width = self.input_shape[2:]
            image_height, image_width = image_shape[:2]
            results = np.divide(results, np.tile([input_width, input_height], len(results[0])//2), dtype=np.float32)
            results *= np.tile([image_width, image_height], len(results[0])//2)
            indices = cv2.dnn.NMSBoxes(results[:,:4], scores, self.CONFIDENCE_THRESHOLD, self.NMS_THRESHOLD)
            return np.array(results[indices]).astype(np.int32), np.array(scores[indices])
        else:
            return np.zeros((1,4)), np.zeros(1)

class tracked_obj:
    def __init__(self, bbox, index):
        self.tracked_box = bbox      # xywh
        self.tracked_index = index
        self.direction = np.zeros(2)    # xy
        self.direction_new = False
        self.lost_frames = 0
        self.mosse = None
        self.active = 1
        self.movement = np.zeros(2)     # moved direction from last frame
    
    def update(self, detected_heads, frame):
        # get intersections from all detected heads
        xy_old = self.tracked_box[:2]
        intersections = intersec(self.tracked_box, detected_heads)
        self.direction_new = False
        if np.max(intersections) == 0:  
            self.lost_frames += 1        # no matching object
            if self.lost_frames == 1:
                #print('init mosse')
                self.mosse = MOSSE_Tracker(frame, self.tracked_box)
            elif self.lost_frames < 3*25:  # max length for mosse tracking, 3 seconds at 25 fps
                #print('update mosse', self.lost_frames)
                self.tracked_box = self.mosse.update(frame)
            else:
                #print('face lost')
                return 0
        else:
            self.tracked_index = np.argmax(intersections)
            self.tracked_box = detected_heads[self.tracked_index]
            self.movement = self.tracked_box[:2]-xy_old
            if self.lost_frames != 0:
                self.lost_frames = 0
                self.mosse = None
            #print('tracked')
        return 1

class CamController:
    def __init__(self, cam, Vorschau, center, speed=2, lock_hvz=[False, True, True]):
        self.cam = cam
        self.shape = Vorschau
        self.lock_ptz = lock_hvz
        self.center = np.array(center)
        self.xy = np.divide(self.center, self.shape)    # normalized target
        self.target_height = 0.25                       # normalized target height
        self.deadzone = np.array((0.03, 0.03, 0.02))     # pan, tilt, zoom
        self.pidcontroller = [PIDController(speed,0,0), PIDController(speed,0,0), PIDController(6,0,0)]
        self.speed_factor = np.zeros(3)
        self.speed_factor_strength = 200        # big number(200)->weak, small number(10)->strong
        self.max_speed = 1
        self.moving_flag = 0
        self.fps = 25

    def calc_controlls(self, actual_point, target_point, pidcontroller, deadzone_size=0.1, movement_direction=0, mov_dir_strength=80, max_speed=1, exponent=1):
        if movement_direction > mov_dir_strength:
            movement_direction = mov_dir_strength
        if actual_point < target_point - deadzone_size/2:
            actual_point *= target_point/(target_point-deadzone_size/2)
            speed_factor = -movement_direction/mov_dir_strength
        elif actual_point > target_point + deadzone_size/2:
            actual_point = target_point + (actual_point-target_point-deadzone_size/2)*(1-target_point)/(1-target_point-deadzone_size/2)
            speed_factor = movement_direction/mov_dir_strength
        else:
            actual_point = target_point
            speed_factor = 0
        speed_factor += 1
        actual_point = np.clip(actual_point, 0, 1)
        result = speed_factor*pidcontroller.calculate(target_point, actual_point)
        result = -(abs(result)**exponent) if result < 0 else (result**exponent)
        return np.clip(-result, -max_speed, max_speed)

    def update(self, tracked):
        tracked_box = tracked.tracked_box
        tracked_point = np.divide(tracked_box[0:2], self.shape)     # norm position of the tracked object
        if tracked_box[2] > tracked_box[3]:                         # norm by height of tracked object
            tracked_height = tracked_box[2] / self.shape[1]  # width larger than height
        else:
            tracked_height = tracked_box[3] / self.shape[1]  # height larger than width
        if self.lock_ptz[0]:
            pan = 0
        else:
            pan = self.calc_controlls(tracked_point[0], self.xy[0], self.pidcontroller[0], self.deadzone[0], tracked.movement[0], mov_dir_strength=self.speed_factor_strength, exponent=1.3)
            if self.fps < 24:
                pan *= self.fps / 24
        if self.lock_ptz[1]:
            tilt = 0
        else:
            tilt = self.calc_controlls(tracked_point[1], self.xy[1], self.pidcontroller[1], self.deadzone[1], tracked.movement[1], mov_dir_strength=self.speed_factor_strength, exponent=1.3, max_speed=0.5)
            if self.fps < 24:
                tilt *= self.fps / 24
        self.cam.pan_tilt((pan, tilt))
        if not self.lock_ptz[2]:
            zoom = self.calc_controlls(tracked_height, self.target_height, self.pidcontroller[2], self.deadzone[2], exponent=1.2)
            if self.fps < 24:
                zoom *= self.fps / 24
            self.cam.zoom(-zoom)

    def control(self, keys):
        speed = 0.3
        if 'e' in keys:
            self.moving_flag = 1
            self.cam.zoom(speed*2)
        if 'c' in keys:
            self.moving_flag = 1
            self.cam.zoom(-speed*2)
        if 'w' in keys:
            self.moving_flag = 1
            self.cam.pan_tilt((0, -speed/3))
        if 'a' in keys:
            self.moving_flag = 1
            self.cam.pan_tilt((-speed, 0))
        if 's' in keys:
            self.moving_flag = 1
            self.cam.pan_tilt((0, speed/3))
        if 'd' in keys:
            self.moving_flag = 1
            self.cam.pan_tilt((speed, 0))
        if keys == [] and self.moving_flag:
            self.moving_flag = 0
            self.cam.zoom(0)
            self.cam.pan_tilt((0, 0))

    def stop(self):
        self.cam.pan_tilt((0, 0))
        self.cam.zoom(0)

class windowGUI:
    def __init__(self, name, controller):
        self.name = name
        self.target_center = controller.center
        self.deadzone = controller.deadzone
        self.pid = controller.pidcontroller
        self.head_color = (0,150,0)
        self.tracked_color = (150,0,0)
        self.lost_color = (150,0,150)
        self.image = None

    def update(self, image, boundig_boxes, tracked_object, ptz_values):
        def xywh2xyxy(x):
            # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
            y = np.copy(x)
            y[..., 0] = x[..., 0] - x[..., 2] / 2
            y[..., 1] = x[..., 1] - x[..., 3] / 2
            y[..., 2] = x[..., 0] + x[..., 2] / 2
            y[..., 3] = x[..., 1] + x[..., 3] / 2
            return y
        image_draw = image.copy()
        if boundig_boxes.any():
            for bbox in zip(xywh2xyxy(boundig_boxes)):
                bbox = np.array(bbox[0])
                cv2.rectangle(image_draw, bbox[:2], bbox[2:], self.head_color, thickness=1)
        cv2.line(image_draw, (int(self.target_center[0]-self.target_center[0]*self.deadzone[0]), self.target_center[1]), (int(self.target_center[0]+self.target_center[0]*self.deadzone[0]), self.target_center[1]), color=(150,0,100), thickness=1)
        cv2.line(image_draw, (self.target_center[0], int(self.target_center[1]-self.target_center[1]*self.deadzone[1])), (self.target_center[0], int(self.target_center[1]+self.target_center[1]*self.deadzone[1])), color=(150,0,100), thickness=1)
        if tracked_object != None:
            bbox = xywh2xyxy(np.array((tracked_object.tracked_box)))
            if tracked_object.lost_frames > 0:
                color = self.lost_color
            else:
                color = self.tracked_color
            cv2.rectangle(image_draw, bbox[:2], bbox[2:], color, thickness=1)
            if tracked_object.direction[0] != 0:
                if not tracked_object.direction_new:
                    color = self.lost_color
                else:
                    color = self.tracked_color
                cv2.line(image_draw, np.add(tracked_object.direction, tracked_object.tracked_box[:2]).astype(np.int32), np.add(tracked_object.direction*3, tracked_object.tracked_box[:2]).astype(np.int32), color, thickness=1)
        self.draw_controls(image_draw, ptz_values)
        self.image = image_draw

    def draw_controls(self, img, ptz_values):
        if img.any():
            draw_center = np.subtract((img.shape[1], img.shape[0]), (100, 70))
            size = 55
            black = (0,0,0)
            gray = (100,100,100)
            light_gray = (150,150,150)
            cv2.circle(img, draw_center, radius=size, color=light_gray, thickness=1)
            cv2.line(img, draw_center, (int(draw_center[0]+ptz_values[0]*size), draw_center[1]), color=gray, thickness=2)
            cv2.line(img, draw_center, (draw_center[0], int(draw_center[1]+ptz_values[1]*size)), color=gray, thickness=2)
            cv2.circle(img, draw_center, radius=0, color=black, thickness=2)
            cv2.line(img, (draw_center[0]+size, draw_center[1]), (int(draw_center[0]+size*1.2), draw_center[1]), color=light_gray, thickness=1)
            cv2.line(img, (int(draw_center[0]+size*1.1), draw_center[1]), (int(draw_center[0]+size*1.1), int(draw_center[1]+size*ptz_values[2])), color=gray, thickness=2)

class TrackingApp:
    def __init__(self, config):
        self.config = f'config{config}.json'
        try:
            with open(self.config, 'r') as file:
                self.data = json.load(file)
        except:
            with open(f'default_config.json', 'r') as file:
                self.data = json.load(file)
        self.name = self.data['cam']
        speed = self.data['speed']
        lock_hvz = [self.data['lock_horizontal'], self.data['lock_vertical'], self.data['lock_zoom']]
        center = np.array((320, 90))
        self.Vorschau = np.array((640, 352))
        self.stream = NDIstream(self.name, self.Vorschau)
        self.head_detector = YOLOv8('model/head_s_640.onnx', 0.3, 0.5)
        self.head_pose = YOLOv8('model/nose-pose19Ps.onnx', 0.6)
        self.move = CamController(self.stream, self.Vorschau, center, speed, lock_hvz)
        self.window = windowGUI(f'PTZ Tracker {self.name}', self.move)
        self.head_results = None
        self.tracked = None
        self.prep_thread = None
        self.pressed_key = 0

        self.record = False
        if self.record:
            self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
            self.video = cv2.VideoWriter(f'{self.name}.mp4', self.fourcc, fps=25, frameSize=self.Vorschau)
    
    def preprocess(self):
        frame = self.stream.get_frame()
        self.head_results, scores = self.head_detector.detection(self.stream.frame)       # cut to 640 x 352

    def update_mt(self, keys):
        self.pressed_keys = keys
        if self.tracked != None:
            if self.tracked.update(self.head_results, self.stream.frame) == 0:
                self.tracked = None
                self.move.stop()
                #self.stream.load_preset(20)
            else:
                self.move.update(self.tracked)
                self.tracked.direction = self.get_direction()
        ptz = self.stream.send_ptz_values()        
        self.window.update(self.stream.frame, self.head_results, self.tracked, ptz)

        if self.record:
            self.video.write(self.window.image)

        if 'x' in self.pressed_keys:
            self.tracked = None
            self.move.stop()
            #self.stream.load_preset(20)
        if 'l' in self.pressed_keys:
            self.stream.save_preset(20)
        self.move.control(self.pressed_keys)
        #self.auto_track()       # automatically track first found object
        return
    
    def close(self):
        self.data['cam'] = self.stream.name
        with open(self.config, "w") as json_file:
            json.dump(self.data, json_file, indent=4)
        self.stream.close()
        print('app close')

        if self.record:
            self.video.release()

    def find_head(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            #print('clicked',x,y)
            intersections = intersec([x,y,5,5], self.head_results)
            if np.max(intersections) != 0:
                index = np.argmax(intersections)
                self.tracked = tracked_obj(self.head_results[index], index)
    
    def find_head_kivy(self,x,y):   # x,y normalized
        #print('clicked',x,y)
        x *= self.Vorschau[0]
        y *= self.Vorschau[1]
        intersections = intersec([x,y,25,25], self.head_results)
        if np.max(intersections) != 0:
            index = np.argmax(intersections)
            self.tracked = tracked_obj(self.head_results[index], index)

    def auto_track(self):       # automatically track the first found object
        if self.tracked == None and self.head_results.any() != 0:
            self.tracked = tracked_obj(self.head_results[0], 0)
    
    def get_direction(self):
        if self.tracked.mosse == None:
            t_box = self.tracked.tracked_box
            zoom_factor = 1.2
            if t_box[2] > t_box[3]:
                size = int(t_box[2]*zoom_factor)
            else:
                size = int(t_box[3]*zoom_factor)
            pose_shape = [int(t_box[1]-size//2), int(t_box[1]+size//2), int(t_box[0]-size//2), int(t_box[0]+size//2), size]    # [y1:y2, x1:x2]
            frame_shape = self.stream.frame.shape
            if pose_shape[0] < 0:
                pose_shape[0] = 0
                pose_shape[1] = size
            elif pose_shape[1] > frame_shape[0]:
                pose_shape[1] = frame_shape[0]
                pose_shape[0] = frame_shape[0]-size
            if pose_shape[2] < 0:
                pose_shape[2] = 0
                pose_shape[3] = size
            elif pose_shape[3] > frame_shape[1]:
                pose_shape[3] = frame_shape[1]
                pose_shape[2] = frame_shape[1]-size
            image = self.stream.frame[pose_shape[0]:pose_shape[1], pose_shape[2]:pose_shape[3],:]       # crop[y1:y2, x1:x2]
            try:
                head_points, score = self.head_pose.detection(image)
            except:
                print('pose detection error !!!')
                head_points, score = (np.zeros((2,2)), 0)
            if len(head_points[0]) > 4:
                head_points = head_points[0]
                points = np.array((head_points[4::2], head_points[5::2]))
                points = points.transpose()
                direction = np.array(np.mean(points, axis=0))
                if all(direction > (0,0)) and all(direction < [pose_shape[4], pose_shape[4]]):
                    direction = np.add(direction, (pose_shape[2], pose_shape[0]))
                    self.tracked.direction_new = True
                    return np.subtract(direction, self.tracked.tracked_box[0:2])
        return self.tracked.direction

class kivyApp(App):
    def build(self):
        self.keys = []
        Window.size = (1280, 704)
        Window.bind(on_key_down=self.on_keyboard_down)
        Window.bind(on_key_up=self.on_keyboard_up)
        self.title = f'tracking App 3.10'
        layout = BoxLayout(orientation='vertical')

        # image widget
        bgr_image = cv2.imread('images/cat.jpg')
        # Convert BGR to RGB (Kivy uses RGB format)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        height, width, _ = rgb_image.shape
        self.frame = cv2.flip(rgb_image, 0)
        texture = Texture.create(size=(width, height), colorfmt='rgb')
        texture.blit_buffer(self.frame.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        self.image = AsyncImage(texture=texture, fit_mode='contain')
        self.image.bind(on_touch_down=self.on_click)

        # init ndi app
        self.dt = np.zeros(25)
        self.dt_new = 1
        self.tracker = TrackingApp(config=len([p for p in psutil.process_iter(['pid', 'name']) if p.info['name'] == "tracker.exe"]))
        self.tracker.preprocess()
        self.thread = Thread(target=self.loop, daemon=True)
        self.thread.start()
        Clock.schedule_interval(self.update_image, 1/27)

        # pack layout
        layout.add_widget(self.image)
        return layout
    
    def on_keyboard_down(self, instance, keyboard, keycode, text, modifiers):
        if not chr(keycode+93) in self.keys:
            self.keys.append(chr(keycode+93))
        return True

    def on_keyboard_up(self, instance, keyboard, keycode):
        self.keys.remove(chr(keycode+93))
        return True
    
    def on_stop(self):
        # Function to be called when the app is closing
        self.tracker.close()   
        print('kivy close')

    def on_click(self, instance, touch):
        if instance.collide_point(*touch.pos):
            # Check if the touch event corresponds to a left mouse click
            x, y = touch.pos
            #print('window size', Window.size)
            aspect_ratio = 9/16
            win_aspect_ratio = self.image.size[1] / self.image.size[0]
            if win_aspect_ratio > aspect_ratio:
                xn = x/self.image.size[0]
                height = self.image.size[0]*aspect_ratio
                yn = (y-(self.image.size[1]-height)/2) / height
            else:
                width = self.image.size[1]/aspect_ratio
                xn = (x-(self.image.size[0]-width)/2) / width
                yn = y/self.image.size[1]
            self.tracker.find_head_kivy(xn,1-yn)
    
    def update_image(self, dt):     # load new texture
        self.thread.join()
        if 'q' in self.keys:
            self.stop()
        self.dt_new = dt
        height, width, _ = self.frame.shape
        texture = Texture.create(size=(width, height), colorfmt='rgb')
        texture.blit_buffer(self.frame.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        self.image.texture = texture
        self.thread = Thread(target=self.loop, daemon=True)
        self.thread.start()

    def loop(self):   # update texture
        self.tracker.prep_thread = Thread(target=self.tracker.preprocess)
        self.tracker.prep_thread.start()
        self.tracker.update_mt(self.keys)
        # Display fps
        self.dt = np.append(self.dt[1:], self.dt_new)
        #print(self.dt)
        fps = len(self.dt)/sum(self.dt)
        self.tracker.move.fps = fps
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 0.6
        if fps > 23.5:
            font_color = (0, 255, 0)  # BGR color
        else:
            font_color = (0, 0, 255)
        font_thickness = 1
        text_position = (25, 30)
        text = f'FPS: {fps:.0f}'
        bgr_image = self.tracker.window.image
        cv2.putText(bgr_image, text, text_position, font, font_size, font_color, font_thickness)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        self.frame = cv2.flip(rgb_image, 0)
        self.tracker.prep_thread.join()

if __name__ == '__main__':
    kivyApp().run()
    print('Done')