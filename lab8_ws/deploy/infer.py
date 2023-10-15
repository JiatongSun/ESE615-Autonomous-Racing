import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import cv2
from time import time

from helper import *

TRT_FILE_PATH = "../model/trt.engine"
IMG_FILE_PATH = "car.jpg"


def load_engine(trt_runtime, engine_path):
    trt.init_libnvinfer_plugins(None, "")
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine


def preprocess_image(img_path):
    img = cv2.imread(img_path) / 255.0
    img = cv2.resize(img, (input_dim[1], input_dim[0]))
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def postprocess(result):
    voting_iou_threshold = 0.5
    confi_threshold = 0.4

    bboxs, result_prob = label_to_box_xyxy(result[0], confi_threshold)
    vote_rank = voting_suppression(bboxs, voting_iou_threshold)
    bbox = bboxs[vote_rank[0]]
    [c_x, c_y, w, h] = bbox_convert_r(bbox[0], bbox[1], bbox[2], bbox[3])
    return np.array([[c_x, c_y, w, h]])


def trt_infer():
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = load_engine(runtime, TRT_FILE_PATH)
    context = engine.create_execution_context()

    # get sizes of input and output and allocate memory required for input data and for output data
    for binding in engine:
        if engine.binding_is_input(binding):  # we expect only one input
            input_shape = engine.get_binding_shape(binding)
            input_size = trt.volume(input_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize  # in bytes
            device_input = cuda.mem_alloc(input_size)
        else:  # and one output
            output_shape = engine.get_binding_shape(binding)
            # create page-locked memory buffers (i.e. won't be swapped to disk)
            host_output = cuda.pagelocked_empty(trt.volume(output_shape) * engine.max_batch_size, dtype=np.float32)
            device_output = cuda.mem_alloc(host_output.nbytes)

    # create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()

    # preprocess input data
    host_input = np.array(preprocess_image(IMG_FILE_PATH), dtype=np.float32, order='C')
    start_time = time()
    cuda.memcpy_htod_async(device_input, host_input, stream)

    # run inference
    context.execute_async(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_output, device_output, stream)
    stream.synchronize()

    # postprocess results
    output_data = host_output.reshape(output_shape)
    end_time = time()
    bbox = postprocess(output_data)

    print("Infer time: {}s".format(round(end_time - start_time, 6)))

    # display result
    DisplayLabel(np.transpose(host_input[0], (1, 2, 0)), bbox)

    return bbox_convert(bbox[0][0], bbox[0][1], bbox[0][2], bbox[0][3])


if __name__ == "__main__":
    bbox = trt_infer()