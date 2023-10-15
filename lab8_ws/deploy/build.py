import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt

FP16 = True

ONNX_FILE_PATH = "../model/yolo.onnx"
TRT_FILE_PATH = "../model/trt.engine"

# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger()


def build_engine(onnx_file_path, mode):
    # initialize TensorRT engine and parse ONNX model
    builder = trt.Builder(TRT_LOGGER)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # allow TensorRT to use up to 1GB of GPU memory for tactic selection
    builder.max_workspace_size = 1 << 30
    # we have only one image in batch
    builder.max_batch_size = 1
    # use FP16 mode if possible
    if mode and builder.platform_has_fast_fp16:
        builder.fp16_mode = True
        print("Using FP16 Mode")
    else:
        builder.fp16_mode = False
        print("Using FP32 Mode")

    # parse ONNX
    with open(onnx_file_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        parser.parse(model.read())
    print('Completed parsing of ONNX file')

    # generate TensorRT engine optimized for the target platform
    print('Building an engine...')
    engine = builder.build_cuda_engine(network)
    context = engine.create_execution_context()
    print("Completed creating Engine")
    return engine, context


def main():
    # initialize TensorRT engine and parse ONNX model
    engine, context = build_engine(ONNX_FILE_PATH, FP16)

    # save engine
    with open(TRT_FILE_PATH, "wb") as f:
        f.write(engine.serialize())


if __name__ == '__main__':
    main()
