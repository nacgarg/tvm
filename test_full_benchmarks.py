import numpy as np
from tvm import relay
from tvm.relay import testing
import tvm
from tvm import te
from tvm.contrib import graph_executor
import tvm.testing
import scipy
import time
import tensorflow as tf
import torch.nn.functional as F
import torch
import cv2
import tensorflow_addons as tfa
import torchvision.transforms as T
import skimage.color
import sys
import matplotlib.pyplot as plt
from graphviz import Digraph

current_op = ""

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Insert a channel dimension
    img = np.expand_dims(img, axis=2)
    return img

def visualize_relay(mod, filename):
    def _traverse_expr(node, node_dict):
        if node in node_dict:
            return
        if isinstance(node, tvm.ir.op.Op):
            return 
        node_dict[node] = len(node_dict)

    dot = Digraph(format='svg')
    dot.attr(rankdir='BT')
    dot.attr('node', shape='box')

    node_dict = {}
    relay.analysis.post_order_visit(mod['main'], lambda node: _traverse_expr(node, node_dict))
    for node, node_idx in node_dict.items():
        if isinstance(node, relay.Var):
            print(f'node_idx: {node_idx}, Var(name={node.name_hint}, type=Tensor[{tuple(node.type_annotation.shape)}, {node.type_annotation.dtype}])')
            dot.node(str(node_idx), f'{node.name_hint}:\nTensor[{tuple(node.type_annotation.shape)}, {node.type_annotation.dtype}]')
        elif isinstance(node, relay.Call):
            args = [node_dict[arg] for arg in node.args]
            print(f'node_idx: {node_idx}, Call(op_name={node.op.name}, args={args})')
            dot.node(str(node_idx), f'Call(op={node.op.name})')
            for arg in args:
                dot.edge(str(arg), str(node_idx))
        elif isinstance(node, relay.Function):
            print(f'node_idx: {node_idx}, Function(body={node_dict[node.body]})')
            dot.node(str(node_idx), f'Function')
            dot.edge(str(node_dict[node.body]), str(node_idx))
        elif isinstance(node, relay.TupleGetItem):
            print(f'node_idx: {node_idx}, TupleGetItem(tuple={node_dict[node.tuple_value]}, idx={node.index})')
            dot.node(str(node_idx), f'TupleGetItem(idx={node.index})')
            dot.edge(str(node_dict[node.tuple_value]), str(node_idx))
        elif isinstance(node, relay.expr.Constant):
            # Show constant shape
            print(f'node_idx: {node_idx}, Constant(shape={node.data.shape}, dtype={node.data.dtype})')
            dot.node(str(node_idx), f'Constant(shape={node.data.shape}, dtype={node.data.dtype})')
        elif isinstance(node, relay.expr.Tuple):
            print(f'node_idx: {node_idx}, Tuple(fields={node.fields})')
            dot.node(str(node_idx), f'Tuple')
            for field in node.fields:
                dot.edge(str(node_dict[field]), str(node_idx))
        else:
            raise RuntimeError(f'Unknown node type. node_idx: {node_idx}, node: {type(node)}')

    # Save dot
    dot.render(filename)

def make_random_image(shape):
    """
    Make a random image.
    """
    img = np.random.rand(*shape).astype("float32")
    return img

def time_runs(func, num_runs):
    start = time.time()
    for i in range(num_runs):
        func()
    end = time.time()
    return (end - start)/num_runs

def sobel_model(shape_a):
    """
    Model for sobel.
    """
    img_in = relay.var("input", shape=shape_a)
    #
    out = relay.image.sobel(img_in, (3, 3), layout="NCHW", out_dtype="float32", padding="SAME")
    func = relay.Function([img_in], out)
    return func

def gaussian_model(shape_a, sigma):
    """
    Model for Gaussian.
    """
    img_in = relay.var("input", shape=shape_a)
    #
    out = relay.image.gaussian_blur(img_in, sigma=sigma)
    func = relay.Function([img_in], out)
    return func

def colorspace_model(shape_a, out_mode):
    """
    Model for colorspace transform.
    """
    img_in = relay.var("input", shape=shape_a)
    #
    out = relay.image.color_space(img_in, "RGB", out_mode)
    func = relay.Function([img_in], out)
    return func


def run_tvm(image, model, num_times):
    image_shape = image.shape
    # model = sobel_model(image_shape)
    fix_sobel = relay.transform.LowerSobel()
    mod = tvm.IRModule.from_expr(model)
    visualize_relay(mod, current_op+'_before')
    model = fix_sobel(mod)
    visualize_relay(model, current_op+'_after')

    params = {}

    target = "llvm"
    ctx = tvm.cpu(0)
    dtype = "float32"
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(model, target, params=params)
        # print(lib)
        
    dev = tvm.cpu()

    module = graph_executor.GraphModule(lib["default"](dev))

    # module.set_input("A", image)
    # module.run()
    # Duplicate image 100 times in batch dimension

    module.set_input("input", image)
    tvm_time = time_runs(lambda: module.run(), num_times)

    tvm_output = module.get_output(0).asnumpy()[0:1]
    return tvm_time, tvm_output
    

# ------------------------- SOBEL BENCHMARKS ------------------------#
def tensorflow_sobel(image):
    image = tf.convert_to_tensor(image)
    image = tf.transpose(image, [0, 2, 3, 1])
    sobel = tf.image.sobel_edges(image)
    return tf.norm(sobel, axis=-1)
    sobel_y = np.asarray(sobel[0, :, :, :, 0])
    sobel_x = np.asarray(sobel[0, :, :, :, 1])
    output = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    return output

def scipy_sobel(image):
    x_sobel = scipy.ndimage.sobel(image.copy(), axis=0)
    y_sobel = scipy.ndimage.sobel(image.copy(), axis=1)
    sp_sobel = np.hypot(x_sobel, y_sobel)
    return sp_sobel

def pytorch_sobel(image):
    with torch.no_grad():
        kernel_v = [[0, -1, 0],[0, 0, 0],[0, 1, 0]]
        kernel_h = [[0, 0, 0],[-1, 0, 1],[0, 0, 0]]
        sobel_kernel_x = torch.FloatTensor(kernel_v).reshape((1,1,3,3))
        sobel_kernel_y = torch.FloatTensor(kernel_h).reshape((1,1,3,3))
        image = torch.tensor(image)
        # conv2 = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0)
        # conv2(torch.from_numpy(image_d))

        x_v = F.conv2d(image, sobel_kernel_x, padding=0)
        y_h = F.conv2d(image, sobel_kernel_y, padding=0)

        x_i = torch.sqrt(torch.pow(x_v, 2) + torch.pow(y_h, 2) + 1e-6)
        return x_i
    
def opencv_sobel(image):
    dx = cv2.Sobel(image[0,0], cv2.CV_32F, 0, 1, ksize=3)
    dy = cv2.Sobel(image[0,0], cv2.CV_32F, 1, 0, ksize=3)
    return np.hypot(dx, dy)

# --------------------- GAUSSIAN BENCHMARKS ------------------------#
def _gaussian_kernel(kernel_size, sigma, n_channels, dtype):
    x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype)
    g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, dtype), 2))))
    g_norm2d = tf.pow(tf.reduce_sum(g), 2)
    g_kernel = tf.tensordot(g, g, axes=0) / g_norm2d
    g_kernel = tf.expand_dims(g_kernel, axis=-1)
    return tf.expand_dims(tf.tile(g_kernel, (1, 1, n_channels)), axis=-1)

def tensorflow_gaussian(image, sigma):
    img = tf.convert_to_tensor(image)
    # blur = _gaussian_kernel(5, sigma, 3, image.dtype)
    # img = tf.nn.depthwise_conv2d(img[None], blur, [1,1,1,1], 'SAME')
    return tfa.image.gaussian_filter2d(img, sigma=sigma)

def scipy_gaussian(image, sigma):
    # blur = np.asarray(_gaussian_kernel(5, sigma, 3, image.dtype))
    # scipy.ndimage.convolve(image, blur, mode='constant', cval=1.0)
    return scipy.ndimage.gaussian_filter(input = image, sigma = 1)

def opencv_gaussian(image, sigma):
    image = np.transpose(image, [0, 2, 3, 1])
    return cv2.GaussianBlur(src=image[0], ksize=(5,5), sigmaX=1, sigmaY=1)
    
def pytorch_gaussian(image, sigma):
    with torch.no_grad():
        image = torch.tensor(image)
        blurrer = T.GaussianBlur(kernel_size=(5, 5), sigma=(1,1))
        return blurrer(image)

# --------------------- COLORSPACE BENCHMARKS ------------------------#

def opencv_colorspace(image, out_mode):
    image = np.transpose(image, [0, 2, 3, 1])[0]
    if out_mode == "GRAY":
        return cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    if out_mode == "HSV":
        return cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    if out_mode == "YUV":
        return cv2.cvtColor(image,cv2.COLOR_RGB2YUV)

def scipy_colorspace(image, out_mode):
    image = np.transpose(image, [0, 2, 3, 1])
    if out_mode == "GRAY":
        skimage.color.rgb2gray(image)
    if out_mode == "HSV":
        skimage.color.rgb2hsv(image)
    if out_mode == "YUV":
        skimage.color.rgb2yuv(image)
    
    
def pytorch_colorspace(image, out_mode):
    return 0 # sadge

def tensorflow_colorspace(image, out_mode):
    with torch.no_grad():
        img = tf.convert_to_tensor(image)
        img = tf.transpose(img, [0, 2, 3, 1])
        if out_mode == "GRAY":
            return tf.image.rgb_to_grayscale(img)
        if out_mode == "HSV":
            return tf.image.rgb_to_hsv(img)
        if out_mode == "YUV":
            return tf.image.rgb_to_yuv(img)

# --------------------- TESTING INFRASTRUCTURE ------------------------#

def run_test(operation, params, img, num_times):
    global current_op
    print("Testing {}...".format(operation))
    model = eval('{}_model'.format(operation))
    current_op = operation + "_" + str(params[0]) if len(params) > 0 else operation

    tvm_time, tvm_output = run_tvm(img, model(img.shape, *params), num_times)
    times = [tvm_time]

    for framework in ["scipy", "opencv", "pytorch", "tensorflow"]:
        print("Running {}...".format(framework))
        def run_framework():
            return eval('{}_{}'.format(framework, operation))(img, *params)
        times.append(time_runs(run_framework, num_times))
        
    return times, tvm_output


def format_times(times):
    if times[0] < 1e-6:
        times = [t * 1e9 for t in times]
        return "{:<20} {:<20} {:<20} {:<20} {:<20}".format(*[f"{t:.3f} ns" for t in times])
    if times[0] < 1:
        times = [t * 1000 for t in times]
        return "{:<20} {:<20} {:<20} {:<20} {:<20}".format(*[f"{t:.3f} ms" for t in times])
    else:
        return "{:<20} {:<20} {:<20} {:<20} {:<20}".format(*[f"{t:.3f} s" for t in times])
    
def print_results(sizes, times):
    print("{:<20} {:<20} {:<20} {:<20} {:<20} {:<20}".format("Input Shape", "TVM", "Scipy", "OpenCV", "PyTorch", "TensorFlow"))
    for i in range(len(times)):
        print("{:<20} ".format("{}x{}".format(*sizes[i])) + format_times(times[i]))
   


def main():
    #testing against scipy, cv2, tensorflow, pytorch
    #sizes: 64 by 64, 320 by 240, 1920 by 1080, 3840 by 2160, 8700 by 5800
    #colorspace: rgb to grayscale, rbg to hsv, rbg to yuv


    sizes = [(64, 64), (320, 240), (1920, 1080), (3840, 2160), (8700, 5800)]

    input_image = None
    if len(sys.argv) == 2:
        input_image = sys.argv[1]
        sizes = [0]
    
    sobel_times = []
    gaussian_times = []
    rgb2gray_times = []
    rgb2hsv_times = []
    rgb2yuv_times = []
    for size in range(len(sizes)):
        if (input_image is None):
            img = make_random_image((1, 3, sizes[size][0], sizes[size][1]))
            num_times = int(100 * np.exp(-(size/2)))
            gray_img = img[:,0:1,:,:]
        else:
            img = cv2.imread(input_image)
            img = img.astype(np.float32)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.transpose(img, [2, 0, 1])
            img = np.expand_dims(img, axis=0)
            gray_img = np.expand_dims(gray_img, axis=0)
            gray_img = np.expand_dims(gray_img, axis=0)
            print(img.shape, gray_img.shape)
            num_times = 1
            sizes[0] = img.shape[2:]
        
        sobel_time, sobel_out = run_test("sobel", (), gray_img, num_times)
        sobel_times.append(sobel_time)
        gaussian_time, gaussian_out = run_test("gaussian", (1,), img, num_times)
        gaussian_times.append(gaussian_time)
        rgb2gray_time, rgb2gray_out = run_test("colorspace", ("GRAY",), img, num_times)
        rgb2gray_times.append(rgb2gray_time)
        rgb2hsv_time, rgb2hsv_out = run_test("colorspace", ("HSV",), img, num_times)
        rgb2hsv_times.append(rgb2hsv_time)
        rgb2yuv_time, rgb2yuv_out = run_test("colorspace", ("YUV",), img, num_times)
        rgb2yuv_times.append(rgb2yuv_time)

        if (input_image is not None):
            # Use matplotlib to display the images. They are in NCHW format, so they need to be converted to HWC.
            # Also, for color space transforms, show each channel separately.
            plt.subplot(2, 4, 1)
            plt.imshow(np.transpose(img[0,:,:,:], [1, 2, 0])/255)
            plt.title("Input")
            plt.subplot(2, 4, 2)
            plt.imshow(np.transpose(sobel_out[0,:,:,:], [1, 2, 0]), cmap='gray')
            plt.title("Sobel")
            plt.subplot(2, 4, 3)
            plt.imshow(np.transpose(gaussian_out[0,:,:,:], [1, 2, 0])/255)
            plt.title("Gaussian")
            plt.subplot(2, 4, 4)
            plt.imshow(np.transpose(rgb2gray_out[0,:,:,:], [1, 2, 0]), cmap='gray')
            plt.title("RGB to Gray")

            # Show each HSV channel separately.
            plt.subplot(2, 4, 5)
            plt.imshow(np.transpose(rgb2hsv_out[0,0:1,:,:], [1, 2, 0]), cmap='gray')
            plt.title("HSV Hue Channel")
            plt.subplot(2, 4, 6)
            plt.imshow(np.transpose(rgb2hsv_out[0,1:2,:,:], [1, 2, 0]), cmap='gray')
            plt.title("HSV Saturation Channel")
            plt.subplot(2, 4, 7)
            plt.imshow(np.transpose(rgb2hsv_out[0,2:3,:,:], [1, 2, 0]), cmap='gray')
            plt.title("HSV Value Channel")
            plt.show()


    print("Sobel Benchmark Results:")
    print_results(sizes, sobel_times)
    print()
    print("Gaussian Benchmark Results:")
    print_results(sizes, gaussian_times)
    print()
    print("Colorspace Benchmark Results:")
    print("RGB to Grayscale:")
    print_results(sizes, rgb2gray_times)
    print()
    print("RGB to HSV:")
    print_results(sizes, rgb2hsv_times)
    print()
    print("RGB to YUV:")
    print_results(sizes, rgb2yuv_times)

if __name__ == "__main__":
    main()
    
