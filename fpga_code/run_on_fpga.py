"""
Copyright 2020 Xilinx Inc.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from ctypes import *
from typing import List
import cv2
import numpy as np
import vart
import os
import pathlib
import xir
import threading
import time
import sys
import argparse

_divider = "-------------------------------"


def preprocess_fn_opencv(image_path, fix_scale=None):
    # Getting input image
    means = [0.485, 0.456, 0.406] #RGB Format asdjkhaskdjahkfjhasjfkhakjdhfaskjfhaksdjfh askjdhaskjdhkj askdjhasjkdhk kjhk jahkjh
    scales = [0.229, 0.224, 0.225]
    input_size = 224
    crop_pct = 224 / 256
    size = int(input_size / crop_pct)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.
    R, G, B = cv2.split(image)
    if fix_scale is not None:
        R = ((R - means[0]) / scales[0]) * fix_scale
        G = ((G - means[1]) / scales[1]) * fix_scale
        B = ((B - means[2]) / scales[2]) * fix_scale
        image = cv2.merge([R, G, B])
        image = image.astype(np.int8)
    else:
        R = ((R - means[0]) / scales[0])
        G = ((G - means[1]) / scales[1])
        B = ((B - means[2]) / scales[2])
        image = cv2.merge([R, G, B])

    # Getting target label
    fname = image_path.split('/')[-1]
    cls = int(fname.split('_')[0])
    return image, cls


def preprocess_fn(data_path, fix_scale=None):
    from torchvision_dataloader import build_torchvision_loader
    valid_queue, _ = build_torchvision_loader(
        data_path, batch_size=1, num_workers=8, subset_len=None
    )
    images = []
    targets = []
    for step, (input, target) in enumerate(valid_queue):
        if fix_scale is not None:
            image = input.numpy().squeeze() * fix_scale
            image = np.moveaxis(image, 0, -1)
            images.append(image.astype(np.int8))
        else:
            image = input.numpy().squeeze()
            image = np.moveaxis(image, 0, -1)
            images.append(image)

        targ = target.numpy()
        targets.append(targ[0])

    return images, targets


def obtain_acc(targets, out_q):
    assert len(targets) == len(out_q)
    correct = 0
    wrong = 0
    for i in range(len(out_q)):
        if out_q[i] == targets[i]:
            correct += 1
        else:
            wrong += 1
    accuracy = correct / len(targets)
    return correct, wrong, accuracy


def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (
        root_subgraph is not None
    ), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]


def runDPU(id, start, dpu, img):
    """get tensor"""
    inputTensors = dpu.get_input_tensors()
    outputTensors = dpu.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)
    output_ndim = tuple(outputTensors[0].dims)

    # we can avoid output scaling if use argmax instead of softmax
    # output_fixpos = outputTensors[0].get_attr("fix_point")
    # output_scale = 1 / (2**output_fixpos)

    batchSize = input_ndim[0]
    n_of_images = len(img)
    count = 0
    write_index = start
    ids = []
    ids_max = 10
    outputData = []
    for i in range(ids_max):
        outputData.append([np.empty(output_ndim, dtype=np.int8, order="C")])
    while count < n_of_images:
        if count + batchSize <= n_of_images:
            runSize = batchSize
        else:
            runSize = n_of_images - count

        """prepare batch input/output """
        inputData = []
        inputData = [np.empty(input_ndim, dtype=np.int8, order="C")]

        """init input image to input buffer """
        for j in range(runSize):
            imageRun = inputData[0]
            imageRun[j, ...] = img[(count + j) % n_of_images].reshape(input_ndim[1:])
        """run with batch """
        job_id = dpu.execute_async(inputData, outputData[len(ids)])
        ids.append((job_id, runSize, start + count))
        count = count + runSize
        if count < n_of_images:
            if len(ids) < ids_max - 1:
                continue
        for index in range(len(ids)):
            dpu.wait(ids[index][0])
            write_index = ids[index][2]
            """store output vectors """
            for j in range(ids[index][1]):
                # we can avoid output scaling if use argmax instead of softmax
                # out_q[write_index] = np.argmax(outputData[0][j] * output_scale)
                out_q[write_index] = np.argmax(outputData[index][0][j])
                write_index += 1
        ids = []


def obtain_images_targets(data_path, dataloader):
    if dataloader == 'torchvision':
        images, targets = preprocess_fn(data_path)
    elif dataloader == 'opencv':
        listimage = os.listdir(data_path)
        runTotal = len(listimage)
        images, targets = [], []
        for i in range(runTotal):
            path = os.path.join(data_path, listimage[i])
            image, target = preprocess_fn_opencv(path)
            images.append(image)
            targets.append(target)
    else:
        raise ValueError(f'Dataloader {dataloader} not supported...')

    return images, targets


def execute_on_fpga(images, targets, data_path, threads, model, dataloader):
    g = xir.Graph.deserialize(model)
    subgraphs = get_child_subgraph_dpu(g)
    all_dpu_runners = []
    for i in range(threads):
        all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))

    # input scaling
    input_fixpos = all_dpu_runners[0].get_input_tensors()[0].get_attr("fix_point")
    input_scale = 2**input_fixpos
    # input_scale = 1
    '''
    for image in images:
        image = image * input_scale
        image = image.astype(np.int8)
    '''
    """ preprocess images """
    print(_divider)
    print("Pre-processing dataset")
    if dataloader == 'torchvision':
        images, targets = preprocess_fn(data_path, input_scale)
    elif dataloader == 'opencv':
        listimage = os.listdir(data_path)
        runTotal = len(listimage)
        images, targets = [], []
        for i in range(runTotal):
            path = os.path.join(data_path, listimage[i])
            image, target = preprocess_fn_opencv(path, input_scale)
            images.append(image)
            targets.append(target)

    global out_q
    out_q = [None] * len(images)

    """run threads """
    print(_divider)
    print("Starting", threads, "threads...")
    threadAll = []
    start = 0
    for i in range(threads):
        if i == threads - 1:
            end = len(images)
        else:
            end = start + (len(images) // threads)
        in_q = images[start:end]
        t1 = threading.Thread(target=runDPU, args=(i, start, all_dpu_runners[i], in_q))
        threadAll.append(t1)
        start = end

    time1 = time.time()
    for x in threadAll:
        x.start()
    for x in threadAll:
        x.join()
    time2 = time.time()
    timetotal = time2 - time1

    fps = float(len(images) / timetotal)
    print(_divider)
    print(
        "Throughput=%.2f fps, total frames = %.0f, time=%.4f seconds"
        % (fps, len(images), timetotal)
    )

    """ post-processing """
    correct, wrong, accuracy = obtain_acc(targets, out_q) 
    print("Correct:%d, Wrong:%d, Accuracy:%.4f" % (correct, wrong, accuracy))
    print(_divider)
    return


# only used if script is run as 'main' from command line
def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-d",
        "--image_dir",
        type=str,
        default="/workspace/imagenet/val",
        help="Path to folder of images. Default is images",
    )
    ap.add_argument(
        "-t", "--threads", type=int, default=1, help="Number of threads. Default is 1"
    )
    ap.add_argument(
        "-m",
        "--model",
        type=str,
        help="Path of xmodel",
    )
    ap.add_argument(
        "--dataloader",
        type=str,
        default="torchvision",
        help="Dataloader to use: Options are [torchvision, opencv]"
    )
    args = ap.parse_args()

    print("Command line options:")
    print(" --image_dir : ", args.image_dir)
    print(" --threads   : ", args.threads)
    print(" --model     : ", args.model)
    
    print(_divider)
    print("Pre-processing dataset")
    images, targets = obtain_images_targets(args.image_dir, args.dataloader)
    execute_on_fpga(images, targets, args.image_dir, args.threads, args.model, args.dataloader)

if __name__ == "__main__":
    main()
