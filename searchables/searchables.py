import random


def EfficientNetB0Conf(d):
    num_blocks = 7
    strides = [1, 2, 2, 2, 1, 2, 1]
    ich = [32, 16, 24, 40, 80, 112, 192]
    och = [16, 24, 40, 80, 112, 192, 320]
    # kchoices = [3, 3, 5, 3, 5, 5, 3]
    kchoices = [3, 3, 5, 3, 5, 5, 3]
    echoices = [1, 6, 6, 6, 6, 6, 6]
    # blocktypes = ['MB', 'FMB']
    if d == 1:
        lchoices = [1, 2, 2, 3, 3, 4, 1]
    elif d == 0.5:
        lchoices = [1, 1, 1, 2, 2, 2, 1]
    else:
        raise ValueError(f"Depth d={d} unsupported...")
    # lchoices = [3, 3, 3, 3, 3, 3, 3]
    layer_confs = []
    bchoices = ["MB", "MB", "MB", "MB", "MB", "MB", "MB"]
    # bchoices = random.choices(blocktypes, k=num_blocks)

    for i in range(num_blocks):
        conf = [
            bchoices[i],
            echoices[i],
            kchoices[i],
            strides[i],
            ich[i],
            och[i],
            lchoices[i],
        ]
        layer_confs.append(conf)
    return layer_confs


def RandomSearchable():
    num_blocks = 7
    strides = [1, 2, 2, 2, 1, 2, 1]
    ich = [32, 16, 24, 40, 80, 112, 192]
    och = [16, 24, 40, 80, 112, 192, 320]
    blocktypes = ["MB"]
    kernel_sizes = [3, 5]
    exp = [1, 4, 6]
    layers = [1, 2, 3]
    layer_confs = []
    bchoices = random.choices(blocktypes, k=num_blocks)
    kchoices = random.choices(kernel_sizes, k=num_blocks)
    echoices = random.choices(exp, k=num_blocks)
    lchoices = random.choices(layers, k=num_blocks)
    # echoices[0] = 1
    # lchoices[0] = 1
    # lchoices = []
    # for i in range(num_blocks):
    #    l = random.choice(layers)
    #    if l != layers[0]:
    #        lind = layers.index(l)
    #        del layers[:lind]
    #    lchoices.append(l)

    for i in range(num_blocks):
        conf = [
            bchoices[i],
            echoices[i],
            kchoices[i],
            strides[i],
            ich[i],
            och[i],
            lchoices[i],
        ]
        layer_confs.append(conf)
    return layer_confs


def TestSearchable():
    num_blocks = 7
    strides = [1, 2, 2, 2, 1, 2, 1]
    ich = [32, 16, 24, 40, 80, 112, 192]
    och = [16, 24, 40, 80, 112, 192, 320]
    blocktypes = ["MB"]
    layer_confs = []
    bchoices = random.choices(blocktypes, k=num_blocks)
    kchoices = [3, 3, 3, 5, 5, 3, 5]
    echoices = [6, 6, 6, 4, 4, 6, 6]
    lchoices = [3, 1, 3, 3, 2, 2, 1]
    for i in range(num_blocks):
        conf = [
            bchoices[i],
            echoices[i],
            kchoices[i],
            strides[i],
            ich[i],
            och[i],
            lchoices[i],
        ]
        layer_confs.append(conf)
    return layer_confs
