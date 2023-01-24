import random


def EffNetB0Conf():
    num_blocks = 7
    strides = [1, 2, 2, 2, 1, 2, 1]
    ich = [32, 16, 24, 40, 80, 112, 192]
    och = [16, 24, 40, 80, 112, 192, 320]
    blocktypes = ["MB"]
    kernel_sizes = [3, 5]
    exp = [1, 4, 6]
    layers = [1, 2, 3]
    squeeze_excite_options = [True, False]
    layer_confs = []
    bchoices = ['MB' for i in range(num_blocks)]
    kchoices = [3,3,5,3,5,5,3]
    echoices = [1,6,6,6,6,6,6]
    lchoices = [1,2,2,3,3,4,1]
    sechoices = [True for i in range(num_blocks)]
    # bchoices = random.choices(blocktypes, k=num_blocks)
    #kchoices = random.choices(kernel_sizes, k=num_blocks)
    #echoices = random.choices(exp, k=num_blocks)
    #lchoices = random.choices(layers, k=num_blocks)
    #sechoices = random.choices(squeeze_excite_options, k=num_blocks)
    for i in range(num_blocks):
        conf = [
            bchoices[i],
            echoices[i],
            kchoices[i],
            strides[i],
            ich[i],
            och[i],
            lchoices[i],
            sechoices[i],
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
    squeeze_excite_options = [True, False]
    layer_confs = []
    bchoices = random.choices(blocktypes, k=num_blocks)
    kchoices = random.choices(kernel_sizes, k=num_blocks)
    echoices = random.choices(exp, k=num_blocks)
    lchoices = random.choices(layers, k=num_blocks)
    sechoices = random.choices(squeeze_excite_options, k=num_blocks)
    for i in range(num_blocks):
        conf = [
            bchoices[i],
            echoices[i],
            kchoices[i],
            strides[i],
            ich[i],
            och[i],
            lchoices[i],
            sechoices[i],
        ]
        if i == 5:
            conf[6] = 4
        layer_confs.append(conf)
    return layer_confs


def CustomSearchable(e, k, la, se):
    num_blocks = 7
    strides = [1, 2, 2, 2, 1, 2, 1]
    ich = [32, 16, 24, 40, 80, 112, 192]
    och = [16, 24, 40, 80, 112, 192, 320]
    blocktypes = ["MB"]
    layer_confs = []
    bchoices = random.choices(blocktypes, k=num_blocks)
    kchoices = k
    echoices = e
    lchoices = la
    sechoices = se
    for i in range(num_blocks):
        conf = [
            bchoices[i],
            echoices[i],
            kchoices[i],
            strides[i],
            ich[i],
            och[i],
            lchoices[i],
            sechoices[i],
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
