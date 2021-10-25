from models.common import *
import numpy as np
import torch
import torch.nn as nn

if __name__ == "__main__":
    model = ChannelAttention(3)

    inputs = torch.Tensor(np.zeros((1, 3, 64, 64)))
    outpus = model(inputs)
    print(outpus.shape)
