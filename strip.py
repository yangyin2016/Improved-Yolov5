import torch
import sys

if __name__ == "__main__":
    if (len(sys.argv) == 1):
        print("usage:{} [模型路径]".format(sys.argv[0]))

    for i in  range(1, len(sys.argv)):
        model = torch.load(sys.argv[i])
        prefix, _ = sys.argv[i].split('.')
        torch.save(model['model'].state_dict(), prefix + ".pth")
