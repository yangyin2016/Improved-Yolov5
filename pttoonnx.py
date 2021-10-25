import torch
from models.common import *

class test(nn.Module):
    def __init__(self, channel, reduction=1):
        super(test, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid())
        self.fc2 = nn.Sequential(
            nn.Conv2d(channel , channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel , channel // reduction, 1, bias=False),
            nn.Sigmoid()
        )
        self.fc1[0].weights = nn.Parameter(torch.Tensor([channel, channel / reduction, 1, 1]))
        self.fc1[1].weights = nn.Parameter(torch.Tensor([channel / reduction, channel, 1, 1]))

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) # 全连接层只接受2维的输入
        y = self.fc1(y).view(b, c, 1, 1)
        return y
        #return x * y
 

#torch_model = SENetBottleneck(3, 3) 					# 由研究员提供python.py文件
torch_model = test(3, 1)

batch_size = 1 								# 批处理大小
input_shape = (3, 640, 640) 				# 输入数据

# set the model to inference mode
torch_model.eval()

x = torch.randn(batch_size,*input_shape) 	# 生成张量
export_onnx_file = "bottleneck.onnx"
torch.onnx.export(torch_model,
                    x,
                    export_onnx_file,
                    opset_version=10,
                    do_constant_folding=True,	# 是否执行常量折叠优化
                    input_names=["input"],		# 输入名
                    output_names=["output"],	# 输出名
                    dynamic_axes={"input":{0:"batch_size"},	# 批处理变量
                                    "output":{0:"batch_size"}})


