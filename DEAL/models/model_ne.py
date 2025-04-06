import torch
import torch.nn as nn

class NE(nn.Module):
    def __init__(self):
        super(NE, self).__init__()

        self.input_conv = ReflectiveConvLayer(3, 32, kernel_size=3, stride=1)
        self.feature_conv = ReflectiveConvLayer(32, 32, kernel_size=3, stride=1)
        self.refine_conv = ReflectiveConvLayer(32, 32, kernel_size=3, stride=1)
        self.output_conv = ReflectiveConvLayer(32, 3, kernel_size=3, stride=1)

        self.scale_block1 = ScaleTransformBlock(k1_size=7, k2_size=5, dilation=1)
        self.spike_block1 = SpikeSeparationBlock(k1_size=7, k2_size=5, dilation=2)
        self.spike_block2 = SpikeSeparationBlock(k1_size=11, k2_size=7, dilation=2)
        self.scale_block2 = ScaleTransformBlock(k1_size=11, k2_size=5, dilation=1)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.relu(self.input_conv(x))
        out = self.relu(self.feature_conv(out))
        res = out

        out, res = self.scale_block1(out, res)
        out, res = self.spike_block1(out, res)
        out, res = self.spike_block2(out, res)
        out, res = self.scale_block2(out, res)

        out = self.relu(self.refine_conv(out))
        out = self.tanh(self.output_conv(out))
        out = out + x

        return out

class SpikeSeparationBlock(nn.Module):
    def __init__(self, in_dim=32, out_dim=32, res_dim=32, k1_size=3, k2_size=1, dilation=1, with_relu=True):
        super(SpikeSeparationBlock, self).__init__()

        self.conv1 = ReflectiveConvLayer(in_dim, in_dim, 3, 1)
        self.conv2 = ReflectiveConvLayer(in_dim, in_dim, 3, 1)

        self.up_conv = ReflectiveConvLayer(in_dim, res_dim, kernel_size=k1_size, stride=1, dilation=dilation)

        self.spiking_conv = SpikingConvLayer(32, 32)

        self.down_conv = ReflectiveConvLayer(res_dim, out_dim, kernel_size=k2_size, stride=1)

        self.with_relu = with_relu
        self.relu = nn.ReLU()

    def forward(self, x, res):
        x_r = x

        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        x += x_r
        x = self.relu(x)

        x = self.up_conv(x)
        x += res
        x = self.relu(x)
        res = x

        x = self.down_conv(x)
        if len(x.shape) < 5:
            x = (x.unsqueeze(0)).repeat(4, 1, 1, 1, 1)
        x = self.spiking_conv(x)
        x = x.mean(dim=0)
        x += x_r

        if self.with_relu:
            x = self.relu(x)
        else:
            pass

        return x, res

class ScaleTransformBlock(nn.Module):
    def __init__(self, in_dim=32, out_dim=32, res_dim=32, k1_size=3, k2_size=1, dilation=1, with_relu=True):
        super(ScaleTransformBlock, self).__init__()

        self.conv1 = ReflectiveConvLayer(in_dim, in_dim, 3, 1)
        self.first_norm = BatchNormLayer(in_dim)
        self.conv2 = ReflectiveConvLayer(in_dim, in_dim, 3, 1)
        self.second_norm = BatchNormLayer(in_dim)

        self.up_conv = ReflectiveConvLayer(in_dim, res_dim, kernel_size=k1_size, stride=1, dilation=dilation)
        self.upscale_norm = BatchNormLayer(res_dim)

        self.down_conv = ReflectiveConvLayer(res_dim, out_dim, kernel_size=k2_size, stride=1)
        self.downscale_norm = BatchNormLayer(out_dim)

        self.with_relu = with_relu
        self.relu = nn.ReLU()

    def forward(self, x, res):
        x_r = x

        x = self.relu(self.first_norm(self.conv1(x)))
        x = self.conv2(x)
        x += x_r
        x = self.relu(self.second_norm(x))

        x = self.upscale_norm(self.up_conv(x))
        x += res
        x = self.relu(x)
        res = x

        x = self.downscale_norm(self.down_conv(x))
        x += x_r

        if self.with_relu:
            x = self.relu(x)
        else:
            pass

        return x, res

class ReflectiveConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, dilation=1):
        super(ReflectiveConvLayer, self).__init__()
        reflect_padding = int(dilation * (kernel_size - 1) / 2)
        self.reflection_pad = nn.ReflectionPad2d(reflect_padding)
        self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size, stride, dilation=dilation)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class BatchNormLayer(nn.Module):
    def __init__(self, dim):
        super(BatchNormLayer, self).__init__()
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        out = self.norm(x)
        return out

class SpikingConvLayer(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.lif_neuron = LIFNeuron()
        self.bn = nn.BatchNorm2d(c2)
        self.s = s

    def forward(self, x):
        T, B, C, H, W = x.shape
        H_new = int(H / self.s)
        W_new = int(W / self.s)
        x = self.lif_neuron(x)
        x = self.bn(self.conv(x.flatten(0, 1))).reshape(T, B, -1, H_new, W_new)
        return x

class LIFNeuron(nn.Module):
    def __init__(self):
        super(LIFNeuron, self).__init__()
        self.spike_quant = SpikeQuantizer()

    def forward(self, x):
        decay = 0.25
        spike = torch.zeros_like(x[0]).to(x.device)
        output = torch.zeros_like(x)
        mem_old = 0
        time_window = x.shape[0]
        for i in range(time_window):
            if i >= 1:
                mem = (mem_old - spike.detach()) * decay + x[i]
            else:
                mem = x[i]
            spike = self.spike_quant(mem)

            mem_old = mem.clone()
            output[i] = spike
        return output

class SpikeQuantizer(nn.Module):
    class quant4(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.round(torch.clamp(input, min=0, max=4))

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            grad_input[input < 0] = 0
            grad_input[input > 4] = 0
            return grad_input

    def forward(self, x):
        return self.quant4.apply(x)

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p