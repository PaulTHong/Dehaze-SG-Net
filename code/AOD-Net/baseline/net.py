import torch
import torch.nn as nn
import math


class PALayer(nn.Module):
	def __init__(self, channel, mid_channel=None):
		super(PALayer, self).__init__()
		if mid_channel is None:
			mid_channel = channel // 8
		self.pa = nn.Sequential(
			nn.Conv2d(channel, mid_channel, 1, padding=0, bias=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(mid_channel, 1, 1, padding=0, bias=True),
			nn.Sigmoid()
		)

	def forward(self, x):
		y = self.pa(x)
		return x * y


class CALayer(nn.Module):
	def __init__(self, channel, mid_channel=None, return_attention=False):
		super(CALayer, self).__init__()
		if mid_channel is None:
			mid_channel = channel // 8
		self.return_attention = return_attention
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.ca = nn.Sequential(
			nn.Conv2d(channel, mid_channel, 1, padding=0, bias=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(mid_channel, channel, 1, padding=0, bias=True),
			nn.Sigmoid()
		)

	def forward(self, x):
		y = self.avg_pool(x)
		y = self.ca(y)
		return y if self.return_attention else x * y


# assert the in-channel and out-channel is same
class ResLayer(nn.Module):
	def __init__(self, channel):
		super(ResLayer, self).__init__()
		self.conv1 = nn.Conv2d(channel, channel, 1, 1, 0, bias=True)
		self.in1 = nn.InstanceNorm2d(channel, affine=True)
		# self.conv2 = nn.Conv2d(channel, channel, 3, 1, 1, bias=True)
		# self.in2 = nn.InstanceNorm2d(channel, affine=True)
		# self.conv3 = nn.Conv2d(channel, channel, 1, 1, 0, bias=True)
		# self.in3 = nn.InstanceNorm2d(channel, affine=True)
		self.relu = nn.ReLU(inplace=True)

		self.conv4 = nn.Conv2d(channel, channel, 3, 1, 1, bias=True)
		self.in4 = nn.InstanceNorm2d(channel, affine=True)


	def forward(self, x):
		y = self.relu(self.in1(self.conv1(x)))
		y = self.in4(self.conv4(y))
		return self.relu(x + y)


class dehaze_net(nn.Module):
	def __init__(self, gps=3, dim=64, kernel_size=3):
		super(dehaze_net, self).__init__()
		self.gps = gps
		self.dim = dim
		self.kernel_size = kernel_size

		self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
		self.in1 = nn.InstanceNorm2d(32, affine=True)
		self.conv2 = nn.Conv2d(32, self.dim, 3, 1, 1, bias=True)  # kernel_size=1
		self.in2 = nn.InstanceNorm2d(self.dim, affine=True)
		self.relu = nn.ReLU(inplace=True)

		self.res1 = ResLayer(self.dim)
		self.res2 = ResLayer(self.dim)
		self.res3 = ResLayer(self.dim)
		self.res4 = ResLayer(self.dim)
		self.res5 = ResLayer(self.dim)
		self.res6 = ResLayer(self.dim)
		self.res7 = ResLayer(self.dim)

		self.ca = CALayer(self.dim * self.gps, mid_channel=self.dim // 16, return_attention=True)  # 8
		self.pa = PALayer(self.dim)

		self.conv3 = nn.Conv2d(self.dim, 32, 3, 1, 1, bias=True)
		self.in3 = nn.InstanceNorm2d(32, affine=True)
		self.conv4 = nn.Conv2d(32, 3, 3, 1, 1, bias=True)
		self.in4 = nn.InstanceNorm2d(3, affine=True)

	def forward(self, x):
		y = self.relu(self.in1(self.conv1(x)))

		y1 = self.relu(self.in2(self.conv2(y)))
		y = self.res1(y1)
		y = self.res2(y)
		y = self.res3(y)
		y2 = self.res4(y)
		y = self.res5(y2)
		y = self.res6(y)
		y3 = self.res7(y)

		y = torch.cat([y1, y2, y3], dim=1)
		w = self.ca(y)
		w = w.view(-1, self.gps, self.dim)[:, :, :, None, None]
		y = w[:, 0, ::] * y1 + w[:, 1, ::] * y2 + w[:, 2, ::] * y3
		y = self.pa(y)

		y = self.relu(self.in3(self.conv3(y)))
		y = self.conv4(y)
		y = x + y
		out = self.relu(y)

		return out


if __name__ == '__main__':
	net = dehaze_net()
	print(net)
	num_para = torch.cat([p.view(-1) for p in net.parameters()]).size()
	print('number of network parameters: ', num_para)
	x = torch.randn(1, 3, 256, 256)
	y = net(x)
	print(y.size())
	from torchsummary import summary
	net = net.cuda()
	summary(net, (3, 256, 256))

			

			
			






