import torch
import torch.nn as nn
import torch.optim as optim
from glow.modules import InvertibleConv1x1
from glow.models import FlowNet


model = FlowNet(image_shape=(32,32,3), hidden_channels=32, K=2, L=2)


optimizer = optim.Adam(model.parameters(), lr=2e-4)


for idx in range(10000):
	A = torch.randn(2,3,32,32)
	B, B_logdet = model(A)
	import pdb; pdb.set_trace()
	A_t, A_logdet = model(B, B_logdet, reverse=True)

	#loss = torch.mean(A)
	#loss.backward()
	#optimizer.step()
	print("loss:",torch.mean(A-A_t).data)
	
import pdb; pdb.set_trace()
print()