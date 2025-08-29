
import torch
import torch.nn as nn

# ---------------------------- CLASS ----------------------------
class VGG16_1D(nn.Module):
	def __init__(self, in_channels=12, num_classes=5, input_length=1000):
		super(VGG16_1D, self).__init__()

		self.features = nn.Sequential(
			# Block 1
			nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
			nn.BatchNorm1d(64),
			nn.ReLU(inplace=True),
			nn.Conv1d(64, 64, kernel_size=3, padding=1),
			nn.BatchNorm1d(64),
			nn.ReLU(inplace=True),
			nn.MaxPool1d(kernel_size=2, stride=2),

			# Block 2
			nn.Conv1d(64, 128, kernel_size=3, padding=1),
			nn.BatchNorm1d(128),
			nn.ReLU(inplace=True),
			nn.Conv1d(128, 128, kernel_size=3, padding=1),
			nn.BatchNorm1d(128),
			nn.ReLU(inplace=True),
			nn.MaxPool1d(kernel_size=2, stride=2),

			# Block 3
			nn.Conv1d(128, 256, kernel_size=3, padding=1),
			nn.BatchNorm1d(256),
			nn.ReLU(inplace=True),
			nn.Conv1d(256, 256, kernel_size=3, padding=1),
			nn.BatchNorm1d(256),
			nn.ReLU(inplace=True),
			nn.Conv1d(256, 256, kernel_size=3, padding=1),
			nn.BatchNorm1d(256),
			nn.ReLU(inplace=True),
			nn.MaxPool1d(kernel_size=2, stride=2),

			# Block 4
			nn.Conv1d(256, 512, kernel_size=3, padding=1),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 512, kernel_size=3, padding=1),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 512, kernel_size=3, padding=1),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.MaxPool1d(kernel_size=2, stride=2),

			# Block 5
			nn.Conv1d(512, 512, kernel_size=3, padding=1),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 512, kernel_size=3, padding=1),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 512, kernel_size=3, padding=1),
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.MaxPool1d(kernel_size=2, stride=2),
		)

		# Dynamisch berechnen, wie viele Features nach den Convs übrig sind
		with torch.no_grad():
			dummy = torch.zeros(1, in_channels, input_length)
			features_out = self.features(dummy)
			self.flattened_size = features_out.shape[1] * features_out.shape[2]

		self.classifier = nn.Sequential(
			nn.Linear(self.flattened_size, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, num_classes),
		)

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x

# Beispiel für die Initialisierung:
# model = VGG16_1D(in_channels=12, num_classes=5, input_length=1000)
