
import torch
import torch.nn as nn

# ---------------------------- CLASS ----------------------------
class VGG16_1D(nn.Module):
	def __init__(self, in_channels=12, num_classes=2, input_length=70):
		"""
		1D VGG16 model adapted for time series data. Shape of input data should be (batch_size, in_channels, input_length). 
		Args:
			in_channels (int): Number of input channels (e.g., 12 for 12-lead ECG).
			num_classes (int): Number of output classes.
			input_length (int): Length of the input time series.
		"""
		super(VGG16_1D, self).__init__()

		self.features = nn.Sequential(
			# Block 1												# → Input: [Batch, 12, 70]
			nn.Conv1d(in_channels, 64, kernel_size=3, padding=1), 	# Length 70 -> 70
			nn.BatchNorm1d(64),
			nn.ReLU(inplace=True),
			nn.Conv1d(64, 64, kernel_size=3, padding=1),			# Length 70 -> 70
			nn.BatchNorm1d(64),
			nn.ReLU(inplace=True),
			nn.MaxPool1d(kernel_size=2, stride=2),					# Length 70 -> 35
																	# → Output: [Batch, 64, 35]

			# Block 2
			nn.Conv1d(64, 128, kernel_size=3, padding=1),			# Length 35 -> 35
			nn.BatchNorm1d(128),
			nn.ReLU(inplace=True),
			nn.Conv1d(128, 128, kernel_size=3, padding=1), 			# Length 35 -> 35
			nn.BatchNorm1d(128),
			nn.ReLU(inplace=True),
			nn.MaxPool1d(kernel_size=2, stride=2), 					# Length 35 -> 17
																	# → Output: [Batch, 128, 17]

			# Block 3
			nn.Conv1d(128, 256, kernel_size=3, padding=1),			# Length 17 -> 17
			nn.BatchNorm1d(256),
			nn.ReLU(inplace=True),
			nn.Conv1d(256, 256, kernel_size=3, padding=1), 			# Length 17 -> 17
			nn.BatchNorm1d(256),
			nn.ReLU(inplace=True),
			nn.Conv1d(256, 256, kernel_size=3, padding=1), 			# Length 17 -> 17
			nn.BatchNorm1d(256),
			nn.ReLU(inplace=True),
			nn.MaxPool1d(kernel_size=2, stride=2), 					# Length 17 -> 8
																	# → Output: [Batch, 256, 8]

			# Block 4
			nn.Conv1d(256, 512, kernel_size=3, padding=1),			# Length 8 -> 8
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 512, kernel_size=3, padding=1),			# Length 8 -> 8
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.Conv1d(512, 512, kernel_size=3, padding=1), 			# Length 8 -> 8
			nn.BatchNorm1d(512),
			nn.ReLU(inplace=True),
			nn.MaxPool1d(kernel_size=2, stride=2), 					# Length 8 -> 4
																	# → Output: [Batch, 512, 4]
																	# → Flatten Size: 512*4 = 2048

			# Block 5
			# nn.Conv1d(512, 512, kernel_size=3, padding=1), 		# Length 4 -> 4
			# nn.BatchNorm1d(512),
			# nn.ReLU(inplace=True),
			# nn.Conv1d(512, 512, kernel_size=3, padding=1), 		# Length 4 -> 4
			# nn.BatchNorm1d(512),
			# nn.ReLU(inplace=True),
			# nn.Conv1d(512, 512, kernel_size=3, padding=1), 		# Length 4 -> 4
			# nn.BatchNorm1d(512),
			# nn.ReLU(inplace=True),
			# nn.MaxPool1d(kernel_size=2, stride=2), 				# Length 4 -> 2
			 														# → Output: [Batch, 512, 2]
			 														# → Flatten Size: 512*2 = 1024
		)

		# Dynamisch berechnen, wie viele Features nach den Convs übrig sind
		with torch.no_grad():
			dummy = torch.zeros(1, in_channels, input_length)
			features_out = self.features(dummy)
			self.flattened_size = features_out.shape[1] * features_out.shape[2]

		self.classifier = nn.Sequential(
			nn.Linear(self.flattened_size, 256),
			nn.ReLU(True),
			nn.Dropout(p=0.5), 
			nn.Linear(256, 256),
			nn.ReLU(True),
			nn.Dropout(p=0.5),
			nn.Linear(256, num_classes)
			# nn.Softmax(dim=1) # Softmax wird in CrossEntropyLoss integriert, daher hier nicht notwendig
		)

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x

