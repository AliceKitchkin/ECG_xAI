import torch
from torch.utils.data import TensorDataset

# ------------------------------ CLASS ------------------------------
class TensorPreprocessor:
	"""
	Hilfsklasse zur Umwandlung von NumPy-Arrays in PyTorch-Tensoren
	und zur Anpassung der Achsen für 1D-Modelle (z.B. VGG16_1D).
	"""
	def __init__(self):
		pass


	def to_tensor(self, X, y=None, channels_first=True, dtype_x=torch.float32, dtype_y=torch.long, expected_channels=None, expected_length=None):
		"""
		Wandelt X (und optional y) in PyTorch-Tensoren um und permutiert die Achsen bei Bedarf.
		Args:
			X: np.ndarray, shape [N, L, C] oder [N, C, L]
			y: np.ndarray oder None
			channels_first: Wenn True, permutiere zu [N, C, L] falls nötig
			dtype_x: Datentyp für X
			dtype_y: Datentyp für y
		Returns:
			X_tensor, y_tensor (oder nur X_tensor)
		"""
		X_tensor = torch.tensor(X, dtype=dtype_x)
		if channels_first and X_tensor.ndim == 3 and X_tensor.shape[1] < X_tensor.shape[2]:
			# Vermute, dass [N, L, C] vorliegt, permutiere zu [N, C, L]
			X_tensor = X_tensor.permute(0, 2, 1)
		if y is not None:
			y_tensor = torch.tensor(y, dtype=dtype_y)
			X_tensor = self.auto_fix_and_check(X_tensor, y_tensor, expected_channels, expected_length)
			return X_tensor, y_tensor
		X_tensor = self.auto_fix_and_check(X_tensor, None, expected_channels, expected_length)
		return X_tensor


	def to_dataset(self, X, y=None, channels_first=True, dtype_x=torch.float32, dtype_y=torch.long, expected_channels=None, expected_length=None):
		"""
		Erstellt einen TensorDataset (mit oder ohne Labels) aus den Eingabedaten.
		"""
		if y is not None:
			X_tensor, y_tensor = self.to_tensor(X, y, channels_first, dtype_x, dtype_y, expected_channels, expected_length)
			return TensorDataset(X_tensor, y_tensor)
		else:
			X_tensor = self.to_tensor(X, None, channels_first, dtype_x, dtype_y, expected_channels, expected_length)
			return TensorDataset(X_tensor)


	def check_shapes(self, X_tensor, y_tensor=None, expected_channels=None, expected_length=None):
		"""
		Prüft und gibt die Shapes und Typen der Tensoren aus. Optional: Erwartete Channels/Length prüfen.
		"""
		print(f"Final shapes - X: {tuple(X_tensor.shape)}")
		print(f"X_tensor: shape={tuple(X_tensor.shape)}, dtype={X_tensor.dtype}")
		if y_tensor is not None:
			print(f"Final shapes - y: {tuple(y_tensor.shape)}")
		print("--------------------------")


	def auto_fix_and_check(self, X_tensor, y_tensor=None, expected_channels=None, expected_length=None):
		"""
		Prüft die Shapes und permutiert X_tensor automatisch, falls Channels/Length vertauscht sind.
		Gibt die finale Shape-Info aus und gibt den ggf. korrigierten Tensor zurück.
		"""
		if expected_channels is not None and expected_length is not None and X_tensor.ndim == 3:
			if X_tensor.shape[1] == expected_length and X_tensor.shape[2] == expected_channels:
				X_tensor = X_tensor.permute(0, 2, 1)
			self.check_shapes(X_tensor, y_tensor, expected_channels, expected_length)
		return X_tensor