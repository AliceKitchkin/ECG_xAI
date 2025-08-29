
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


class LRP_1D:
	"""
	Layer-wise Relevance Propagation (LRP) für 1D-EKG-Daten und VGG16_1D-Modelle.
	Diese Klasse ist unabhängig und kann später mit anderen Methoden kombiniert werden.
	"""
	def __init__(self, model, device='cpu'):
		self.model = model.eval()
		self.device = device
		self.model.to(self.device)


	def explain(self, x, target_class=None, rule="epsilon", eps=1e-6):
		"""
		Führt echtes LRP (z-Rule/epsilon-Rule) für 1D-VGG durch.
		Args:
			x: torch.Tensor, shape (12, T) oder (B, 12, T)
			target_class: int oder None. Wenn None, wird die vorhergesagte Klasse verwendet.
			rule: "z" oder "epsilon" (nur epsilon-Rule implementiert)
			eps: Stabilisierungskonstante für epsilon-Rule
		Returns:
			relevance: np.ndarray, shape wie x
			prediction: int (vorhergesagte Klasse)
		"""
		self.model.zero_grad()
		if x.ndim == 2:
			x = x.unsqueeze(0)  # (1, 12, T)
		x = x.to(self.device)

		# Forward-Pass: speichere alle Zwischenoutputs
		activations = [x]
		input_ = x
		for layer in self.model.features:
			input_ = layer(input_)
			activations.append(input_)
		# Flatten
		input_ = input_.view(input_.size(0), -1)
		activations.append(input_)
		for layer in self.model.classifier:
			input_ = layer(input_)
			activations.append(input_)
		output = input_

		pred = output.argmax(dim=1).item()
		class_idx = target_class if target_class is not None else pred

		# Initialrelevanz: nur Zielklasse
		R = torch.zeros_like(output)
		R[0, class_idx] = output[0, class_idx]

		# Backward durch classifier
		idx = len(self.model.classifier)
		for layer in reversed(self.model.classifier):
			idx -= 1
			if isinstance(layer, torch.nn.Linear):
				R = self._lrp_linear(activations[-2-idx], layer, R, rule=rule, eps=eps)
			elif isinstance(layer, torch.nn.ReLU) or isinstance(layer, torch.nn.Dropout):
				pass  # Relevanz einfach durchreichen
			else:
				raise NotImplementedError(f"LRP für Layer {type(layer)} nicht implementiert")

		# Unflatten
		R = R.view_as(activations[len(self.model.features)])

		# Backward durch features
		idx = len(self.model.features)
		for i, layer in enumerate(reversed(self.model.features)):
			idx -= 1
			if isinstance(layer, torch.nn.Conv1d):
				R = self._lrp_conv1d(activations[idx], layer, R, rule=rule, eps=eps)
			elif isinstance(layer, torch.nn.BatchNorm1d) or isinstance(layer, torch.nn.ReLU) or isinstance(layer, torch.nn.MaxPool1d):
				pass  # Relevanz einfach durchreichen
			else:
				raise NotImplementedError(f"LRP für Layer {type(layer)} nicht implementiert")

		relevance = R.detach().cpu().numpy().squeeze()
		return relevance, pred


	def _lrp_linear(self, input, layer, relevance, rule="epsilon", eps=1e-6):
		# input: [B, N], relevance: [B, M], layer: nn.Linear(N, M)
		weight = layer.weight
		bias = layer.bias if layer.bias is not None else 0
		with torch.no_grad():
			z = torch.matmul(input, weight.t()) + bias
			z += eps * torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))
			s = relevance / z
			c = torch.matmul(s, weight)
			return input * c


	def _lrp_conv1d(self, input, layer, relevance, rule="epsilon", eps=1e-6):
		# input: [B, C_in, T], relevance: [B, C_out, T_out], layer: nn.Conv1d
		weight = layer.weight
		bias = layer.bias if layer.bias is not None else 0
		stride = layer.stride
		padding = layer.padding
		dilation = layer.dilation
		groups = layer.groups
		with torch.no_grad():
			z = torch.nn.functional.conv1d(input, weight, bias, stride, padding, dilation, groups)
			z += eps * torch.where(z >= 0, torch.ones_like(z), -torch.ones_like(z))
			s = relevance / z
			c = torch.nn.functional.conv_transpose1d(s, weight, None, stride, padding, dilation, groups)
			return input * c


	@staticmethod
	def plot_relevance(signal, relevance, lead_names=None, figsize=(16, 10), cmap="coolwarm"):
		"""
		Plotte die 12 Ableitungen und deren Relevanz als farbige Punkte auf den Signalen.
		Args:
			signal: np.ndarray, shape (12, T)
			relevance: np.ndarray, shape (12, T)
			lead_names: list of str, optional
			cmap: str, Matplotlib Colormap
		"""
		n_leads = signal.shape[0]
		if lead_names is None:
			lead_names = [f"Lead {i+1}" for i in range(n_leads)]
		fig, axes = plt.subplots(n_leads, 1, figsize=figsize, sharex=True)
		if n_leads == 1:
			axes = [axes]
		for i, ax in enumerate(axes):
			ax.plot(signal[i], color='black', lw=1, zorder=1)
			ax.set_ylabel(lead_names[i])
			rel = relevance[i]
			rel_norm = (rel - rel.min()) / (rel.ptp() + 1e-8)
			# Farbverlauf für Punkte
			points = ax.scatter(np.arange(len(rel)), signal[i], c=rel_norm, cmap=cmap, s=10, zorder=2)
		axes[-1].set_xlabel('Zeit')
		fig.colorbar(points, ax=axes, orientation='vertical', label='LRP-Relevanz', pad=0.01, aspect=30)
		plt.tight_layout()
		plt.show()
