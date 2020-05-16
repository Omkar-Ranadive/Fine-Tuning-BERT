from transformers import BertModel, BertTokenizer
import torch
import os


class BertEmbeddings:
	def __init__(self):
		self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.model = self.model.to(self.device)
		self.model.eval()

	def generate_embeddings(self, src, segs):
		"""

		Args:
			src (List of integers): These integers represent word identifiers
			segs (List of token type ids): These are integers 0 or 1

		Returns:

		"""
		# Convert the list to Pytorch tensor
		src = torch.LongTensor(src).unsqueeze(0)
		segs = torch.LongTensor(segs).unsqueeze(0)
		src = src.to(self.device)
		segs = segs.to(self.device)

		with torch.no_grad():
			out = self.model(input_ids=src, token_type_ids=segs)
			hidden_states = out[2]
			print(hidden_states[-1].shape)

			# get last four layers
			last_four_layers = [hidden_states[i] for i in (-1, -2, -3, -4)]
			# cast layers to a tuple and concatenate over the last dimension
			cat_hidden_states = torch.cat(tuple(last_four_layers), dim=-1)
			print(cat_hidden_states.size())

			# take the mean of the concatenated vector over the token dimension
			cat_sentence_embedding = torch.mean(cat_hidden_states, dim=1).squeeze()
			# print(cat_sentence_embedding)
			print(cat_sentence_embedding.size())

			return cat_sentence_embedding


def load_data(dir_path, file_type):
	"""

	Args:
		dir_path (str): A string which points to the directory with training files
		file_type (str): Can be either 'train', 'test', 'val'.

	Returns: Loads the PyTorch files
	"""

	for file in os.listdir(dir_path):
		if file_type in str(file):
			yield torch.load(dir_path + file)


if __name__ == "__main__":

	# Simple use case of this file
	path = '../bert_data/'

	bertObj = BertEmbeddings()

	loaded_files = load_data(path, 'train')

	for files in loaded_files:
		for f in files:
			# print(f.keys())
			bertObj.generate_embeddings(src=f['src'], segs=f['segs'])
			break
		break

