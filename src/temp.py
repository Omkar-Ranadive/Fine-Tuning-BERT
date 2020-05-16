"""
Refer to https://github.com/BramVanroy/bert-for-inference/blob/master/introduction-to-bert.ipynb for a quick intro

Code credits to: Bram Vanroy

"""

from transformers import BertModel, BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Convert the string "granola bars" to tokenized vocabulary IDs
granola_ids = tokenizer.encode('granola bars', return_tensors='pt')
# Print the IDs
print('granola_ids', granola_ids)
print('type of granola_ids', type(granola_ids))
# Convert the IDs to the actual vocabulary item
# Notice how the subword unit (suffix) starts with "##" to indicate
# that it is part of the previous string
print('granola_tokens', tokenizer.convert_ids_to_tokens(granola_ids[0]))

model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
# Set the device to GPU (cuda) if available, otherwise stick with CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = model.to(device)
granola_ids = granola_ids.to(device)

model.eval()

print(granola_ids.size())
# unsqueeze IDs to get batch size of 1 as added dimension
# granola_ids = granola_ids.unsqueeze(0)
# print(granola_ids.size())

print(type(granola_ids))
with torch.no_grad():
    out = model(input_ids=granola_ids)

# the output is a tuple
print(type(out))
# the tuple contains three elements as explained above)
print(len(out))
# we only want the hidden_states
hidden_states = out[2]
print(len(hidden_states))
print(hidden_states[-1].shape)

sentence_embedding = torch.mean(hidden_states[-1], dim=1).squeeze()
print(sentence_embedding)
print(sentence_embedding.size())

# get last four layers
last_four_layers = [hidden_states[i] for i in (-1, -2, -3, -4)]
# cast layers to a tuple and concatenate over the last dimension
cat_hidden_states = torch.cat(tuple(last_four_layers), dim=-1)
print(cat_hidden_states.size())

# take the mean of the concatenated vector over the token dimension
cat_sentence_embedding = torch.mean(cat_hidden_states, dim=1).squeeze()
# print(cat_sentence_embedding)
print(cat_sentence_embedding.size())