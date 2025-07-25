# NIR: A Neural Instance Retriever for Description Logic Concepts
A neural instance retriever in Python which leverages the convenient transformers' API.

# Installation

- Run this command: 
```bash
git clone https://github.com/fosterreproducibleresearch/NIR && cd NIR && conda create -n nir python=3.12.9 --y && conda activate nir && pip install -e .
```

# Training NIR

- Download datasets and pretrained models from [datasets and models](https://figshare.com/s/04189ac2f5b8de24aeef) and decompress files in the main directory.

- Execute the following for training:

```bash
nir --dataset_dir {path_to_dataset} # Use `-h` for more options. An example path is --dataset_dir ./datasets/semantic_bible
```

- Pretrained model can be loaded before training:
```bash
nir --dataset_dir {path_to_dataset} --pretrained_model_path {path}
```

# Inference

```python
import torch
from transformers import AutoModel, AutoConfig, AutoTokenizer
from nir.models import NIRTransformer
from nir.config import NIRConfig
from nir.utils import read_embs
AutoConfig.register("nir", NIRConfig)
AutoModel.register(NIRConfig, NIRTransformer)
pretrained_model_path = "nir_pretrained_models/NIR_Transformer_animals"
model = AutoModel.from_pretrained(f"{pretrained_model_path}")
tokenizer = AutoTokenizer.from_pretrained(f"{pretrained_model_path}")

## First example
print("First example\n")
class_expression = "(¬HasGills) ⊓ (¬Penguin)"
individual = "animals#eagle01"
embeddings = read_embs("./datasets/animals/")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
inputs = tokenizer([class_expression], padding="max_length", truncation=True, max_length=model.max_length, return_tensors='pt')
input_ids = inputs['input_ids'].to(device)
attention_mask = inputs['attention_mask'].to(device)
ind_embs = torch.FloatTensor(embeddings.loc[individual].values).unsqueeze(0).to(device)

probability = model(input_ids, attention_mask, ind_embs)

print(f"Probability that `{individual}` is an instance of `{class_expression}` is {probability}")

# Probability that `animals#eagle01` is an instance of `(¬HasGills) ⊓ (¬Penguin)` is 0.9818605184555054


## Second example
print("\n\nSecond example\n")
class_expression = "(¬Eel) ⊓ (¬Bird)"
individual = "animals#eel01"
ind_embs = torch.FloatTensor(embeddings.loc[individual].values).unsqueeze(0).to(device)
inputs = tokenizer([class_expression], padding="max_length", truncation=True, max_length=model.max_length, return_tensors='pt')
input_ids = inputs['input_ids'].to(device)
attention_mask = inputs['attention_mask'].to(device)

probability = model(input_ids, attention_mask, ind_embs)

print(f"Probability that `{individual}` is an instance of `{class_expression}` is {probability}")

# Probability that `animals#eel01` is an instance of `(¬Eel) ⊓ (¬Bird)` is 0.06383775174617767
```
