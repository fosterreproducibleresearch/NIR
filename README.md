# NIR: A Neural Instance Retriever for Description Logic Concepts
A neural instance retriever in Python which leverages the convenient transformers' API.

# Installation

- Run this command: 
```bash
git clone https://github.com/fosterreproducibleresearch/NIR && cd NIR && conda create -n nir python=3.12.9 --y && conda activate nir && pip install -e .
```

# Training NIR

- Download datasets and pretrained models from [datasets and models](https://figshare.com/s/0a144cb1ce88cfa046a3) and decompress files in the main directory.

- Execute the following for training:

```bash
nir --dataset_dir {path_to_dataset} # Use `-h` for more options. An example path is --dataset_dir ./datasets/animals
```

An example is

```bash
nir --dataset_dir datasets/lymphography --output_dir NIR_Composite_lymph --num_example 50 --epochs 400 --model composite --pma_model_path pma_pretrained/PMA_lymph/model.pt --use_pma True --batch_size 256 --num_workers 0
```

- Pretrained model can be loaded before training:
```bash
nir --dataset_dir {path_to_dataset} --pretrained_model_path {path}
```

# Reproducing Results

- To reproduce retrieval results on all datasets (Animals as an example), please first set the following files as executables:
```bash
1. chmod +x retrieval_eval_compositional
2. chmod +x retrieval_eval_encoders
```
then run:

```bash
./retrieval_eval_compositional
```
for the compositional architecture, or

```bash
./retrieval_eval_encoders
```
for encoding-based models.
One may also execute commands that are inside the above files, e.g.,
```bash
python nir/scripts/retrieval_eval.py --dataset_dir ./datasets/animals/ --model Transformer --output_dir Results/NIR_Transformer_Eval_animals --pretrained_model_path nir_pretrained_models/NIR_Transformer_animals/
```


# Inference (Hands-on Example)


```python
import torch
from transformers import AutoModel, AutoConfig, AutoTokenizer
from nir.models import NIRTransformer
from nir.config import NIRConfig
from nir.utils import read_embs
AutoConfig.register("nir", NIRConfig)
AutoModel.register(NIRConfig, NIRTransformer)
pretrained_model_path = "new_train_dir/NIR_Transformer_animals"
model = AutoModel.from_pretrained(f"{pretrained_model_path}")
tokenizer = AutoTokenizer.from_pretrained(f"{pretrained_model_path}")

print("First example\n")
class_expression = "¬Penguin ⊓ ∀ hasCovering.Feathers"
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

# Probability that `animals#eagle01` is an instance of `¬Penguin ⊓ ∀ hasCovering.Feathers` is 0.9999998807907104


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

# Probability that `animals#eel01` is an instance of `(¬Eel) ⊓ (¬Bird)` is 9.744724138727179e-07
```
