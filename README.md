<h4 align="center">
    <p>
        <a href="https://transformer-heads.readthedocs.io/en/latest/">Documentation</a> |
        <a href="docs/source/getting_started.md">Getting Started</a> |
        <a href="https://www.reddit.com/r/LocalLLaMA/comments/1bnd621/new_library_transformerheads_for_attaching_heads/">Reddit Post with more info</a>
    </p>
</h4>

# Transformer Heads
This library aims to be an allround toolkit for attaching, training, saving and loading of new heads for transformer models.  
A new head could be: 
* A [linear probe](https://arxiv.org/pdf/1610.01644.pdf) used to get an understanding of the information processing in a transformer architecture
* A head to be finetuned jointly with the weights of a pretrained transformer model to perform a completely different kind of task.
    - E.g. a transformer pretrained to do causal language modelling could get a sequence classification head attached and be finetuned to do sentiment classification.
    - Or one could attach a regression head to turn a large language model into a value function for a reinforcement learning problem.

On top of that, attaching multiple heads at once can make multi-task learning easy, making it possible to train very general models.

## Installation
Install from pypi: `pip install transformer-heads`.

Or, clone this repo and from the root of this repository:
`pip install -e .`

## Usage
Create head configurations
```python
head_config = HeadConfig(
    name=f"imdb_head_3",
    layer_hook=-3,  # Attach at the output of the third-to-last transformer-block
    in_size=hidden_size,
    output_activation="linear",
    pred_for_sequence=True,
    loss_fct="cross_entropy",
    num_outputs=2,
    target="label" # The name of the ground-truth column in the dataset
)
```
Create a model with your head from a pretrained transformer model
```python
model = load_headed(
    LlamaForCausalLM,
    "meta-llama/Llama-2-7b-hf",
    head_configs=[heads_config],
)
```
Train you model using (for example) the simple to use huggingface *Trainer* interface:
```python
trainer = Trainer(
    model,
    args=args,
    train_dataset=imdb_dataset["train"],
    data_collator=collator,
)
```

For a more in-depth introduction and a fully working example, check the [linear probe notebook](notebooks/gpt2/linear_probe.ipynb).

## Explanation of approach for training a transformer value function with QLoRA
* **The Base Model**
    * The value model builds on a pre-trained base large language model. 
    * That is, a transformer model trained on the causal language modelling objective on a large corpus of free flowing text
    * To solve the task, LLMs have a linear causal language modelling head that projects from the hidden dimension for each token to the number of tokens in the vocabulary.
    * The base model is not instruct tuned or trained by RLHF
* **Adding a value head**
    * The causal language modelling head is removed.
    * It is replaced by a value head that projects from the hidden dimension for each token to a one-dimensional value prediction.
    * The value head may be linear or a small multilayer perceptron.
    * The value head is solving a regression task and is trained via the mean-squared-error loss.
* **Preparing for QLoRA training**
    * QLoRA is desired to reduce memory-overhead and enable DDP training.
    * All weights from the model except the value-head are quantized and frozen.
    * LoRA weights are trained for all these frozen weights.
    * The value-head is still fully trained.

## Joint training of multiple linear probes
![_images/multi_linear_probe.svg](_images/multi_linear_probe.svg)

## Notebooks
This repository contains multiple jupyter notebooks for a tutorial/illustration of how do do certain things with this library. Here is an overview of which notebook you should check out depending on the use you are interested in.
* Linear Probes (understanding the inner workings of transformers)
    - Basic example with one probe for causal LM: [notebooks/gpt2/linear_probe.ipynb](notebooks/gpt2/linear_probe.ipynb)
    - Train many probes for causal LM at once: [notebooks/gpt2/multi_linear_probe.ipynb](notebooks/gpt2/multi_linear_probe.ipynb)
    - Train many probes for text classification at once: [notebooks/gpt2/text_classification_linear_probe.ipynb](notebooks/gpt2/text_classification_linear_probe.ipynb)
* Finetuning on a new type of task (with a new head)
    - QLoRA: [notebooks/gpt2/text_classification_qlora.ipynb](notebooks/gpt2/text_classification_qlora.ipynb)
    - Full finetuning: [notebooks/gpt2/text_classification_full_finetune.ipynb](notebooks/gpt2/text_classification_full_finetune.ipynb)
* Joint multi-task learning
    - Many heads doing completely different tasks + QLoRA, all trained at the same time: [notebooks/gpt2/joint_multitask_learning.ipynb](notebooks/gpt2/joint_multitask_learning.ipynb)
* Regression with pretrained transformers
    - Check the regression heads of this notebook: [notebooks/gpt2/joint_multitask_learning.ipynb](notebooks/gpt2/joint_multitask_learning.ipynb)
* Saving and loading
    - Notebook: [notebooks/gpt2/saving_and_loading.ipynb](notebooks/gpt2/saving_and_loading.ipynb)
    - Tests: [transformer_heads/tests/test_load_model.py](transformer_heads/tests/test_load_model.py)

## Joint multi-task training with different types of heads and QLoRA.
![_images/example_architecture.svg](_images/example_architecture.svg)

## More custom loss functions and models
At the state of writing, only a subset of loss functions are supported out of the box. Check [transformer_heads/constants.py](transformer_heads/constants.py) for more up to date info.

However, it is not so hard to add/use different loss functions/models. You'll just need to add their respective information to `loss_fct_map` and `model_type_map`. Just import from `transformer_heads.constants`. To add a loss function, add a mapping from string to torch class. To add a model add a mapping from model type to a 2 tuple out of attribute name of the base model in the Model Class and Base model class. That may sound confusing, but what that means is just the following:

```python
from transformer_heads.constants import model_type_map, loss_fct_map
import torch.nn as nn
from transformers import MistralModel

loss_fct_map["bce"] = nn.BCELoss()
model_type_map["mistral"] = ("model",MistralModel)
```
## Can my transformer architecture be supported?
One of the basic assumtions of my library is that there is a transformer class such as the LlamaForCausalLM class of huggingface that has an [attribute pointing to a base model that outputs raw hidden state](https://github.com/huggingface/transformers/blob/7eb3ba82241c927053689270a0751f4ff5d33c54/src/transformers/models/llama/modeling_llama.py#L1116). If your transformers model is built up in a similar way, adding support may be as easy as adding an entry to the [model_type_map](https://github.com/center-for-humans-and-machines/transformer-heads/blob/8ea0805ab95ca01dff7ea73ed9c844df946c17cb/transformer_heads/constants.py#L20) with the name of the attribute and the class of the base model. You can either do that by importing from [constants.py](transformer_heads/constants.py) or by adding it directly and creating a pull request.

## Q&A
* Is Llama-3 supported? YES! Check [here](https://github.com/center-for-humans-and-machines/transformer-heads/issues/3)
* How do I use my model for inference? Check the notebooks or [this](https://github.com/center-for-humans-and-machines/transformer-heads/issues/2) issue to get started.