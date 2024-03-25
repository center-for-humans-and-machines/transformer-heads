## Getting Started
After you installed the library `pip install transformer-heads`, the jupyter notebooks in this repository will act as a tutorial to get you started with *transformer-heads*.

To start off, I recommend you to check the [linear probe notebook](../../notebooks/gpt2/linear_probe.ipynb). In this one, you will learn to train a single new linear head while keeping the base model frozen.

If you are interested in linear probes, you should now follow up with the [multi linear probe notebook](../../notebooks/gpt2/multi_linear_probe.ipynb) and with the [text classification linear probe notebook](../../notebooks/gpt2/text_classification_linear_probe.ipynb). They will teach you how to probe multiple layers at once to figure out what knowledge is encoded in the hidden representations of a transformer model.

If you are more interested in jointly finetuning model parameters and heads for a new task, you will want to check out the [text classification qlora notebook](../../notebooks/gpt2/text_classification_qlora.ipynb). In that one, you will finetune a model pretrained for causal language modelling towards doing text classification with QLoRA and a new two class classification head.

Now, finally, if you are interested in doing regression with LLMs or joint multitask learning, you may check out the [joint multitask learning notebook](../../notebooks/gpt2/joint_multitask_learning.ipynb) to learn about the most advanced capabilities of this library.