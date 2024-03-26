#!/bin/bash
source $HOME/private/setenv.sh
train_epochs=1
eval_epochs=1
logging_steps=100

mkdir notebooks/llama
papermill notebooks/base/saving_and_loading.ipynb notebooks/llama/saving_and_loading.ipynb -p model_path meta-llama/Llama-2-7b-hf -k sh_finetune
papermill notebooks/base/text_classification.ipynb notebooks/llama/text_classification_linear_probe.ipynb -p model_path meta-llama/Llama-2-7b-hf -p train_batch_size 2 -p eval_batch_size 2 -p train_epochs $train_epochs -p eval_epochs $eval_epochs -p logging_steps $logging_steps -p full_finetune False -p num_heads 6 -k sh_finetune
papermill notebooks/base/joint_multitask_learning.ipynb notebooks/llama/joint_multitask_learning.ipynb -p model_path meta-llama/Llama-2-7b-hf -p train_batch_size 2 -p eval_batch_size 2 -p train_epochs $train_epochs -p eval_epochs $eval_epochs -p logging_steps $logging_steps -k sh_finetune
# papermill notebooks/base/linear_probe.ipynb notebooks/llama/linear_probe.ipynb -p model_path meta-llama/Llama-2-7b-hf -p train_epochs $train_epochs -p eval_epochs $eval_epochs -p logging_steps $logging_steps -k sh_finetune
# papermill notebooks/base/multi_linear_probe.ipynb notebooks/llama/multi_linear_probe.ipynb -p model_path meta-llama/Llama-2-7b-hf -p train_epochs $train_epochs -p eval_epochs $eval_epochs -p logging_steps $logging_steps -k sh_finetune
papermill notebooks/base/text_classification_qlora.ipynb notebooks/llama/text_classification_qlora.ipynb -p model_path meta-llama/Llama-2-7b-hf -p train_batch_size 2 -p eval_batch_size 2 -p train_epochs $train_epochs -p eval_epochs $eval_epochs -p logging_steps $logging_steps -k sh_finetune

mkdir notebooks/gpt2
# papermill notebooks/base/linear_probe.ipynb notebooks/gpt2/linear_probe.ipynb -p model_path gpt2 -p train_epochs $train_epochs -p eval_epochs $eval_epochs -p logging_steps $logging_steps -k sh_finetune 
# papermill notebooks/base/joint_multitask_learning.ipynb notebooks/gpt2/joint_multitask_learning.ipynb -p model_path gpt2 -p train_epochs $train_epochs -p eval_epochs $eval_epochs -p logging_steps $logging_steps -k sh_finetune
# papermill notebooks/base/multi_linear_probe.ipynb notebooks/gpt2/multi_linear_probe.ipynb -p model_path gpt2 -p train_epochs $train_epochs -p eval_epochs $eval_epochs -p logging_steps $logging_steps -k sh_finetune
# papermill notebooks/base/text_classification.ipynb notebooks/gpt2/text_classification_linear_probe.ipynb -p model_path gpt2 -p train_epochs $train_epochs -p eval_epochs $eval_epochs -p logging_steps $logging_steps -p full_finetune False -p num_heads 6 -k sh_finetune
papermill notebooks/base/text_classification.ipynb notebooks/gpt2/text_classification_full_finetune.ipynb -p model_path gpt2 -p train_epochs $train_epochs -p eval_epochs $eval_epochs -p logging_steps $logging_steps -p full_finetune True -p num_heads 2 -k sh_finetune
# papermill notebooks/base/text_classification_qlora.ipynb notebooks/gpt2/text_classification_qlora.ipynb -p model_path gpt2 -p train_epochs $train_epochs -p eval_epochs $eval_epochs -p logging_steps $logging_steps -k sh_finetune
# papermill notebooks/base/saving_and_loading.ipynb notebooks/gpt2/saving_and_loading.ipynb -p model_path gpt2 -k sh_finetune

mkdir notebooks/mistral
papermill notebooks/base/joint_multitask_learning.ipynb notebooks/mistral/joint_multitask_learning.ipynb -p model_path mistralai/Mistral-7B-v0.1 -p train_batch_size 2 -p eval_batch_size 2 -p train_epochs $train_epochs -p eval_epochs $eval_epochs -p logging_steps $logging_steps -k sh_finetune
papermill notebooks/base/linear_probe.ipynb notebooks/mistral/linear_probe.ipynb -p model_path mistralai/Mistral-7B-v0.1 -p train_epochs $train_epochs -p eval_epochs $eval_epochs -p logging_steps $logging_steps -k sh_finetune
papermill notebooks/base/multi_linear_probe.ipynb notebooks/mistral/multi_linear_probe.ipynb -p model_path mistralai/Mistral-7B-v0.1 -p train_epochs $train_epochs -p eval_epochs $eval_epochs -p logging_steps $logging_steps -k sh_finetune
papermill notebooks/base/text_classification.ipynb notebooks/mistral/text_classification_linear_probe.ipynb -p model_path mistralai/Mistral-7B-v0.1 -p train_batch_size 2 -p eval_batch_size 2 -p train_epochs $train_epochs -p eval_epochs $eval_epochs -p logging_steps $logging_steps -p full_finetune False -p num_heads 6 -k sh_finetune
papermill notebooks/base/text_classification_qlora.ipynb notebooks/mistral/text_classification_qlora.ipynb -p model_path mistralai/Mistral-7B-v0.1 -p train_batch_size 2 -p eval_batch_size 2 -p train_epochs $train_epochs -p eval_epochs $eval_epochs -p logging_steps $logging_steps -k sh_finetune
papermill notebooks/base/saving_and_loading.ipynb notebooks/mistral/saving_and_loading.ipynb -p model_path mistralai/Mistral-7B-v0.1 -k sh_finetune
