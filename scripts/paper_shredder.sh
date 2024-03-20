mkdir notebooks/gpt2
papermill notebooks/base/linear_probe.ipynb notebooks/gpt2/linear_probe.ipynb -p model_path gpt2
papermill notebooks/base/joint_multitask_learning.ipynb notebooks/gpt2/joint_multitask_learning.ipynb -p model_path gpt2
papermill notebooks/base/multi_linear_probe.ipynb notebooks/gpt2/multi_linear_probe.ipynb -p model_path gpt2
papermill notebooks/base/text_classification.ipynb notebooks/gpt2/text_classification_linear_probe.ipynb -p model_path gpt2 -p full_finetune False -p num_heads 6
papermill notebooks/base/text_classification.ipynb notebooks/gpt2/text_classification_full_finetune.ipynb -p model_path gpt2 -p full_finetune True -p num_heads 2
papermill notebooks/base/text_classification_qlora.ipynb notebooks/gpt2/text_classification_qlora.ipynb -p model_path gpt2

mkdir notebooks/mistral
papermill notebooks/base/linear_probe.ipynb notebooks/mistral/linear_probe.ipynb -p model_path mistralai/Mistral-7B-v0.1
papermill notebooks/base/joint_multitask_learning.ipynb notebooks/mistral/joint_multitask_learning.ipynb -p model_path mistralai/Mistral-7B-v0.1
papermill notebooks/base/multi_linear_probe.ipynb notebooks/mistral/multi_linear_probe.ipynb -p model_path mistralai/Mistral-7B-v0.1
papermill notebooks/base/text_classification.ipynb notebooks/mistral/text_classification_linear_probe.ipynb -p model_path mistralai/Mistral-7B-v0.1 -p full_finetune False -p num_heads 6
papermill notebooks/base/text_classification_qlora.ipynb notebooks/mistral/text_classification_qlora.ipynb -p model_path mistralai/Mistral-7B-v0.1

mkdir notebooks/llama
papermill notebooks/base/linear_probe.ipynb notebooks/llama/linear_probe.ipynb -p model_path meta-llama/Llama-2-7b-hf
papermill notebooks/base/joint_multitask_learning.ipynb notebooks/llama/joint_multitask_learning.ipynb -p model_path meta-llama/Llama-2-7b-hf
papermill notebooks/base/multi_linear_probe.ipynb notebooks/llama/multi_linear_probe.ipynb -p model_path meta-llama/Llama-2-7b-hf
papermill notebooks/base/text_classification.ipynb notebooks/llama/text_classification_linear_probe.ipynb -p model_path meta-llama/Llama-2-7b-hf -p full_finetune False -p num_heads 6
papermill notebooks/base/text_classification_qlora.ipynb notebooks/llama/text_classification_qlora.ipynb -p model_path meta-llama/Llama-2-7b-hf