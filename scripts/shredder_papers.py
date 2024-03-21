import os

import papermill as pm


def shredder_papers(models, base_path="notebooks/base", out_path="notebooks"):
    for model_path in models:
        os.makedirs(os.path.join(out_path, model_path), exist_ok=True)
        for fname in base_path:
            notebook_path = os.path.join(base_path, fname)
            if fname == "text_classification.ipynb":
                params = {
                    "text_classification_linear_probes.ipynb": {
                        "full_finetune": False,
                        "num_heads": 6,
                        "model_path": model_path,
                    },
                    "text_classification_full_finetune.ipynb": {
                        "full_finetune": True,
                        "num_heads": 2,
                        "model_path": model_path,
                    },
                }
            else:
                params = {fname: {"model_path": model_path}}
            for out_fname, param in params.items():
                pm.execute_notebook(
                    notebook_path, os.path.join(out_path, out_fname), parameters=param
                )
