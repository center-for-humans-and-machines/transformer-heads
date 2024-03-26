import json
from tqdm import tqdm
import sys


def remove_eval_spam(fname):
    with open(fname, "r") as f:
        data = json.load(f)

    for cell in tqdm(data["cells"]):
        if "outputs" in cell:
            last_eval = None
            outputs = []
            for output in cell["outputs"]:
                if (
                    "text" in output
                    and type(output["text"]) == list
                    and len(output["text"]) == 2
                    and output["text"][1].startswith("Evaluating: ")
                ):
                    last_eval = output
                else:
                    outputs.append(output)
            if last_eval is not None:
                outputs.insert(0, last_eval)
                cell["outputs"] = outputs

    with open(fname, "w") as f:
        json.dump(data, f, indent=1)


if __name__ == "__main__":
    remove_eval_spam(sys.argv[1])
