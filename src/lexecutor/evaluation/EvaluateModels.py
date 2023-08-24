import argparse
import torch as t
from ..predictors.DLUtil import device
from ..predictors.codet5.FineTune import evaluate as evaluate_CodeT5, load_CodeT5
from ..predictors.codebert.FineTune import evaluate as evaluate_CodeBERT, load_CodeBERT

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_type", help="CodeT5 or CodeBERT", required=True)
parser.add_argument(
    "--value_abstraction", help="fine-grained or coarse-grained", required=True)
parser.add_argument(
    "--state_file", help=".bin file for validation", default="model.bin")
parser.add_argument(
    "--validate_tensors", help=".pt file for validation", default="validate.pt")

def evaluate_model(model_type, value_abstraction, state_file, validation_data_file):
    if model_type == "CodeT5":
        tokenizer, model = load_CodeT5()
        model.load_state_dict(t.load(state_file, map_location=device))
        topk_accuracies = evaluate_CodeT5(validation_data_file, model, tokenizer)
    else:
        tokenizer, model = load_CodeBERT()
        model.load_state_dict(t.load(state_file, map_location=device))
        topk_accuracies = evaluate_CodeBERT(validation_data_file, model, tokenizer)

    print("="*30)
    print(f"{model_type}-{value_abstraction}\n{topk_accuracies}")
    print("="*30)


if __name__ == "__main__":
    args = parser.parse_args()

    evaluate_model(args.model_type,
                   args.value_abstraction,
                   args.state_file,
                   args.validate_tensors)
