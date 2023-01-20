import torch as t
from ..Util import device
from ..predictors.codet5.FineTune import load_CodeT5, evaluate


def evaluate_model(name, state_file, validation_data_file):
    tokenizer, model = load_CodeT5()
    model.load_state_dict(t.load(state_file, map_location=device))
    topk_accuracies = evaluate(validation_data_file, model, tokenizer)
    print("="*30)
    print(f"{name}\n{topk_accuracies}")
    print("="*30)


if __name__ == "__main__":

    # evaluate_model("fine-grained",
    #                "data/codeT5_models/jan5_5_projects/pytorch_model_epoch9.bin",
    #                "data/codeT5_models/jan5_5_projects/validatejan5_5_projects.pt")
    
    evaluate_model("coarse-grained",
                   "data/codeT5_models/jan5_5_projects_coarse-grained/pytorch_model_epoch9.bin",
                   "data/codeT5_models/jan5_5_projects_coarse-grained/validatejan5_5_projects_coarse-grained.pt")
