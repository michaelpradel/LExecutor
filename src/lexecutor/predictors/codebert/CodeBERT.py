from transformers import RobertaTokenizer, RobertaForMaskedLM
from ...Logging import logger
from ..DLUtil import device

def load_CodeBERT():
    logger.info("Loading pre-trained codebert-base-mlm")
    
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")
    special_tokens_dict = {'additional_special_tokens': ['<extra_id_2>', '<extra_id_3>', '<extra_id_4>', '<extra_id_5>']}
    tokenizer.add_special_tokens(special_tokens_dict)

    model = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    return tokenizer, model