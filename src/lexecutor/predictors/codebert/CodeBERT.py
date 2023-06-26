from transformers import RobertaTokenizer, RobertaForMaskedLM
from ...Logging import logger
from ..DLUtil import device

def load_CodeBERT():
    logger.info("Loading pre-trained codebert-base-mlm")
    
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")

    value_tokens = [
        'None',
        'True',
        'False',
        'bool',
        'str_empty',
        'str_nonempty',
        'str',
        'int_neg',
        'int_zero',
        'int_pos',
        'int',
        'float_neg',
        'float_zero',
        'float_pos',
        'float',
        'list_empty',
        'list_nonempty',
        'list',
        'tuple_empty',
        'tuple_nonempty',
        'tuple',
        'set_empty',
        'set_nonempty',
        'set',
        'dict_empty',
        'dict_nonempty',
        'dict',
        'resource',
        'callable',
        'object'
    ]
    
    additional_tokens = ['<extra_id_2>', '<extra_id_3>', '<extra_id_4>', '<extra_id_5>']

    special_tokens_dict = {'additional_special_tokens': value_tokens + additional_tokens}
    tokenizer.add_special_tokens(special_tokens_dict)

    model = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    return tokenizer, model