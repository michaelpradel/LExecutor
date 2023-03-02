from transformers import AutoTokenizer, T5ForConditionalGeneration, AdamW
from ...Logging import logger
from ..DLUtil import device


def load_CodeT5():
    logger.info("Loading pre-trained codet5-small")
    
    tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5-small')
    
    # logger.info(f"Special tokens: {tokenizer.all_special_tokens=}")
    # logger.info(f"Input ids of special tokens: {tokenizer.all_special_ids=}")
    
    model = T5ForConditionalGeneration.from_pretrained(
        'Salesforce/codet5-small')
    model.to(device)
    
    return tokenizer, model
