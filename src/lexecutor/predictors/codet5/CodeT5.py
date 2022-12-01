from transformers import AutoTokenizer, T5ForConditionalGeneration, AdamW
from ...Logging import logger
from ...Util import device


def load_CodeT5():
    logger.info("Loading pre-trained codet5-small")
    tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5-small')
    model = T5ForConditionalGeneration.from_pretrained(
        'Salesforce/codet5-small')
    model.to(device)
    return tokenizer, model
