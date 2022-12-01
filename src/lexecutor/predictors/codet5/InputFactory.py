import json
import re
import torch as t
from ...Util import dtype, device
from ...Logging import logger


class InputFactory(object):
    def __init__(self, iids, tokenizer):
        self.iids = iids
        self.tokenizer = tokenizer

    def entry_to_inputs(self, entry):
        # Add <target> token before the identifier
        info = self.iids.iid_to_location[str(entry["iid"]-1)]  # TODO why -1 ?

        file = open(info[0]+'.orig')
        file_content = file.readlines()

        line = file_content[info[1]-1]
        name = entry["name"]

        match = (re.search(name, line[info[2]:]))

        if not match:
            start_index = 0
            end_index = 0
        else:
            start_index = match.span()[0] + info[2]
            end_index = start_index + len(name)

        # TODO shouldn't we add <target> and <target/> as custom separator tokens to the tokenizer?
        # (see "Custom special tokens" on https://github.com/huggingface/transformers/issues/7199)
        modified_line = line[:start_index] + '<target>' + \
            name + '</target>' + line[end_index:]
        file_content[info[1]-1] = modified_line

        # Get at most 512 tokens around the target token

        tokens = self.tokenizer.tokenize(''.join(file_content))
        # TODO why "target" and not "<target> or <target/>"?
        target_index = tokens.index('target')

        # fewer context before target
        if target_index < 255:
            previous_target_tokens = tokens[0:target_index]
            after_target_tokens = tokens[target_index:target_index +
                                         (512 - len(previous_target_tokens))]
        # fewer context after target
        elif target_index + 255 > len(tokens):
            after_target_tokens = tokens[target_index:]
            previous_target_tokens = tokens[target_index -
                                            (512 - len(after_target_tokens)):target_index]
        # equal context before and after target
        else:
            previous_target_tokens = tokens[target_index-255:target_index]
            after_target_tokens = tokens[target_index:target_index+255]

        considered_tokens = previous_target_tokens + after_target_tokens
        considered_tokens[0] = '<s>'
        considered_tokens[-1] = '</s>'

        # convert tokens to ids

        input_ids = self.tokenizer.convert_tokens_to_ids(considered_tokens)

        # Add padding
        if len(input_ids) < 512:
            input_ids = input_ids + (512 - len(input_ids)) * [0]

        # Create labels
        if not hasattr(entry, "value") or '@' not in entry["value"]:
            value = "unknown"
        else:
            value = entry["value"][1:]

        # labels: <s><target>value</s>
        label_ids = [1, 32, 3299, 34] + \
            self.tokenizer.convert_tokens_to_ids([value]) + [2]

        input_ids = t.tensor(input_ids, device='cpu')
        label_ids = t.tensor(label_ids, device='cpu')

        return input_ids, label_ids
