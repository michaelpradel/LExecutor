import itertools
import json
import re
import torch as t
from ...Util import dtype, device
from ...Logging import logger
from ...Hyperparams import Hyperparams as params


# special tokens already provided by the tokenizer
mask_token = "<mask>"
target_begin_token = "<extra_id_0>"
target_end_token = "<extra_id_1>"
kind_name_token = "<extra_id_2>"
kind_call_token = "<extra_id_3>"
kind_attribute_token = "<extra_id_4>"
sep_token = "<extra_id_5>"


class InputFactory(object):

    def __init__(self, iids, tokenizer):
        self.iids = iids
        self.tokenizer = tokenizer
        self.file_to_lines = {}
        self.file_to_tokenized_lines = {}

    def __tokenize_lines(self, file_name):
        if file_name in self.file_to_tokenized_lines:
            return self.file_to_lines[file_name], self.file_to_tokenized_lines[file_name]
        else:
            with open(file_name, "r") as f:
                lines = f.readlines()
            self.file_to_lines[file_name] = lines
            tokenized_lines = self.tokenizer(
                lines, return_attention_mask=False, add_special_tokens=False).input_ids
            self.file_to_tokenized_lines[file_name] = tokenized_lines

            if len(self.file_to_tokenized_lines) > 10000:  # prevent OOM
                self.file_to_lines = {}
                self.file_to_tokenized_lines = {}

            return lines, tokenized_lines

    def _encode_input(self, entry, info, lines, tokenized_lines):
        # mark the target name in the corresponding line
        target_line = lines[info[1]-1]
        name = entry["name"]

        match = (re.search(name, target_line[info[2]:]))

        if not match:
            start_index = 0
            end_index = 0
        else:
            start_index = match.span()[0] + info[2]
            end_index = start_index + len(name)

        modified_line = target_line[:start_index] + target_begin_token + \
            name + target_end_token + target_line[end_index:]

        tokenized_target_line = self.tokenizer(
            modified_line, return_attention_mask=False, add_special_tokens=False).input_ids
        tokenized_lines[info[1]-1] = tokenized_target_line
        token_ids = list(itertools.chain(*tokenized_lines))

        # Get at most 512 tokens around the target token
        id_of_target_begin = self.tokenizer.encode(target_begin_token)[1]
        target_index = token_ids.index(id_of_target_begin)

        # fewer context before target
        if target_index < 255:
            previous_target_tokens = token_ids[0:target_index]
            after_target_tokens = token_ids[target_index:target_index +
                                            (512 - len(previous_target_tokens))]
        # fewer context after target
        elif target_index + 255 > len(token_ids):
            after_target_tokens = token_ids[target_index:]
            previous_target_tokens = token_ids[target_index -
                                               (512 - len(after_target_tokens)):target_index]
        # equal context before and after target
        else:
            previous_target_tokens = token_ids[target_index-255:target_index]
            after_target_tokens = token_ids[target_index:target_index+255]

        input_ids = previous_target_tokens + after_target_tokens
        input_ids[0] = self.tokenizer.bos_token_id
        input_ids[-1] = self.tokenizer.eos_token_id

        # Add padding
        if len(input_ids) < 512:
            input_ids = input_ids + \
                (512 - len(input_ids)) * [self.tokenizer.pad_token_id]

        return input_ids

    def _encode_output(self, entry):
        # Create labels
        if not hasattr(entry, "value") or '@' not in entry["value"]:
            value = "unknown"
        else:
            value = entry["value"][1:]

        label_ids = self.tokenizer(
            value, padding="max_length", max_length=params.max_output_length).input_ids
        return label_ids

    def entry_to_inputs(self, entry):
        info = self.iids.iid_to_location[str(entry["iid"]-1)]

        lines, tokenized_lines = self.__tokenize_lines(info[0]+'.orig')

        input_ids = self._encode_input(entry, info, lines, tokenized_lines)
        label_ids = self._encode_output(entry)

        input_ids = t.tensor(input_ids, device='cpu')
        label_ids = t.tensor(label_ids, device='cpu')

        assert len(input_ids) == 512
        assert len(label_ids) == params.max_output_length
        return input_ids, label_ids
