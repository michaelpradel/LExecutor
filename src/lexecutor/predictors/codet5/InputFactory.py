import itertools
import json
import re
import torch as t
from ..DLUtil import dtype, device
from ...Logging import logger
from ...Hyperparams import Hyperparams as params


# special tokens already provided by the tokenizer
mask_token = "<mask>"
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

        self.kind_name_token_id = self.tokenizer(
            kind_name_token, add_special_tokens=False, return_attention_mask=False).input_ids[0]
        self.kind_call_token_id = self.tokenizer(
            kind_call_token, add_special_tokens=False, return_attention_mask=False).input_ids[0]
        self.kind_attribute_token_id = self.tokenizer(
            kind_attribute_token, add_special_tokens=False, return_attention_mask=False).input_ids[0]
        self.sep_token_id = self.tokenizer(
            sep_token, add_special_tokens=False, return_attention_mask=False).input_ids[0]

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

    def _extract_context_window(self, token_ids, marker_token):
        # Get at most 512 tokens around the target token
        id_of_target_begin = self.tokenizer.encode(marker_token)[1]
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

        return previous_target_tokens, after_target_tokens


    def _encode_input(self, entry, location, lines, tokenized_lines):
        # format of input:
        # name <sep> kind <sep> pre-context <mask> post-context

        target_line = lines[location.line-1]
            
        start_index = location.column_start
        end_index = location.column_end

        modified_line = target_line[:start_index] + \
            mask_token + target_line[end_index:]

        tokenized_target_line = self.tokenizer(
            modified_line, return_attention_mask=False, add_special_tokens=False).input_ids
        
        # store and later restore the original tokenized line
        # (because we use the same tokenized_lines for all entries)
        original_tokenized_target_line = tokenized_lines[location.line-1]
        tokenized_lines[location.line-1] = tokenized_target_line
        token_ids = list(itertools.chain(*tokenized_lines))
        tokenized_lines[location.line-1] = original_tokenized_target_line

        name = entry["name"]
        name_ids = self.tokenizer(name, return_attention_mask=False,
                                  add_special_tokens=False).input_ids

        previous_target_tokens, after_target_tokens = self._extract_context_window(
            token_ids, mask_token)
        context_ids = previous_target_tokens + after_target_tokens

        # shrink context to fit everything (incl. the variable-sized name_ids) into 512 tokens
        while len(name_ids) + len(context_ids) + 5 > 512:
            context_ids = context_ids[1:-1]

        if entry["kind"] == "name":
            kind_token = self.kind_name_token_id
        elif entry["kind"] == "call":
            kind_token = self.kind_call_token_id
        elif entry["kind"] == "attribute":
            kind_token = self.kind_attribute_token_id

        input_ids = [self.tokenizer.bos_token_id] + \
            name_ids + \
            [self.sep_token_id, kind_token, self.sep_token_id] + \
            context_ids + \
            [self.tokenizer.eos_token_id]

        # Add padding
        if len(input_ids) < 512:
            input_ids = input_ids + \
                (512 - len(input_ids)) * [self.tokenizer.pad_token_id]

        return input_ids

    def _encode_output(self, entry):
        # Create labels
        if not hasattr(entry, "value"):
            # during prediction
            value = "unknown"
        else:
            # during training
            assert entry["value"].startswith("@"), entry["value"]
            value = entry["value"][1:]

        label_ids = self.tokenizer(
            value, padding="max_length", max_length=params.max_output_length).input_ids
        return label_ids

    def entry_to_inputs(self, entry):
        location = self.iids.location(str(entry["iid"]))

        lines, tokenized_lines = self.__tokenize_lines(location.file+'.orig')

        input_ids = self._encode_input(entry, location, lines, tokenized_lines)
        label_ids = self._encode_output(entry)

        input_ids = t.tensor(input_ids, device='cpu')
        label_ids = t.tensor(label_ids, device='cpu')

        assert len(input_ids) == 512, len(input_ids)
        assert len(label_ids) == params.max_output_length, len(label_ids)
        return input_ids, label_ids
