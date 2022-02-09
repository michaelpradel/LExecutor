import torch as t
from torch import nn
import torch.nn.functional as F
from .Hyperparams import Hyperparams as p


class ValuePredictionModel(nn.Module):
    def __init__(self):
        super(ValuePredictionModel, self).__init__()

        self.joining_layer = nn.Linear(
            in_features=4+p.token_emb_len+p.max_call_args*p.value_emb_len+3*p.value_emb_len+p.token_emb_len, out_features=p.value_emb_len)
        self.intermediate_fc = nn.Linear(
            in_features=p.joined_layer_len, out_features=p.intermediate_layer_len)
        self.final_fc = nn.Linear(
            in_features=p.intermediate_layer_len, out_features=p.value_emb_len)
        self.sigmoid = nn.Sigmoid().

    def forward(self, xs):
        xs_kind, xs_name, xs_args, xs_base, xs_left, xs_right, xs_operator = xs

        joined_args = t.cat(xs_args, dim=1)
        all_joined_input = t.cat((xs_kind, xs_name, joined_args, xs_base,
                                 xs_right, xs_operator, xs_left, xs_right, xs_operator), dim=1)

        all_joined_out = F.relu(self.joining_layer(all_joined_input, dim=1))
        intermediate_out = F.relu(self.intermediate_fc(all_joined_out))
        final_out = self.sigmoid(self.final_fc(intermediate_out))

        return final_out
