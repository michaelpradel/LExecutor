import torch as t
from torch import nn
import torch.nn.functional as F
from ...Hyperparams import Hyperparams as p


class ValuePredictionModel(nn.Module):
    def __init__(self):
        super(ValuePredictionModel, self).__init__()

        self.intermediate_fc = nn.Linear(
            in_features=4+p.token_emb_len+p.max_call_args*p.value_emb_len+3*p.value_emb_len+p.token_emb_len, out_features=p.intermediate_layer_len)
        self.final_fc = nn.Linear(
            in_features=p.intermediate_layer_len, out_features=p.value_emb_len)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, xs):
        xs_kind, xs_name, xs_args, xs_base, xs_left, xs_right, xs_operator = xs

        joined_args = xs_args.view(
            (p.batch_size, p.max_call_args*p.value_emb_len))
        all_joined = t.cat((xs_kind, xs_name, joined_args, xs_base, xs_left,
                            xs_right, xs_operator), dim=1)

        intermediate_out = F.relu(self.intermediate_fc(all_joined))
        final_out = self.softmax(self.final_fc(intermediate_out))

        return final_out
