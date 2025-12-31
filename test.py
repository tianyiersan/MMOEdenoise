import torch.nn as nn
import os
from torch.functional import F
import torch
if __name__ == '__main__':
    class SRSMLP(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(SRSMLP, self).__init__()
            self.linear_one = nn.Linear(input_size, hidden_size)
            self.linear_two = nn.Linear(hidden_size, output_size)
            self.linear_three = nn.Linear(hidden_size, output_size)

        def forward(self,hidden, x):
            # x = F.relu(self.fc1(x))
            # x = self.fc2(x)
            mm = hidden[:, -x:-1]
            ht  = hidden[:, -x:-1].sum(dim=1)
            q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
            q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
            alpha = self.linear_three(torch.sigmoid(q1 + q2))
            # a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
            return alpha


    class SessionLengthSearch(nn.Module):
        def __init__(self, num_candidates, input_size, hidden_size, output_size):
            super(SessionLengthSearch, self).__init__()
            self.candidate_models = nn.ModuleList(
                [SRSMLP(input_size, hidden_size, output_size) for _ in range(num_candidates)])
            self.architectural_weights = nn.Parameter(torch.randn(num_candidates))

        def forward(self, x):
            hidden = torch.randn([100,69,100])
            outputs = torch.stack([model(hidden,x) for model in self.candidate_models])
            weighted_outputs = F.softmax(self.architectural_weights, dim=0) * outputs
            final_output = weighted_outputs.sum(0)
            return final_output


    model = SessionLengthSearch(num_candidates=5, input_size=100, hidden_size=100, output_size=100)
    # data = torch.randn([5,100])
    b = model(5)