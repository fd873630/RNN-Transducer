import torch
import torch.nn as nn


class BaseEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, dropout=0.2, bidirectional=False):
        super(BaseEncoder, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

        self.output_proj = nn.Linear(2 * hidden_size if bidirectional else hidden_size,
                                     output_size, bias=True)
        


    def forward(self, inputs, input_lengths):
        assert inputs.dim() == 3
        
        batch_size = inputs.size(0)
        max_len = inputs.size(1)
        
        if input_lengths is not None:
            sorted_seq_lengths, indices = torch.sort(input_lengths, descending=True)
            inputs = inputs[indices]
            inputs = nn.utils.rnn.pack_padded_sequence(inputs, sorted_seq_lengths, batch_first=True)
        
        self.lstm.flatten_parameters()
        outputs, hidden = self.lstm(inputs)
        
        if input_lengths is not None:
            _, desorted_indices = torch.sort(indices, descending=False)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            outputs = outputs[desorted_indices]
        
        padded_output = torch.zeros(batch_size, max_len, outputs.size(2))
        
        if inputs.is_cuda: padded_output = padded_output.cuda()

        max_output_size = outputs.size(1)
        padded_output[:, :max_output_size, :] = outputs        
        
        #logits = self.output_proj(outputs)
        logits = self.output_proj(padded_output)

        return logits, hidden

