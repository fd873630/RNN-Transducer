import torch
import torch.nn as nn


class BaseDecoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, output_size, n_layers, dropout=0.2):
        super(BaseDecoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )

        self.output_proj = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, length=None, hidden=None):
        embed_inputs = self.embedding(inputs)
        
        batch_size = inputs.size(0)
        max_len = inputs.size(1)

        if length is not None:
            sorted_seq_lengths, indices = torch.sort(length, descending=True)
            
            embed_inputs = embed_inputs[indices]
            embed_inputs = nn.utils.rnn.pack_padded_sequence(
                embed_inputs, sorted_seq_lengths, batch_first=True)
        
        self.lstm.flatten_parameters()
        outputs, hidden = self.lstm(embed_inputs, hidden)    
        
        if length is not None:
            _, desorted_indices = torch.sort(indices, descending=False)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            outputs = outputs[desorted_indices]

        padded_output = torch.zeros(batch_size, max_len, outputs.size(2))
                
        if inputs.is_cuda: padded_output = padded_output.cuda()

        max_output_size = outputs.size(1)
        padded_output[:, :max_output_size, :] = outputs 
        
        #outputs = self.output_proj(outputs)
        outputs = self.output_proj(padded_output)
        
        return outputs, hidden

