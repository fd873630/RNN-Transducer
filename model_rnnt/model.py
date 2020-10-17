import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, autograd

class JointNet(nn.Module):
    def __init__(self, input_size, inner_dim, vocab_size):
        super(JointNet, self).__init__()

        self.forward_layer = nn.Linear(input_size, inner_dim, bias=True)

        self.tanh = nn.Tanh()
        self.project_layer = nn.Linear(inner_dim, vocab_size, bias=True)

    def forward(self, enc_state, dec_state):
        if enc_state.dim() == 3 and dec_state.dim() == 3:
            dec_state = dec_state.unsqueeze(1)
            enc_state = enc_state.unsqueeze(2)
   
            t = enc_state.size(1)
            u = dec_state.size(2)

            enc_state = enc_state.repeat([1, 1, u, 1])
            dec_state = dec_state.repeat([1, t, 1, 1])
        
        else:
            assert enc_state.dim() == dec_state.dim()

        concat_state = torch.cat((enc_state, dec_state), dim=-1)
        outputs = self.forward_layer(concat_state)

        outputs = self.tanh(outputs)
        outputs = self.project_layer(outputs)
        
        return outputs


class Transducer(nn.Module):
    def __init__(self, encoder, decoder, input_size, inner_dim, vocab_size):
        super(Transducer, self).__init__()
        # define model
        self.encoder = encoder
        self.decoder = decoder

        self.input_size = input_size
        self.inner_dim = inner_dim
        self.vocab_size = vocab_size

        # define JointNet
        self.joint = JointNet(input_size=self.input_size, inner_dim=self.inner_dim, vocab_size=self.vocab_size)

    def forward(self, inputs, inputs_lengths, targets, targets_lengths):
        
        zero = torch.zeros((targets.shape[0], 1)).long()
        if targets.is_cuda: zero = zero.cuda()
        
        targets_add_blank = torch.cat((zero, targets), dim=1)
        
        enc_state, _ = self.encoder(inputs, inputs_lengths)
        
        dec_state, _ = self.decoder(targets_add_blank, targets_lengths+1)
        
        logits = self.joint(enc_state, dec_state)

        return logits

    #only one
    def recognize(self, inputs, inputs_length):
    
        batch_size = inputs.size(0)

        enc_states, _ = self.encoder(inputs, inputs_length)

        zero_token = torch.LongTensor([[0]])
        if inputs.is_cuda:
            zero_token = zero_token.cuda()

        def decode(enc_state, lengths):
            token_list = []

            dec_state, hidden = self.decoder(zero_token)

            for t in range(lengths):
                logits = self.joint(enc_state[t].view(-1), dec_state.view(-1))
                out = F.softmax(logits, dim=0).detach()
                pred = torch.argmax(out, dim=0)
                pred = int(pred.item())

                if pred != 0:
                    token_list.append(pred)
                    token = torch.LongTensor([[pred]])

                    if enc_state.is_cuda:
                        token = token.cuda()

                    dec_state, hidden = self.decoder(token, hidden=hidden)

            return token_list

        results = []

        for i in range(batch_size):
            decoded_seq = decode(enc_states[i], inputs_length[i])
            results.append(decoded_seq)

        return results

    #not yet
    def recognize_old(self, inputs, inputs_length):
        
        enc_states, _ = self.encoder(inputs, inputs_length)

        batch_size = inputs.size(0)
      
        zero = torch.zeros((batch_size, 1)).long()

        if inputs.is_cuda: zero = zero.cuda()
     
        def decode(enc_states, lengths):
            token_list = torch.zeros((batch_size, 1)).long().cuda()
            dec_state, hidden = self.decoder(zero)

            pre_list = token_list.squeeze()

            for t in range(enc_states.size(1)):
                logits = self.joint(enc_states[:,t,:].unsqueeze(1), dec_state)
                logits = logits.squeeze()
                out = F.softmax(logits, dim=-1)
                pred = torch.argmax(out, dim=-1)
                
                index = torch.where(pred == 0)[0].cpu().numpy()
                
                pred[index] = pre_list[index] 
                pred_1 = pred.unsqueeze(1)

                dec_state, hidden = self.decoder(pred_1, hidden=hidden)
                token_list = torch.cat((token_list, pred_1), dim=1)

                pre_list = pred

            return token_list

        decoded_seq = decode(enc_states, inputs_length)

        all_results = []

        results = decoded_seq.tolist()
        
        for i in range(len(results)):
            a = results[i]
            a = list(filter((0).__ne__, a))
            all_results.append(a)

        return all_results
