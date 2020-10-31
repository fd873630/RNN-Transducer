import os
import time
import yaml
import random
import shutil
import argparse
import datetime
import editdistance
import scipy.signal
import numpy as np 

# torch 관련
import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from warprnnt_pytorch import RNNTLoss

from model_rnnt.eval_distance import eval_wer, eval_cer
from model_rnnt.model import Transducer
from model_rnnt.encoder import BaseEncoder
from model_rnnt.decoder import BaseDecoder
from model_rnnt.hangul import moasseugi
from model_rnnt.data_loader_deepspeech import SpectrogramDataset, AudioDataLoader, AttrDict

def load_label(label_path):
    char2index = dict() # [ch] = id
    index2char = dict() # [id] = ch
    with open(label_path, 'r') as f:
        for no, line in enumerate(f):
            if line[0] == '#': 
                continue
            
            index, char = line.split('   ')
            char = char.strip()
            if len(char) == 0:
                char = ' '

            char2index[char] = int(index)
            index2char[int(index)] = char

    return char2index, index2char

def computer_cer(preds, labels):
    char2index, index2char = load_label('./label,csv/hangul.labels')
    
    total_wer = 0
    total_cer = 0

    total_wer_len = 0
    total_cer_len = 0

    for label, pred in zip(labels, preds):
        units = []
        units_pred = []
        for a in label:
            units.append(index2char[a])
            
        for b in pred:
            units_pred.append(index2char[b])

        label = moasseugi(units)
        pred = moasseugi(units_pred)
    
        wer = eval_wer(pred, label)
        cer = eval_cer(pred, label)
        
        wer_len = len(label.split())
        cer_len = len(label.replace(" ", ""))

        total_wer += wer
        total_cer += cer

        total_wer_len += wer_len
        total_cer_len += cer_len

    return total_wer, total_cer, total_wer_len, total_cer_len

def reference(model, val_loader, device):
    model.eval()

    total_loss = 0
   
    total_cer = 0
    total_wer = 0
    
    total_wer_len = 0
    total_cer_len = 0

    with open("./all_reference2.txt", "w") as f:
        f.write('\n')
        f.write("reference 시작")
        f.write('\n')

    char2index, index2char = load_label('./label,csv/hangul.labels')

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            total_batch_num = len(val_loader)
        
            inputs, targets, inputs_lengths, targets_lengths = data

            inputs_lengths = torch.IntTensor(inputs_lengths)
            targets_lengths = torch.IntTensor(targets_lengths)

            inputs = inputs.to(device) # (batch_size, time, freq)
            targets = targets.to(device)
            inputs_lengths = inputs_lengths.to(device)
            targets_lengths = targets_lengths.to(device)
            
            #preds = model.module.recognize(inputs, inputs_lengths)

            transcripts = [targets.cpu().numpy()[i][:targets_lengths[i].item()]
                       for i in range(targets.size(0))]
            
            if False:
                preds, _ = model.beam_search(inputs, inputs_lengths, W=5)
                preds = preds[1:]
                wow = []
                wow.append(preds)
                
                preds = wow
            else:
                preds = model.recognize(inputs, inputs_lengths)

            wer, cer, wer_len, cer_len = computer_cer(preds, transcripts)
            
            total_wer += wer
            total_cer += cer
                
            total_wer_len += wer_len
            total_cer_len += cer_len

            final_wer = (total_wer / total_wer_len) * 100
            final_cer = (total_cer / total_cer_len) * 100

            for a, b in zip(transcripts,preds):
                chars = []
                predic_chars = []
                
                for w in a:
                    chars.append(index2char[w])

                for y in b:
                    predic_chars.append(index2char[y])
                
                with open("./all_reference2.txt", "a") as f:
                    f.write('\n')
                    f.write(moasseugi(chars))
                    f.write('\n')
                    f.write(moasseugi(predic_chars))
                    f.write('\n')
                

    return final_wer, final_cer

def main():
    
    yaml_name = "/home/jhjeong/jiho_deep/rnn-t/label,csv/RNN-T_mobile_2.yaml"

    configfile = open(yaml_name)
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))

    windows = { 'hamming': scipy.signal.hamming,
                'hann': scipy.signal.hann,
                'blackman': scipy.signal.blackman,
                'bartlett': scipy.signal.bartlett
                }

    SAMPLE_RATE = config.audio_data.sampling_rate
    WINDOW_SIZE = config.audio_data.window_size
    WINDOW_STRIDE = config.audio_data.window_stride
    WINDOW = config.audio_data.window

    audio_conf = dict(sample_rate=SAMPLE_RATE,
                        window_size=WINDOW_SIZE,
                        window_stride=WINDOW_STRIDE,
                        window=WINDOW)
    
    random.seed(config.data.seed)
    torch.manual_seed(config.data.seed)
    torch.cuda.manual_seed_all(config.data.seed)

    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    #model
    #Prediction Network
    enc = BaseEncoder(input_size=config.model.enc.input_size,
                      hidden_size=config.model.enc.hidden_size, 
                      output_size=config.model.enc.output_size,
                      n_layers=config.model.enc.n_layers, 
                      dropout=config.model.dropout, 
                      bidirectional=config.model.enc.bidirectional)
    
    #Transcription Network
    dec = BaseDecoder(embedding_size=config.model.dec.embedding_size,
                      hidden_size=config.model.dec.hidden_size, 
                      vocab_size=config.model.vocab_size, 
                      output_size=config.model.dec.output_size, 
                      n_layers=config.model.dec.n_layers, 
                      dropout=config.model.dropout)

    model = Transducer(enc, dec, config.model.joint.input_size, config.model.joint.inner_dim, config.model.vocab_size) 
    
    # 여기 모델 불러오는거
    model.load_state_dict(torch.load("/home/jhjeong/jiho_deep/rnn-t/model_save/model2_save_epoch_20.pth"))
    
    model = model.to(device)
    #model = nn.DataParallel(model).to(device)

    #val dataset
    val_dataset = SpectrogramDataset(audio_conf, 
                                     "/home/jhjeong/jiho_deep/rnn-t/label,csv/AI_hub_reference.csv", 
                                     feature_type=config.audio_data.type,
                                     normalize=True,
                                     spec_augment=False)

    val_loader = AudioDataLoader(dataset=val_dataset,
                                    shuffle=False,
                                    num_workers=config.data.num_workers,
                                    batch_size=1,
                                    drop_last=False)
    
    for i in range(3):
        print(" ")

    print('{} reference 시작'.format(datetime.datetime.now()))

    final_wer, final_cer = reference(model, val_loader, device)

    print("final_wer -> ")
    print(final_wer)
    print("final_cer -> ")
    print(final_cer)
    
    print('{} reference 끝'.format(datetime.datetime.now()))

if __name__ == '__main__':
    main()
    
