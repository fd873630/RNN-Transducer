import os
import time
import yaml
import random
import shutil
import argparse
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

from tensorboardX import SummaryWriter

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

def train(model, train_loader, optimizer, criterion, device):
    model.train()

    total_loss = 0
    start_time = time.time()
    total_batch_num = len(train_loader)
    for i, data in enumerate(train_loader):
        
        optimizer.zero_grad()

        inputs, targets, inputs_lengths, targets_lengths = data

        inputs_lengths = torch.IntTensor(inputs_lengths)
        targets_lengths = torch.IntTensor(targets_lengths)

        inputs = inputs.to(device) # (batch_size, time, freq)
        targets = targets.to(device)
        inputs_lengths = inputs_lengths.to(device)
        targets_lengths = targets_lengths.to(device)

        logits = model(inputs, inputs_lengths, targets, targets_lengths)
        
        loss = criterion(logits, targets.int(), inputs_lengths.int(), targets_lengths.int())

        max_inputs_length = inputs_lengths.max().item()
        max_targets_length = targets_lengths.max().item()
        
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()

        if i % 500 == 0:
            print('train_batch: {:4d}/{:4d}, train_loss: {:.4f}, train_time: {:.4f}'.format(i, total_batch_num, loss.item(), time.time() - start_time))
            start_time = time.time()

    train_loss = total_loss / total_batch_num

    return train_loss

def eval(model, val_loader, criterion, device):
    model.eval()

    total_loss = 0
    total_batch_num = len(val_loader)
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, targets, inputs_lengths, targets_lengths = data

            inputs_lengths = torch.IntTensor(inputs_lengths)
            targets_lengths = torch.IntTensor(targets_lengths)

            inputs = inputs.to(device) # (batch_size, time, freq)
            targets = targets.to(device)
            inputs_lengths = inputs_lengths.to(device)
            targets_lengths = targets_lengths.to(device)
                
            logits = model(inputs, inputs_lengths, targets, targets_lengths)
            loss = criterion(logits, targets.int(), inputs_lengths.int(), targets_lengths.int())
    
            total_loss += loss.item()
    
        val_loss = total_loss / total_batch_num

    return val_loss

def main():
    
    with open("./train.txt", "w") as f:
        f.write("학습 시작")
        f.write('\n')

    configfile = open("/home/jhjeong/jiho_deep/rnn-t/label,csv/RNN-T_mobile.yaml")
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))

    summary = SummaryWriter()
    summary1 = SummaryWriter()

    windows = {
    'hamming': scipy.signal.hamming,
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

    train_manifest_filepath = config.data.train_path
    val_manifest_filepath = config.data.val_path
    
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
    
    model = nn.DataParallel(model).to(device)

    if config.optim.type == "AdamW":
        optimizer = optim.AdamW(model.module.parameters(), 
                                lr=config.optim.lr, 
                                weight_decay=config.optim.weight_decay)
    
    elif config.optim.type == "Adam":
        optimizer = optim.Adam(model.module.parameters(), 
                               lr=config.optim.lr, 
                               weight_decay=config.optim.weight_decay)
    else:
        pass

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                                               milestones=config.optim.milestones, 
                                               gamma=config.optim.decay_rate)

    criterion = RNNTLoss().to(device)
    """
    acts: Tensor of [batch x seqLength x (labelLength + 1) x outputDim] containing output from network
    (+1 means first blank label prediction)
    labels: 2 dimensional Tensor containing all the targets of the batch with zero padded
    act_lens: Tensor of size (batch) containing size of each output sequence from the network
    label_lens: Tensor of (batch) containing label length of each example
    """

    #train dataset
    train_dataset = SpectrogramDataset(audio_conf, 
                                       config.data.train_path,
                                       feature_type=config.audio_data.type, 
                                       normalize=True, 
                                       spec_augment=False)


    train_loader = AudioDataLoader(dataset=train_dataset,
                                    shuffle=True,
                                    num_workers=config.data.num_workers,
                                    batch_size=config.data.batch_size,
                                    drop_last=True)
    #val dataset
    val_dataset = SpectrogramDataset(audio_conf, 
                                     config.data.val_path, 
                                     feature_type=config.audio_data.type,
                                     normalize=True)

    val_loader = AudioDataLoader(dataset=val_dataset,
                                    shuffle=True,
                                    num_workers=config.data.num_workers,
                                    batch_size=config.data.batch_size,
                                    drop_last=True)

    print("학습을 시작합니다.")
    
    pre_val_loss = 100000
    for epoch in range(config.training.begin_epoch, config.training.end_epoch):
        
        train_time = time.time()
        train_loss = train(model, train_loader, optimizer, criterion, device)
        train_total_time = time.time()-train_time
        print('Epoch %d (Training) Loss %0.4f time %0.4f' % (epoch+1, train_loss, train_total_time))
        
        #summary.add_scalar('Loss', train_loss)
        
        eval_time = time.time()
        val_loss = eval(model, val_loader, criterion, device)
        eval_total_time = time.time()-eval_time
        print('Epoch %d (val) Loss %0.4f time %0.4f' % (epoch+1, val_loss, eval_total_time))       
        scheduler.step()

        #summary.add_scalar('Loss', val_loss)
        #summary1.add_scalar('cer', val_cer)

        with open("./train.txt", "a") as ff:
            ff.write('Epoch %d (Training) Loss %0.4f time %0.4f' % (epoch+1, train_loss, train_total_time))
            ff.write('\n')
            ff.write('Epoch %d (val) Loss %0.4f time %0.4f ' % (epoch+1, val_loss, eval_total_time))
            ff.write('\n')
            ff.write('\n')
        
        if pre_val_loss > val_loss:
            print("best model을 저장하였습니다.")
            torch.save(model.module.state_dict(), "./model_save/model_save.pth")
            pre_val_loss = val_loss
        
if __name__ == '__main__':
    main()
