
# Домашнее задание: сделать модель, которая может переводить тексты с немецкого языка в англиский. 
# Для обучения будет использоваться датасет [wmt-14](https://huggingface.co/datasets/wmt14). 

#------------------Imports-------------------

import torch
import torch.nn as nn
from IPython.display import FileLink
from somajo import SoMaJo
import math
import evaluate
from datasets import load_dataset
from tqdm.notebook import tqdm
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


#------------------Download Data-------------------
wmt14 = load_dataset("wmt14", "de-en")

train_data = wmt14['train'].map(lambda x: {'len':len(x['translation']['de'].split())})
train_data = train_data.sort('len')
train_data = wmt14['train'].select([i for i in range(450000)])

tokenizer_de = SoMaJo("de_CMC", split_camel_case=True)
tokenizer_en = SoMaJo(language="en_PTB")


def prepare(data):

    de = tokenizer_de.tokenize_text([data['translation']['de']])
    en = tokenizer_en.tokenize_text([data['translation']['en']])

    return {'de':[token.text for sent in de for token in sent ], 
            'en':[token.text for sent in en for token in sent ]}


train_data = train_data.map(prepare)
test_data = wmt14['test'].map(prepare)
validation_data = wmt14['validation'].map(prepare)


#------------------Prepare Data-------------------
PAD = 0
BOS = 1
EOS = 2
UNK = 3

class AttrDict(dict):
 
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self

class TranslationDataset(Dataset):
    def __init__(self, dataset,  src_vocab=None, tgt_vocab=None, 
                 max_vocab_size=50000, min_freq=10, max_length=100):

        self.dataset = dataset
        self.min_freq = min_freq
        self.max_length = max_length

        def build_counters(sents):
            counter_tgt = Counter()
            counter_src = Counter()
            for sent in tqdm(sents, file=sys.stdout):
                counter_src.update(sent['de'])
                counter_tgt.update(sent['en'])
            return counter_src, counter_tgt

        if src_vocab is None or tgt_vocab is None:
            print('- Building counters...')
            self.src_counter, self.tgt_counter = build_counters(dataset)

            print('- Building source vocabulary...')
            self.src_vocab = self.build_vocab(self.src_counter, max_vocab_size)
            print('- Building target vocabulary...')
            self.tgt_vocab = self.build_vocab(self.tgt_counter, max_vocab_size)
        else:
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab         

        print('='*100)
        print('Dataset Info:')
        print('- Source vocabulary size: {}'.format(len(self.src_vocab.token2id)))
        print('- Target vocabulary size: {}'.format(len(self.tgt_vocab.token2id)))
        print('='*100 + '\n')
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        src_sent = self.dataset.select([index])['de'][0][:self.max_length-2]
        tgt_sent = self.dataset.select([index])['en'][0][:self.max_length-2]
        src_seq = self.tokens2ids(src_sent, self.src_vocab.token2id, 
                                  append_BOS=True, append_EOS=True)
        tgt_seq = self.tokens2ids(tgt_sent, self.tgt_vocab.token2id, 
                                  append_BOS=True, append_EOS=True)

        return src_seq, tgt_seq


    def build_vocab(self, counter, max_vocab_size):
        vocab = AttrDict()
        vocab.token2id = {'<PAD>': PAD, '<BOS>': BOS, '<EOS>': EOS, '<UNK>': UNK}
        vocab.token2id.update({token: _id+4 for _id, (token, count) in 
                               tqdm(enumerate(counter.most_common(max_vocab_size)), 
                               file=sys.stdout) if count >= self.min_freq})
        vocab.id2token = {v:k for k,v in tqdm(vocab.token2id.items(), file=sys.stdout)}    
        return vocab
    
    def tokens2ids(self, tokens, token2id, append_BOS=True, append_EOS=True):
        seq = []
        if append_BOS: seq.append(BOS)
        seq.extend([token2id.get(token, UNK) for token in tokens])
        if append_EOS: seq.append(EOS)
        return seq


def collate_fn(data):

    def _pad_sequences(seqs):
        lens = [len(seq) for seq in seqs]
        padded_seqs = torch.zeros(len(seqs), max(lens)).long()
        for i, seq in enumerate(seqs):
            end = lens[i]
            padded_seqs[i, :end] = torch.LongTensor(seq[:end])
        return padded_seqs, lens

    data.sort(key=lambda x: len(x[0]), reverse=True)

    src_seqs, tgt_seqs = zip(*data)
    
    src_seqs, src_lens = _pad_sequences(src_seqs)
    tgt_seqs, tgt_lens = _pad_sequences(tgt_seqs)
    

    return src_seqs, tgt_seqs


#------------------Define Model-------------------
class EncoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim,  
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        _src, _ = self.self_attention(src, src, src, src_mask)
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        _src = self.positionwise_feedforward(src)
        src = self.ff_layer_norm(src + self.dropout(_src))
        return src


class Encoder(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim,
                 dropout, 
                 device,
                 max_length = 100):
        super().__init__()

        self.device = device
        
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim,
                                                  dropout, 
                                                  device) 
                                     for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, src, src_mask):
        batch_size = src.shape[0]
        src_len = src.shape[1]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        for layer in self.layers:
            src = layer(src, src_mask)
        return src


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        
        batch_size = query.shape[0]
           
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
    
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
     
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
                
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim = -1)

        x = torch.matmul(self.dropout(attention), V)
                
        x = x.permute(0, 2, 1, 3).contiguous()
                
        x = x.view(batch_size, -1, self.hid_dim)
        
        x = self.fc_o(x)
                
        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):

        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
            
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
                    
        _trg = self.positionwise_feedforward(trg)
        
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
 
        return trg, attention


class Decoder(nn.Module):
    def __init__(self, 
                 output_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device,
                 max_length = 100):
        super().__init__()
        
        self.device = device
        
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim, 
                                                  dropout, 
                                                  device)
                                     for _ in range(n_layers)])
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):   
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
    
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
    
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        
        output = self.fc_out(trg)
                    
        return output, attention


class TranslationModel(nn.Module):
    def __init__(self, 
                 encoder, 
                 decoder, 
                 src_pad_idx, 
                 trg_pad_idx, 
                 device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):

        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        return src_mask
    
    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool() 
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def forward(self, src, trg):      
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output, attention


#------------------Get data and model-------------------
MAX_LENGTH = 200
batch_size = 128

train_dataset = TranslationDataset(train_data, max_length = MAX_LENGTH, min_freq=20)

valid_dataset = TranslationDataset(validation_data, 
                                   src_vocab=train_dataset.src_vocab,
                                    tgt_vocab=train_dataset.tgt_vocab, 
                                    max_length = MAX_LENGTH)

train_iter = DataLoader(dataset=train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=2,
                        collate_fn=collate_fn)

valid_iter = DataLoader(dataset=valid_dataset,
                        batch_size=batch_size, 
                        shuffle=True,
                        num_workers=2,
                        collate_fn=collate_fn)


INPUT_DIM = len(train_dataset.src_vocab.token2id)
OUTPUT_DIM = len(train_dataset.tgt_vocab.token2id)
HID_DIM = 512
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
LEARNING_RATE = 0.0001
MAX_LENGTH = 200


enc = Encoder(INPUT_DIM, 
              HID_DIM, 
              ENC_LAYERS, 
              ENC_HEADS, 
              ENC_PF_DIM, 
              ENC_DROPOUT, 
              device,
              MAX_LENGTH)

dec = Decoder(OUTPUT_DIM, 
              HID_DIM, 
              DEC_LAYERS, 
              DEC_HEADS, 
              DEC_PF_DIM, 
              DEC_DROPOUT, 
              device,
              MAX_LENGTH)

model = TranslationModel(enc, dec, PAD, PAD, device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index = PAD)
print(f'model has {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6}M params')


def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    for batch in tqdm(iterator):
        
        src = batch[0].to(device)
        trg = batch[1].to(device)
        
        optimizer.zero_grad()
        
        output, _ = model(src, trg[:,:-1])
        output_dim = output.shape[-1]
            
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)
                 
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)



def evaluate(model, iterator, criterion):
    
    model.eval()

    epoch_loss = 0
    
    with torch.no_grad():
    
        for batch in tqdm(iterator):

            src = batch[0].to(device)
            trg = batch[1].to(device)

            output, _ = model(src, trg[:,:-1])
            output_dim = output.shape[-1]
            
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)
 
            loss = criterion(output, trg)
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


#------------------Train-------------------
N_EPOCHS = 30
CLIP = 1

best_valid_loss = float('inf')

for epoch in tqdm(range(N_EPOCHS)):
    
    train_loss = train(model, train_iter, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iter, criterion)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), f'model_{epoch}.pt')
        FileLink('weights.pt')

    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


#------------------Inference-------------------
def translate_sentence(sentence, tokenizer, model, device, max_len = MAX_LENGTH):
    
    model.eval()
        
    if isinstance(sentence, str):
        tokens = [token.text for sent in tokenizer([sentence]) for token in sent ]
    else:
        tokens = [token.text for sent in tokenizer(sentence) for token in sent ]

    tokens = [BOS] + tokens + [EOS]
    
    src_indexes = train_dataset.tokens2ids(tokens,train_dataset.src_vocab.token2id) 

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    
    src_mask = model.make_src_mask(src_tensor)
    
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [BOS]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)
        
        with torch.no_grad():
            output, _ = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        
        pred_token = output.argmax(2)[:,-1].item()
        
        trg_indexes.append(pred_token)

        if pred_token == EOS:
            break
    
    trg_tokens = [train_dataset.tgt_vocab.id2token[i] for i in trg_indexes]
    
    return ' '.join(trg_tokens[1:-1])


src = '##TEXT TO TRANSLATE##'
translation = translate_sentence(src, tokenizer_de.tokenize_text, model, device)
