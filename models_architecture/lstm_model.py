import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class ATTENTION(nn.Module):
    def __init__(self, hidden_dim):
        super(ATTENTION, self).__init__()
        self.W1 = nn.Linear(hidden_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
    
    def forward(self, hidden, encoder_outputs):
        h = hidden[0]
        forward = h[-2, :, :]
        backward = h[-1, :, :]
        last_hidden = torch.cat([forward, backward], dim=1)
        last_hidden = last_hidden.unsqueeze(1)
        
        score = self.V(torch.tanh(self.W1(encoder_outputs) + self.W2(last_hidden)))
        attention_weights = torch.softmax(score, dim=1)
        context = torch.sum(attention_weights * encoder_outputs, dim=1)
        
        return context

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class LSTMENCODERDECODER(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256, num_layers=1, dropout=0.3, pad_idx=0, use_attention=True, use_vae=False):
        super(LSTMENCODERDECODER, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pad_idx = pad_idx
        self.use_attention = use_attention
        self.use_vae = use_vae
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=5000)
        
        self.encoder = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        if use_attention:
            self.attention = ATTENTION(hidden_dim * 2)
        
        if use_vae:
            self.fc_mu = nn.Linear(hidden_dim * 2, hidden_dim * 2)
            self.fc_logvar = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        
        self.bridge = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        
        self.decoder = nn.LSTM(
            embed_dim, hidden_dim * 2, num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc_out = nn.Linear(hidden_dim * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() >= 2:
                    if 'lstm' in name:
                        # Better LSTM init
                        if 'weight_ih' in name:
                            nn.init.xavier_uniform_(param)
                        elif 'weight_hh' in name:
                            nn.init.orthogonal_(param)
                    else:
                        nn.init.xavier_uniform_(param)
                else:
                    nn.init.ones_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x, lengths=None, teacher_forcing_ratio=0.5):
        batch_size, seq_len = x.shape
        
        encoder_outputs, encoder_hidden = self.encode(x, lengths)
        
        if self.use_attention:
            context = self.attention(encoder_hidden, encoder_outputs)
        else:
            forward_hidden = encoder_hidden[0][-2, :, :]
            backward_hidden = encoder_hidden[0][-1, :, :]
            context = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        kl_loss = 0
        if self.use_vae:
            mu = self.fc_mu(context)
            logvar = self.fc_logvar(context)
            context = self.reparameterize(mu, logvar)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        context = self.bridge(context)
        context = torch.relu(context)
        decoder_hidden = self.init_decoder_hidden(context)
        
        outputs = []
        decoder_input = x[:, 0].unsqueeze(1)
        
        for t in range(1, seq_len):
            embedded = self.dropout(self.embedding(decoder_input))
            decoder_output, decoder_hidden = self.decoder(embedded, decoder_hidden)
            decoder_output = self.layer_norm(decoder_output)
            logits = self.fc_out(decoder_output)
            outputs.append(logits)
            
            if np.random.random() < teacher_forcing_ratio:
                decoder_input = x[:, t].unsqueeze(1)
            else:
                decoder_input = logits.argmax(dim=-1)
        
        outputs = torch.cat(outputs, dim=1)
        return outputs, kl_loss

    def encode(self, x, lengths=None):
        embedded = self.dropout(self.embedding(x))
        embedded = self.pos_encoding(embedded)
        
        if lengths is not None:
            packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
            output, hidden = self.encoder(packed)
            output, _ = pad_packed_sequence(output, batch_first=True)
        else:
            output, hidden = self.encoder(embedded)
        
        return output, hidden

    def get_query_vector(self, x, lengths=None):
        with torch.no_grad():
            encoder_outputs, encoder_hidden = self.encode(x, lengths)
            
            if self.use_attention:
                query_vec = self.attention(encoder_hidden, encoder_outputs)
            else:
                forward = encoder_hidden[0][-2, :, :]
                backward = encoder_hidden[0][-1, :, :]
                query_vec = torch.cat([forward, backward], dim=1)
            
            if self.use_vae:
                query_vec = self.fc_mu(query_vec)
            
            return query_vec

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def init_decoder_hidden(self, context):
        hidden = context.unsqueeze(0).repeat(self.num_layers, 1, 1)
        cell = torch.zeros_like(hidden)
        return (hidden, cell)