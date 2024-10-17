# MULTIHEAD ATTENTION 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.utils.logger import logger

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        try:
            assert d_model % num_heads == 0
        except Exception as e:
            logger.error("dimension of the embedding model is not divisable by number of heads")
        
        self.d_models = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        # The query, key, value learnable matrices
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)

        self.FCLayer = nn.Linear(d_model, d_model)
    def split_embedding_perHead(self,x):
        # x shape is (batch_size, seq_len, d_model)
        (batch_size, seq_len, d_model) = x.shape
        # logger.info(f"multi-head; x-shape: {x.shape}")
        # let's reshape to (batch_size, seq_len, num_heads, depth)
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        logger.info(f"Multi-head; x reshaped: {x.shape} ")
        # changing the dimensions order to:(batch_size, num_heads, seq_len, depth)
        x = x.permute(0,2,1,3)
        return x
    
    def cal_attention(self,q,k,v,mask):
        qk = torch.matmul(q, k.permute(0,1,3,2))
        dk=torch.tensor(k.shape[-1], dtype=torch.float32)
        #dk is a tensor scalar!
        attention = qk/torch.sqrt(dk)

        # print("CHECKING PADDING MASK CODE")
        # print("q shape", q.shape)
        # print("attention shape", attention.shape)
        if mask is not None:
            attention += (mask*-1e9)
        # print("attention values after masking", attention[0,0,:,:])
        attention_weights = F.softmax(attention, dim=-1) # should be applied along the sequence which is the 3rd dimension
        output = torch.matmul(attention_weights, v)

        return output, attention_weights
    
    def forward(self, v,k,q,mask):
        batch_size = q.shape[0]
        # shapes for debugging
        # print("v shape", v.shape)

        q = self.split_embedding_perHead(self.Wq(q))
        k = self.split_embedding_perHead(self.Wk(k))
        v = self.split_embedding_perHead(self.Wv(v))

        attention,atten_weights = self.cal_attention(q,k,v,mask)
        attention = attention.permute(0,2,1,3).contiguous()
        attention = attention.reshape(batch_size, -1, self.d_models)

        output = self.FCLayer(attention)
        return output


# THE ENCODER LAYER
class EncoderLayer(nn.Module):
    def __init__(self,d_model,dff):
        super(EncoderLayer,self).__init__()
        self.FeedForwardNN = nn.Sequential(
            nn.Linear(d_model,dff),
            nn.ReLU(),
            nn.Linear(dff,dff)
        )

    def forward(self,x):
        output = self.FeedForwardNN(x)
        logger.info(f"encoder output dimensions {output.shape}")
        return output
    

class EncoderLayer(nn.Module):
    def __init__(self,d_model, num_heads, dff):
        super(EncoderLayer,self).__init__()
        self.MultiHAttention = MultiHeadAttention(d_model, num_heads)
        self.FeedForwardNN = nn.Sequential(
            nn.Linear(d_model,dff),
            nn.ReLU(),
            nn.Linear(dff,d_model)

        )
        self.layerNorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layerNorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layerNorm3 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        # print(f"FIRST MHA WITH LOOK AHEAD MASK")
        attn_output1 = self.MultiHAttention1(x,x,x,look_ahead_mask)
        attn_output1 = self.layerNorm1(x+attn_output1)
        # print(f"decoder input into second multihead attention layer:{attn_output1.shape}")
        attn_output2 = self.MultiHAttention2(enc_output, enc_output,attn_output1, padding_mask)
        attn_output2 = self.layerNorm2(attn_output2+attn_output1)

        Feedforward_output = self.FeedForwardNN(attn_output2)
        final_output = self.layerNorm3(attn_output2+Feedforward_output)
        return final_output
    


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = nn.Embedding(target_vocab_size, d_model) # d_model is the size of embedding vector
        self.pos_encoding = self.positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dff) for _ in range(num_layers)])

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return torch.tensor(pos_encoding, dtype=torch.float32)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        logger.info(f"decoder input shape to the embedding: {x.shape}")
        seq_len = x.size(1)
        x = self.embedding(x)
        logger.info(f"decoder input shape after embedding: {x.shape}")
        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x += self.pos_encoding[:, :seq_len, :]

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, enc_output, look_ahead_mask, padding_mask)

        logger.info(f"final decoder output shape {x.shape}")
        return x


# TRANSFORMER

class Transformer(nn.Module):
    def __init__(self,num_layers, enc_d_model, dec_d_model,
                enc_num_heads, dec_num_heads, enc_dff, 
                dec_dff, target_vocab_size, pe_target):
        super(Transformer, self).__init__()

        # self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size)
        self.encoder = EncoderLayer(enc_d_model, enc_dff)
        self.decoder = Decoder(num_layers, dec_d_model, dec_num_heads, dec_dff, target_vocab_size, pe_target)
        self.final_layer = nn.Linear(dec_d_model, target_vocab_size)

        
    def forward(self, properties, target, look_ahead_mask, dec_padding_mask, training):
        logger.info("ENCODER STARTED")
        enc_output = self.encoder(properties)
        logger.info("ENCODER COMPLETED")
        # currently the encoder output will be [batch_size, 1, d_model] i.e. sequence of size 1
        # to ensure it is compatable with the decoder MHA first layer, 
        # we need to expand sequence length to same length as target
        enc_output_reshaped = enc_output.unsqueeze(1).repeat(1, target.shape[1],1)
        logger.info(f"encoder output dimensions:{enc_output.shape}")
        logger.info(f"encoder output reshaped: {enc_output_reshaped.shape}")
        logger.info("DECODER STARTED")

        dec_output = self.decoder(target, enc_output_reshaped, look_ahead_mask, dec_padding_mask)
        ffl_output = self.final_layer(dec_output)

        #####during training:
        if training:
            return ffl_output
        
        else:

        ##### During inference::
        # # the ffl output is is of shape [batch, seq_len, target_vocab_size]
        # # the last dimension will need to be passed through a softmax to determine 
        # # the most likely token
            # print("transformer output logits: ", ffl_output.shape)
            probabilities = F.softmax(ffl_output, dim=-1)
            # print("probabilities: ", probabilities.shape)
            # To get the predicted tokens
            predicted_tokens = torch.argmax(probabilities, dim=-1)
            # print("final token", predicted_tokens.shape)
            return predicted_tokens