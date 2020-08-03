import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
from params import Params
from utils import Vocab, Hypothesis, word_detector
from typing import Union, List
import copy, math, time
from torch.autograd import Variable
import numpy as np
# from nn import Transformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eps = 1e-31

def clones(module, N):
    return nn.ModuleList(copy.deepcopy(module) for _ in range(N))

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def make_std_mask(tgt, pad):
    "Create a mask to hide padding and future words."
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(
        subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    return tgt_mask

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std1 = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std1 + self.eps) + self.b_2

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, ff_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, ff_size)
        self.w_2 = nn.Linear(ff_size, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, embed_size, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # print("PE embed_size")
        # print(embed_size)
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) *
                             -(math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    # print("Attention")
    # print(query.shape)
    # print(key.shape)
    # print(value.shape)
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)

    # print("Mask shape")
    # print(mask.shape)
    # print("Scores shape")
    # print(scores.shape)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # print("Attention shape")    
    p_attn = F.softmax(scores, dim = -1)
    # print(torch.matmul(p_attn, value).shape)
    at = torch.sum(p_attn, dim=1)

    #  print(at.shape)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        # print(self.attn.shape)
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        # print(x.shape)
        return self.linears[-1](x), self.attn

class EncoderLayer(nn.Module):
    def __init__(self, size, ff_size, n_heads, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(n_heads, size)
        self.feed_forward = PositionwiseFeedForward(size, ff_size, dropout)
        self.norm = clones(LayerNorm(size), 2)
        self.dropout = nn.Dropout(dropout)
        self.size = size

    def forward(self, x, mask):
        x_norm = self.norm[0](x)
        res, p_attn = self.self_attn(x_norm, x_norm, x_norm, mask)
        res1 = x + self.dropout(res)
        res_norm = self.norm[1](res1)
        res2 = res1 + self.dropout(self.feed_forward(res_norm))
        return res2, p_attn


class Encoder(nn.Module):
    def __init__(self, hidden_size, n_heads, ff_size, dropout, n_layers):
        super(Encoder, self).__init__()
        
        self.encoderLayer = EncoderLayer(hidden_size, ff_size, n_heads, dropout)
        self.layers = clones(self.encoderLayer, n_layers)
        self.norm = LayerNorm(hidden_size)

    def forward(self, x, mask):
        # x = self.embed_layer(x)
        for layer in self.layers:
            x, p_attn = layer(x, mask)
        return self.norm(x), p_attn

class DecoderLayer(nn.Module):
    def __init__(self, size, ff_size, n_heads, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.norm = clones(LayerNorm(size), 3)
        self.self_attn = MultiHeadedAttention(n_heads, size)
        self.src_attn = MultiHeadedAttention(n_heads, size)
        self.feed_forward = PositionwiseFeedForward(size, ff_size, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory

        x_norm = self.norm[0](x)
        res, p_attn = self.self_attn(x_norm, x_norm, x_norm, tgt_mask)
        res = x+self.dropout(res)
        res_norm = self.norm[1](res)
        # print("Tgt shape: \n")
        # print(res_norm.shape)
        # print("Memory shape: \n")
        # print(m.shape)
        res_attn, p_attn = self.src_attn(res_norm, m, m, src_mask)
        res_attn = self.dropout(res_attn)
        res = res + res_attn
        res1 =res + self.dropout(self.feed_forward(self.norm[2](res)))
        return res1, p_attn

class Decoder(nn.Module):
    def __init__(self, hidden_size, n_heads, ff_size, dropout, n_layers):
        super(Decoder, self).__init__()
        self.decoderLayer = DecoderLayer(hidden_size, ff_size, n_heads, dropout)
        self.layers = clones(self.decoderLayer, n_layers)
        self.norm = LayerNorm(hidden_size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x, p_attn = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x), p_attn

class Seq2SeqOutput(object):

  def __init__(self, encoder_output: torch.Tensor, decoder_output: torch.Tensor,
               decoded_tokens: torch.Tensor, loss: Union[torch.Tensor, float]=0,
               loss_value: float=0, enc_attn_weights: torch.Tensor=None,
               ptr_probs: torch.Tensor=None):
    self.encoder_output = encoder_output
    self.decoder_output = decoder_output
    self.decoded_tokens = decoded_tokens  # (out seq len, batch size)
    self.loss = loss  # scalar
    self.loss_value = loss_value  # float value, excluding coverage loss
    self.enc_attn_weights = enc_attn_weights  # (out seq len, batch size, src seq len)
    self.ptr_probs = ptr_probs  # (out seq len, batch size)

def get_initial_probs(vocab_size, initial_token_idx):
        """Generate initial probability distribution for vocabulary.

        Args:
            vocab_size (int): Size of vocabulary.
            initial_token_idx (int): Initial token index.

        Returns:
            torch.Tensor: float tensor of shape `(1, vocab_size)`.

        """
        probs = torch.zeros(1, vocab_size)
        probs[0, initial_token_idx] = 1
        return probs.float()

class TransformerPointgen(nn.Module):
    """
    Transformers with an optional pointer generator network and coverage mechansim for 
    question generation
    """
    def __init__(self, vocab: Vocab, params: Params, max_dec_steps=None):

        """
        @@args:
            max_seq_len (int): maximum length of input sequence
            vocab_size (int): size of the vocabulary used: Not extended vocabulary
            initial_idx (int, optional): initial token index
            embedding_weights (torch.Tensor, optional): weights of embeddings matrix
            n_layers (int, optional): Number of encoder decoder layers (default: 8)
            emb_size (int, optional): Size of the word embeddings
            dim_m (int, optional): Size of the hidden dimension
            n_heads (int, optional): Number of attention heads (default: 8)
            dim_i (int, optional): inner dimension of position-wise sublayer.
            dropout (float, optional): dropout probability.
        """

        super(TransformerPointgen, self).__init__()

        self.pad = vocab.PAD
        self.UNK = vocab.UNK
        self.vocab_size = len(vocab)
        if(vocab.embeddings is not None):
            self.embed_size = vocab.embeddings.shape[1]
            if(params.embed_size is not None and self.embed_size != params.embed_size):
                print("Warning: Model embedding size %d is overriden by pretrained embedding size %d." 
                    % (params.embed_size, self.embed_size))
            embedding_weights = torch.from_numpy(vocab.embeddings)
        else:
            self.embed_size = params.embed_size
            embedding_weights = None
        self.initial_probs = get_initial_probs(self.vocab_size, vocab.SOS)
        self.initial_token_idx = 1
        self.max_dec_steps = params.max_tgt_len+1 if max_dec_steps is None else max_dec_steps
        self.pointer = params.pointer
        self.cover_loss = params.cover_loss
        self.cover_func = params.cover_func
        self.enc_hidden_size = params.hidden_size
        self.dec_hidden_size = params.hidden_size
        self.pos_encoding = PositionalEncoding(self.embed_size,  dropout=params.dropout)
        self.input_embed = nn.Linear(self.embed_size, params.hidden_size)
        self.output_embed = nn.Linear(self.embed_size, params.hidden_size)

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=vocab.PAD,
                                _weight=embedding_weights)
        # embed_size, hidden_size, n_heads, ff_size, dropout
        self.encoder = Encoder(params.hidden_size, n_heads=params.num_heads, ff_size=params.ff_size, dropout=params.dropout, n_layers=params.num_encoder_layers)
        self.decoder = Decoder(params.hidden_size, n_heads=params.num_heads, ff_size=params.ff_size, dropout=params.dropout, n_layers=params.num_decoder_layers)

        self.enc_bilinear = nn.Bilinear(params.hidden_size, params.hidden_size, 1)
        self.linear_vocab1 = nn.Linear(2*params.hidden_size, params.hidden_size, bias=True)
        self.linear_vocab2 = nn.Linear(params.hidden_size, self.vocab_size, bias=True)

        self.linear_gen = nn.Linear(2*params.hidden_size + self.embed_size, 1, bias=True)
        

    def filter_oov(self, tensor, ext_vocab_size):
        if ext_vocab_size and ext_vocab_size > self.vocab_size:
            result = tensor.clone()
            result[tensor >= self.vocab_size] = self.UNK
            return result
        return tensor

    def forward(self, input_tensor, target_tensor=None, input_lengths=None, criterion=None, *,
                forcing_ratio=0, partial_forcing=True, ext_vocab_size=None, sample=False,
                saved_out: Seq2SeqOutput=None, visualize: bool=None, include_cover_loss: bool=False) -> Seq2SeqOutput:
        input_length = input_tensor.size(0)
        batch_size = input_tensor.size(1)
        log_prob = not (sample or self.decoder.pointer)
        if(visualize is None):
            visualize = criterion is None
        if(visualize and not(self.enc_attn or self.pointer)):
            visualize = False

        if(target_tensor is None):
            target_length = self.max_dec_steps
        else:
            target_length = target_tensor.size(0)
        target_length = target_length+1å

        use_teacher_forcing = True
        # print("input len: ")
        # print(input_tensor.shape)
        # print("target len: ")
        # print(target_tensor.shape)
        inp_tensor = torch.ones(batch_size, 1)
        inp_tensor = input_tensor.transpose(0, 1)
        # out_tensor = target_tensor.transpose(0, 1)
        input_mask = (inp_tensor != self.pad).unsqueeze(-2)
        # print("Input mask")
        # print(input_mask.shape)
        output_mask = make_std_mask(out_tensor, self.pad)
        # output_mask = output_mask.transpose(1, 2).transpose(0, 1)
        # print("Output mask")
        # print(output_mask.shape)
        input_embedding = self.embedding(self.filter_oov(inp_tensor, ext_vocab_size))
        # print("inp_tensor size")
        # print(inp_tensor.shape)
        input_position = self.pos_encoding(input_embedding)
        input_embed = self.input_embed(input_position)
        # print(input_embedding.shape)
        encoder_output, e_attn = self.encoder(input_embed, input_mask)
        # print("E attn")
        # print(encoder_output.shape)
        # print(e_attn.shape)
        out_tensor = torch.cat((torch.Tensor(batch_size, 1).fill_(self.vocab.SOS), target_tensor.transpose(0, 1)), dim=1)
        output_embedding = self.embedding(self.filter_oov(out_tensor, ext_vocab_size))
        output_position = self.pos_encoding(output_embedding)
        output_embed = self.output_embed(output_position)
        decoder_output, p_attn = self.decoder(output_embed, encoder_output, input_mask, output_mask)
        # print("P attn")
        # print(decoder_output.shape)
        # print(p_attn.shape)
        at = torch.sum(p_attn, dim=1)
        at = torch.softmax(at, dim=2)
        context = torch.bmm(at, encoder_output)
        # print("Context")
        # print(context.shape)
        # print(decoder_output.shape)
        # print(encoder_output.shape)
        # print("Vocab size")
        # print(self.vocab_size)
        # print(ext_vocab_size)
        # vocab_vec and p_gen - (B, OU, V_S)
        dec_output = torch.cat((decoder_output, context), dim=2)
        # print(dec_output.shape)
        vocab1 = self.linear_vocab1(dec_output)
        # print(vocab1.shape)
        vocab_vec = self.linear_vocab2(vocab1)
        # print(vocab_vec.shape)
        p_gen = self.linear_gen(torch.cat((dec_output, output_embedding), dim=2))
        # p_gen = self.linear_gen(torch.cat((torch.cat(torch.cat((at, decoder_output), dim=2))), output_embedding), dim=2)
        gen_probs = p_gen*vocab_vec
        # print(gen_probs.shape)
        p_copy = 1.0-p_gen 
        # Output size - (B, OU, E_V_S, )
        output = torch.zeros(batch_size, target_tensor.shape[0], ext_vocab_size)
        output[:, :, :self.vocab_size] = gen_probs
        p_copy = p_copy.expand(batch_size, target_length, input_length)
        # print("P copy")
        # print(output.shape)
        # print(p_copy.shape)
        # print(at.shape)
        # copy_probs = 
        copy_probs = p_copy*at
        # print("Copy probs")
        # print(copy_probs.shape)
        # print(at.shape)
        # print(p_copy.shape)
        inp = input_tensor.transpose(0, 1).unsqueeze(1).expand(batch_size, target_length, input_length)
        # probs = copy_probs.unsqueeze(1).expand(batch_size, target_length, input_length)
        # print(inp.shape)
        # print(probs.shape)
        output.scatter_add_(2, inp, copy_probs)
        output1 = output[:, :target_length-1, :]
        # output = F.softmax(output, dim=2)

        # print("NLL Loss")
        # print(output.shape)
        # print(target_tensor.transpose(0, 1).shape)
        ce_loss = F.cross_entropy(output1.transpose(1, 2), target_tensor.transpose(0, 1), reduction='mean')
        # print(ce_loss.item())

        # enc_energy = self.enc_bilinear(decoder_output.transpose(1, 0), encoder_output.transpose(1, 0))
        # enc_attn = F.softmax(enc_energy, dim=0).transpose(0, 1).squeeze(2)
        # decoded_tokens = []

        r = Seq2SeqOutput(encoder_output, decoder_output,
                      target_tensor, ce_loss.item())

        r.decoded_tokens = target_tensor
        r.loss = ce_loss
        r.loss_value = ce_loss.item()
        r.enc_attn_weights = p_attn
        r.ptr_probs = output
        # decoder_input = torch.tensor([self.vocab.SOS] * batch_size, device=DEVICE)
        # for i in range(target_length):
        #     decoder_embedding = self.embedding(filter_oov(decoder_input, ext_vocab_size))
        #     res_vector = torch.cat((context.view(1, 0, 2).contiguous()[i], decoder_embedding.view(1, 0, 2).contiguous()), dim=2)
        #     res_vector = torch.cat((res_vector, decoder_output.view(1, 0, 2).contiguous()[i]), dim=2)
        #     p_ptr = torch.sigmoid(self.linear_gen(res_vector))
        #     final_vec = torch.linear_final(res_vector)
        #     logits = F,softmax(torch.linear_logits(final_vec), dim=1)
        #     p_gen = 1-p_ptr

        #     output = torch.zeros(batch_size, ext_vocab_size, device=DEVICE)
        #     output[:, :self.vocab_size] = p_gen * logits
        #     output.scatter_add_(1, input_tensor.transpose(0, 1), prob_ptr*enc_attn)

        #     if not sample:
        #         _, top_idx = output.data.topk(1)
        #     else:
        #         top_idx = torch.multinomial(output, 1)

        #     top_idx = top_idx.squeeze(1).detach()
        #     r.decoder_tokens[di] = top_idx

        #     if(criterion):
        #         if target_tensor is None:
        #             gold_standard = top_idx
        #         else:
        #             gold_standard = target_tensor[di]

        #         decoder_output = torch.log(output + eps)
        #         nll_loss = criterion(decoder_output, gold_standard)
        #         r.loss += nll_loss
        #         r.loss_value += nll_loss.item()

        #     if(target_tensor):
        #         decoder_input = target_tensor[di]
        #     else:
        #         decoder_input = top_idx

        return r

    def beam_search(self, input_tensor, input_lengths=None, ext_vocab_size=None, beam_size=4, *,
                    min_out_len=1, max_out_len=None, len_in_words=True) -> List[Hypothesis]:
        input_length = input_tensor.size(0)
        batch_size = input_tensor.size(1)
        assert batch_size == 1
        if max_out_len is None:
            max_out_len = self.max_dec_steps-1

        inp_tensor = input_tensor.transpose(0, 1)
        input_mask = (inp_tensor != self.pad).unsqueeze(-2)
        input_embedding = self.embedding(self.filter_oov(inp_tensor, ext_vocab_size))
        input_position = self.pos_encoding(input_embedding)
        input_embed = self.input_embed(input_position)
        encoder_output, e_attn = self.encoder(input_embed, input_mask)

        hypos = [Hypothesis([self.vocab.SOS], [], None, [], [], 1)]
        results, backup_results = [], []
        step=0

        out_tensor = torch.Tensor(batch_size, 1).fill_(self.vocab.SOS)
        hypos = [Hypothesis(out_tensor, [], None, [], [], 1)]

        while hypos and step < 2*max_out_len:
            n_hypos = len(hypos)
            # if(n_hypos < beam_size):
            #     hypos.extend(hypos[-1] for _ in range(beam_size-n_hypos))

            output_embedding = self.embedding(self.filter_oov(out_tensor, ext_vocab_size))
            output_position = self.pos_encoding(output_embedding)
            output_embed = self.output_embed(output_position)
            decoder_output, p_attn = self.decoder(output_embed, encoder_output, input_mask, output_mask)

            at = torch.sum(p_attn, dim=1)
            at = torch.softmax(at, dim=2)
            context = torch.bmm(at, encoder_output)

            dec_output = torch.cat((decoder_output, context), dim=2)

            vocab1 = self.linear_vocab1(dec_output)

            vocab_vec = self.linear_vocab2(vocab1)

            p_gen = self.linear_gen(torch.cat((dec_output, output_embedding), dim=2))
            gen_probs = p_gen*vocab_vec

            p_copy = 1.0-p_gen 

            output = torch.zeros(batch_size, target_tensor.shape[0], ext_vocab_size)
            output[:, :, :self.vocab_size] = gen_probs
            p_copy = p_copy.expand(batch_size, target_length, input_length)

            copy_probs = p_copy*at

            inp = input_tensor.transpose(0, 1).unsqueeze(1).expand(batch_size, target_length, input_length)

            output.scatter_add_(2, inp, copy_probs)
            output1 = output[:, :target_length-1, :]

            ce_loss = F.cross_entropy(output1.transpose(1, 2), target_tensor.transpose(0, 1), reduction='mean')

        for i in range(max_out_len):


    def beam_search(self, input_tensor, input_lengths=None, ext_vocab_size=None, beam_size=4, *, 
                    min_out_len=1, max_out_len=None, len_in_words=True) -> List[Hypothesis]:

        batch_size = input_tensor.size(1)
        assert batch_size == 1
        if max_out_len is None:
            max_out_len = self.max_dec_steps-1

        encoder_embedded = self.embedding(self.filter_oov(input_tensor, ext_vocab_size))

        input_mask = (input_mask != pad).unsqueeze(-2)
        output_mask = make_std_mask(self.target_tensor, pad)
        input_embedding = self.embedding(self.filter_oov(input_tensor, ext_vocab_size))
        encoder_output = self.encoder(input_embedding, input_mask)
        decoder_output, p_attn = self.decoder(target_tensor, encoder_output, input_mask, output_mask)
        at = torch.sum(p_attn, dim=1)
        at = torch.softmax(at, dim=2)
        context = torch.bmm(at, encoder_output)
        decoder_hidden = context

        encoder_outputs = encoder_output.expand(-1, beam_size, -1).contiguous()
        input_tensor = input_tensor.expand(-1, beam_size).contiguous()

        hypos = [Hypothesis([self.vocab.SOS], [], decoder_hidden, [], [], 1)]
        results, backup_results = [], []
        step=0
        while hypos and step < 2 * max_out_len:

            n_hypos = len(hypos)
            if(n_hypos < beam_size):
                hypos.extend(hypos[-1] for _ in range(beam_size - n_hypos))

            decoder_input = torch.tensor([h.tokens[-1] for h in hypos], device=DEVICE)
            decoder_hidden = torch.cat([h.dec_hidden for h in hypos], 1)
            decoder_states = torch.cat([torch.cat(h.dec_states, 0) for h in hypos], 1)

            top_v, top_i = decoder_output.data.topk(beam_size)

            new_hypos = []
            for in_idx in range(n_hypos):
                for out_idx in range(beam_size):
                    new_tok = top_i[in_idx][out_idx].item()
                    new_prob = top_vp[in_idx][out_idx].item()
                    if(len_in_words):
                        non_word = not self.vocab.is_word(new_tok)
                    else:
                        non_word = new_tok == self.vocab.EOS

                    new_hypo = hypos[in_idx].create_next(new_tok, new_prob,
                                                         decoder_output[0][in_idx].unsqueeze(0).unsqueeze(0),
                                                         True,
                                                         p_attn[in_idx].unsqueeze(0).unsqueeze(0)
                                                         if p_attn is not None else None, non_word)

                    new_hypos.append(new_hypo)

            new_hypos = sorted(new_hypos, key=lambda h: -h.avg_log_prob)
            hypos = []
            new_complete_results, new_incomplete_results = [], []
            for nh in new_hypos:
                length = len(nh)
                if nh.tokens[-1] == self.vocab.EOS:
                    if(len(new_complete_results) < beam_size and min_out_len <= length <= max_out_len):
                        new_complete_results.append(nh)
                elif len(hypos) < beam_size and length < max_out_len:
                    hypos.append(nh)
                elif length == max_out_len and len(new_incomplete_results) < beam_size:
                    new_incomplete_results.append(nh)
            if new_complete_results:
                results.extend(new_complete_results)
            elif new_incomplete_results:
                backup_results.extend(new_incomplete_results)
            step += 1
        if not results:
            results = backup_results
        return sorted(results, key=lambda h: -h.avg_log_prob)[:beam_size]




















