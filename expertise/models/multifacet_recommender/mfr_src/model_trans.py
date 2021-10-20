import copy
import math
import torch
from torch import nn


def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`."""

    def __init__(
        self,
        hidden_size,
        max_position_embeddings,
        num_hidden_layers=2,
        num_attention_heads=10,
        # intermediate_size=3072,
        intermediate_factor=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        type_vocab_size=3,
        initializer_range=0.02,
    ):
        """Constructs BertConfig.
        Args:
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            intermediate_factor: intermediate_size = hidden_size * intermediate_factor:
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        # self.intermediate_size = intermediate_size
        self.intermediate_size = intermediate_factor * hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range


# try:
#    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
# except ImportError:
#    logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config, add_position_emb):
        super(BertEmbeddings, self).__init__()
        # self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        if add_position_emb:
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings, config.hidden_size
            )
        self.add_position_emb = add_position_emb
        if config.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(
                config.type_vocab_size, config.hidden_size
            )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, words_embeddings, token_type_ids=None):
        if self.add_position_emb:
            seq_length = words_embeddings.size(1)
            bsz = words_embeddings.size(0)
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=words_embeddings.device
            )
            position_ids = position_ids.expand(bsz, seq_length)
            # position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            # if token_type_ids is None:
            #    token_type_ids = torch.zeros_like(input_ids)

            # words_embeddings = self.word_embeddings(input_ids)
            position_embeddings = self.position_embeddings(position_ids)

            # embeddings = words_embeddings + position_embeddings + token_type_embeddings
            embeddings = words_embeddings + position_embeddings
        else:
            embeddings = words_embeddings

        if token_type_ids is not None:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings += token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        # embeddings = self.dropout(embeddings)
        return embeddings


class MultiHeadedAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadedAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    # def forward(self, hidden_states, attention_mask):
    def forward(self, query_in, key_in, value_in, attention_mask):
        mixed_query_layer = self.query(query_in)
        mixed_key_layer = self.key(key_in)
        mixed_value_layer = self.value(value_in)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = MultiHeadedAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, memory_tensors, attention_mask):
        # self_output = self.self(input_tensor, attention_mask)
        if memory_tensors is None:
            self_output = self.self(
                input_tensor, input_tensor, input_tensor, attention_mask
            )
        else:
            self_output = self.self(
                input_tensor, memory_tensors, memory_tensors, attention_mask
            )
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, None, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class Decode_Layer(nn.Module):
    def __init__(self, config):
        super(Decode_Layer, self).__init__()
        self.self_attention = BertAttention(config)
        self.memory_attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, memory_tensors, attention_mask):
        self_attention_output = self.self_attention(hidden_states, None, attention_mask)
        memory_attention_output = self.memory_attention(
            self_attention_output, memory_tensors, attention_mask
        )
        intermediate_output = self.intermediate(memory_attention_output)
        layer_output = self.output(intermediate_output, memory_attention_output)
        return layer_output


# class BertEncoder(nn.Module):
class Transformer(nn.Module):
    def __init__(
        self,
        model_type,
        hidden_size,
        max_position_embeddings,
        num_hidden_layers=2,
        add_position_emb=False,
        num_type_feature=0,
        decoder=False,
        dropout_prob=0.1,
        num_attention_heads=10,
    ):
        super(Transformer, self).__init__()

        config = BertConfig(
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            num_hidden_layers=num_hidden_layers,
            hidden_dropout_prob=dropout_prob,
            attention_probs_dropout_prob=dropout_prob,
            type_vocab_size=num_type_feature,
            num_attention_heads=num_attention_heads,
        )
        self.emb_layer = BertEmbeddings(config, add_position_emb)
        if decoder:
            layer = Decode_Layer(config)
            print("Using Tranformer decoder")
        else:
            layer = BertLayer(config)
        self.decoder = decoder
        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(config.num_hidden_layers)]
        )
        self.model_type = model_type
        # self.add_position_emb = add_position_emb

    def forward(
        self,
        hidden_states,
        token_type_ids=None,
        memory_tensors=None,
        attention_mask=None,
        output_all_encoded_layers=False,
    ):
        hidden_states = self.emb_layer(hidden_states, token_type_ids)
        # hidden_states should have dimension [n_batch, n_seq_len, emb_size]
        # attention_mask should have the same dimension and 0 means not masking while -10000.0 means masking
        all_encoder_layers = []
        for layer_module in self.layer:
            if self.decoder:
                hidden_states = layer_module(
                    hidden_states, memory_tensors, attention_mask
                )
            else:
                hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
