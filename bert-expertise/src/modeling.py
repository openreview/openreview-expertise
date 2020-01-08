# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT models for linking. """

import torch
from torch import nn
import torch.nn.functional as F

from transformers import BertModel, BertPreTrainedModel

from IPython import embed


class BertForSequenceEmbedding(BertPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``evaluate`` is False) ``torch.FloatTensor`` of shape ``(1,)``:
            max-margin loss.
        **embedding**: ``torch.FloatTensor`` of shape ``(batch_size, config.embedding_size)``
            embedding for input sequence
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceEmbedding.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
    """

    def __init__(self, config):
        super(BertForSequenceEmbedding, self).__init__(config)
        self.seq_embed_size = config.seq_embed_size

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.embedding_layer = nn.Linear(config.hidden_size, self.seq_embed_size)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        evaluate=False,
        sample_size=None,
        margin=None
    ):

        assert ((evaluate and sample_size == None)
                or (not evaluate and sample_size != None))

        batch_size, num_seq, seq_len = input_ids.shape

        if not evaluate:
            assert num_seq == 2*sample_size + 1

        input_ids = input_ids.reshape(-1, seq_len)
        attention_mask = attention_mask.reshape(-1, seq_len)
        token_type_ids = token_type_ids.reshape(-1, seq_len)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        seq_embedding = self.embedding_layer(pooled_output)
        seq_embedding = seq_embedding.reshape(batch_size, num_seq, -1)

        seq_embedding = F.normalize(seq_embedding, p=2, dim=-1)

        outputs = (seq_embedding,) + outputs[2:]  # add hidden states and attention if they are here

        if not evaluate:
            mention_embedding = seq_embedding[:, 0:1, :]
            pos_embedding = torch.transpose(seq_embedding[:, 1:sample_size+1, :], 1, 2)
            neg_embedding = torch.transpose(seq_embedding[:, sample_size+1:, :], 1, 2)

            pos_scores = torch.bmm(mention_embedding, pos_embedding).squeeze(1)
            neg_scores = torch.bmm(mention_embedding, neg_embedding).squeeze(1)

            loss = F.relu(torch.max(neg_scores, dim=-1).values
                          - torch.max(pos_scores, dim=-1).values
                          + margin)
            loss = torch.mean(loss)

            outputs = (loss,) + outputs

        return outputs  # (loss), seq_embedding, (hidden_states), (attentions)
