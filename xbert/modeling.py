# -*- coding: utf-8 -*-

#CLASSES PULLED IN BY TRANSFORMER.PY

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.file_utils import add_start_docstrings

#NEED TO UPDATE #NEED TO UPDATE #NEED TO UPDATE

from transformers import(
    AutoTokenizer,
    AutoModel,
    AutoConfig,)

bioclinical_bert_Config = AutoConfig.from_pretrained(
    "emilyalsentzer/Bio_ClinicalBERT")
bioclinical_bert_Model = AutoModel.from_pretrained(
    "emilyalsentzer/Bio_ClinicalBERT")


# from transformers.configuration_bert import BertConfig
# from transformers.configuration_roberta import RobertaConfig
# from transformers.configuration_xlnet import XLNetConfig


from transformers.modeling_utils import SequenceSummary


from transformers.modeling_bert import (
    BERT_START_DOCSTRING,
    BERT_INPUTS_DOCSTRING,
    BERT_PRETRAINED_MODEL_ARCHIVE_MAP,
)
from transformers.modeling_bert import BertModel, BertPreTrainedModel

#should work with clinicalBERT (???), since it's a fine-tuned BERT model.



def repack_output(output_ids, output_mask, num_labels):
    batch_size = output_ids.size(0)
    idx_arr = torch.nonzero(output_mask)
    rows = idx_arr[:, 0]
    cols = output_ids[idx_arr[:, 0], idx_arr[:, 1]]
    c_true = torch.zeros((batch_size, num_labels), dtype=torch.float, device=output_ids.device)
    c_true[rows, cols] = 1.0
    return c_true



@add_start_docstrings(
    """Bert Model transformer with a sequence classification  head on top (a linear layer on top of
    the pooled output) e.g. for eXtreme Multi-label Classification (XMLC). """,
    BERT_START_DOCSTRING,
    BERT_INPUTS_DOCSTRING,
)

class BertForXMLC(BertPreTrainedModel):  # NEED TO UPDATE
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForXMLC.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
  """

    def __init__(self, config):
        super(BertForXMLC, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        # get [cls] hidden states
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        return outputs  # logits, (hidden_states), (attentions)

#!!! removed roberta and xlnet classes !!!

