# -*- coding: utf-8 -*-

#CLASSES PULLED IN BY TRANSFORMER.PY

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.file_utils import add_start_docstrings

from transformers import(
    LongformerTokenizer,
    LongformerModel,
    LongformerConfig,
    )

#PARAMETERIZE
longformer_model = LongformerModel.from_pretrained(
    'simonlevine/Bio_ClinicalBERT-2048', gradient_checkpointing=True)
longformer_config = LongformerConfig.from_pretrained(
    'simonlevine/Bio_ClinicalBERT-2048', gradient_checkpointing=True)
#PARAMETERIZE


#ensure these are actuallly used below...

from transformers.modeling_utils import SequenceSummary

from transformers.modeling_bert import BertPreTrainedModel #yes, bert.
from transformers.modeling_longformer import (
    LONGFORMER_START_DOCSTRING,
    LONGFORMER_INPUTS_DOCSTRING,
    LONGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST,
)

from transformers.modeling_longformer import LongformerModel, LongformerClassificationHead

def repack_output(output_ids, output_mask, num_labels):
    batch_size = output_ids.size(0)
    idx_arr = torch.nonzero(output_mask)
    rows = idx_arr[:, 0]
    cols = output_ids[idx_arr[:, 0], idx_arr[:, 1]]
    c_true = torch.zeros((batch_size, num_labels), dtype=torch.float, device=output_ids.device)
    c_true[rows, cols] = 1.0
    return c_true



@add_start_docstrings(
    """Longformer Model transformer with a sequence classification head on top (a linear layer
  on top of the pooled output) e.g. for eXtreme Multi-label Classification (XMLC). """,
    LONGFORMER_START_DOCSTRING,
    LONGFORMER_INPUTS_DOCSTRING,
)
class LongformerForXMLC(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels]``.
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
        tokenizer = LongformerTokenizer.from_pretrained('.....')
        model = LongformerForSequenceClassification.from_pretrained('.....')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
  """
    config_class = LongformerConfig
    config_class.gradient_checkpointing = True #essential for memory usage!!!
    pretrained_model_archive_map = LONGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST
    base_model_prefix = "longformer"

    def __init__(self, config):
        super(LongformerForXMLC, self).__init__(config)
        self.num_labels = config.num_labels
        self.longformer = LongformerModel(config)
        self.classifier = LongformerClassificationHead(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        # head_mask=None, #N/A for longformers ?
        inputs_embeds=None,
    ):
        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            # head_mask=head_mask, #N/A for roberta/ longformers as implemented
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        # add hidden states and attention if they are here
        outputs = (logits,) + outputs[2:]
        return outputs  # logits, (hidden_states), (attentions)
