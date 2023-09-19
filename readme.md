---
base_model: zoukagh/bert-base-iz-spain-uncased
model-index:
- name: bert-base-iz-spain-uncased
  results: []
language:
- es
pipeline_tag: fill-mask
---
  
<!-- This model card has been generated automatically according to the information Keras had access to. You should
probably proofread and complete it, then remove this comment. -->

# bert-base-iz-spain-uncased
Los pesos del modelo se pueden descargar desde  [zoukagh/bert-base-iz-spain-uncased](https://huggingface.co/zoukagh/bert-base-iz-spain-uncased)
Este modelo se ha entrenado con gran corpus de texto de wikipedia , TED, DGT , 

- Steps: 2M 👷🏗️🚧

### Framework versions
- TensorFlow 2.12.0
# BERT base model (uncased)

Pretrained model on spanish language using a masked language modeling (MLM) objective. It was introduced in
[this paper](https://arxiv.org/abs/1810.04805) 

## Model description

BERT is a transformers model pretrained on a large corpus of English data in a self-supervised fashion. This means it
was pretrained on the raw texts only, with no humans labeling them in any way (which is why it can use lots of
publicly available data) with an automatic process to generate inputs and labels from those texts. More precisely, it
was pretrained with two objectives:

- Masked language modeling (MLM): taking a sentence, the model randomly masks 15% of the words in the input then run
  the entire masked sentence through the model and has to predict the masked words. This is different from traditional
  recurrent neural networks (RNNs) that usually see the words one after the other, or from autoregressive models like
  GPT which internally masks the future tokens. It allows the model to learn a bidirectional representation of the
  sentence.
- Next sentence prediction (NSP): the models concatenates two masked sentences as inputs during pretraining. Sometimes
  they correspond to sentences that were next to each other in the original text, sometimes not. The model then has to
  predict if the two sentences were following each other or not.

This way, the model learns an inner representation of the English language that can then be used to extract features
useful for downstream tasks: if you have a dataset of labeled sentences, for instance, you can train a standard
classifier using the features produced by the BERT model as inputs.


## Intended uses & limitations

You can use the raw model for either masked language modeling or next sentence prediction, but it's mostly intended to
be fine-tuned on a downstream task. See the [model hub](https://huggingface.co/models?filter=bert) to look for
fine-tuned versions of a task that interests you.

Note that this model is primarily aimed at being fine-tuned on tasks that use the whole sentence (potentially masked)
to make decisions, such as sequence classification, token classification or question answering. For tasks such as text
generation you should look at model like GPT2.

### How to use

You can use this model directly with a pipeline for masked language modeling:

```python
>>> from transformers import pipeline
>>> unmasker = pipeline('fill-mask', model='zoukagh/bert-base-iz-spain-uncased')
>>> unmasker("protocolo de mensajeria mas usado es el [MASK].")

[{'sequence': "[CLS] protocolo de mensajeria mas usado es el sms [SEP]",
  'score': 0.1073106899857521,
  'token': 4827,
  'token_str': 'sms'},
 {'sequence': "[CLS] protocolo de mensajeria mas usado es el email [SEP]",
  'score': 0.08774490654468536,
  'token': 2535,
  'token_str': 'email'}]
```


 TensorFlow:

```python
from transformers import BertTokenizer, TFBertModel
tokenizer = BertTokenizer.from_pretrained('zoukagh/bert-base-iz-spain-uncased')
model = TFBertModel.from_pretrained("zoukagh/bert-base-iz-spain-uncased")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='tf')
output = model(encoded_input)
```

This bias will also affect all fine-tuned versions of this model.

## Training data

The BERT model was pretrained on CC100 and Spanish Wikipedia dump (excluding lists, tables and
headers).

## Training procedure

### Preprocessing

The texts are lowercased and tokenized using WordPiece and a vocabulary size of 30,000. The inputs of the model are
then of the form:

```
[CLS] Sentence A [SEP] Sentence B [SEP]
```

With probability 0.5, sentence A and sentence B correspond to two consecutive sentences in the original corpus, and in
the other cases, it's another random sentence in the corpus. Note that what is considered a sentence here is a
consecutive span of text usually longer than a single sentence. The only constrain is that the result with the two
"sentences" has a combined length of less than 512 tokens.

The details of the masking procedure for each sentence are the following:
- 15% of the tokens are masked.
- In 80% of the cases, the masked tokens are replaced by `[MASK]`.
- In 10% of the cases, the masked tokens are replaced by a random token (different) from the one they replace.
- In the 10% remaining cases, the masked tokens are left as is.

### Pretraining

The model was trained on 4 cloud TPUs in Pod configuration (16 TPU chips total) for one million steps with a batch size
of 256. The sequence length was limited to 128 tokens for 90% of the steps and 512 for the remaining 10%. The optimizer
used is Adam with a learning rate of 1e-4, \\(\beta_{1} = 0.9\\) and \\(\beta_{2} = 0.999\\), a weight decay of 0.01,
learning rate warmup for 10,000 steps and linear decay of the learning rate after.


