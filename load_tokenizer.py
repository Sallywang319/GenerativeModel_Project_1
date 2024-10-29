# Example code to load the Tokenizer.
# You can also create your only tokenizer with vocab.txt.
from transformers import BertTokenizer
tokenizer = BertTokenizer(vocab_file='vocab.txt')
tokenizer.add_special_tokens(
    {'pad_token': '[PAD]',
     'cls_token': '[BOS]',
     'sep_token': '[EOS]',
     'mask_token': '[MASK]',
     "unk_token": "[UNK]",
     "bos_token": '[BOS]',
     "eos_token": '[EOS]'})

# Example Usage
# text = "生成模型基础！"
# print(tokenizer(text))
# output:
# {'input_ids': [2, 619, 3323, 1250, 2845, 3267, 1, 2627, 3],
# 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}
# REMARK: 'token_type_ids' is useless for auto-regressive models like GPT2, please remove it if you are using GPT2.
