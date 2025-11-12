import gc
from torch.utils.data import Dataset,DataLoader
from architecture import get_tokenizer
from datasets import load_dataset
from tqdm.notebook import tqdm

batch_size = 5
context_len = 4000    # 输入文本的最大长度

dataset = load_dataset("roneneldan/TinyStories")       # 从huggingface加载数据集
train_text = "".join([ex['text'] for ex in dataset['train']])   # 将训练集的文本数据拼接成字符串
val_text = " ".join([ex["text"] for ex in dataset['validation']])

tokenizer = get_tokenizer()     # 获取tokenizer
print('tokenizing...')
train_tokens = tokenizer.encode(train_text)     # 对训练集的文本进行tokenize
val_tokens = tokenizer.encode(val_text)
print("tokenized")

class TextDataset(Dataset):
    def __init__(self, tokens, max_length=8192, stride=8192):
        self.input_ids = []
        self.target_ids = []
        # 取出每一个故事对应的编码，并构建输入和目标
        for i in tqdm(range(0, len(tokens) - max_length, stride)):
            input_chunk = tokens[i: i + max_length]
            target_chunk = tokens[i + 1: i + max_length + 1]
            self.input_ids.append(input_chunk)
            self.target_ids.append(target_chunk)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# 构建Dataset
train_dataset = TextDataset(train_tokens, max_length=context_len, stride=context_len)
val_dataset = TextDataset(val_tokens, max_length=context_len, stride=context_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

# 释放内存资源
# 删除原始数据集、训练集和验证集文本拼接后的字符串
del dataset, train_text, val_text
# 强制执行垃圾回收，清理那些已经没有引用但还没有被自动回收的内存对象
gc.collect()
