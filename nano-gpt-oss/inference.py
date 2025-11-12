import torch
from torch.nn import functional as F
from architecture.tokenizer import get_tokenizer

context_len = 8192
tokenizer = get_tokenizer()


# 将文本转换为token id张量，并将编码结果转换为PyTorch张量返回
def text_to_token_ids(text, tokenizer):
    encoder = tokenizer.encode(text)
    encoder_tensor = torch.tensor(encoder)
    return encoder_tensor


# 将token id张量转换为文本，并返回结果
def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.tolist())


def generate_text(model, prompt, max_tokens=100, temperature=0.8, top_k=50):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # 将prompt转换为token id张量，并移动到指定设备
    idx = text_to_token_ids(prompt, tokenizer).to(device)
    for _ in range(max_tokens):
        idx_cond = idx[-context_len:]   # 只保留最近context_len(8192)个token作为输入
        # 利用模型预测下一个token的logits值，在推理模式下运行以提高效率
        with torch.inference_mode():
            logits = model(idx_cond)
        # 对最后一步输出的logits应用温度调节，控制输出分布的平滑程度
        logits = logits[-1, :] / temperature

        # 如果设置了top_k参数，则只保留最有可能的top-k项，其他设为负无穷大
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[[-1]]] = -float('Inf')

        # 归一化为概率分布
        probs = F.softmax(logits, dim=-1)
        # 多项式抽样选取下一个 token
        idx_next = torch.multinomial(probs, num_samples=1)
        # 更新索引序
        idx = torch.cat((idx, idx_next), dim=0)

    # 解码并返回
    result = token_ids_to_text(idx,tokenizer)
    return result
