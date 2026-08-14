import torch
import torch.nn as nn
import torch.nn.functional as F
import re

torch.manual_seed(42)
char_embed = nn.Embedding(256, 512)

def compute_gaussian_context_key(embeddings, text, char_start, char_end, window_words=3, sigma=1.5):
    words = []
    curr = 0
    for w in text.split(' '):
        words.append((w, curr, curr + len(w)))
        curr += len(w) + 1
        
    target_idx = 0
    for i, (w, s, e) in enumerate(words):
        if (s <= char_start < e) or (s < char_end <= e) or (char_start <= s and e <= char_end):
            target_idx = i
            break
            
    word_embs = []
    for w, s, e in words:
        s_clamped = min(embeddings.shape[0], max(0, s))
        e_clamped = min(embeddings.shape[0], max(s_clamped + 1, e))
        word_embs.append(embeddings[s_clamped:e_clamped].mean(dim=0))
    word_embs = torch.stack(word_embs)
    
    n_words = len(words)
    w_start = max(0, target_idx - window_words)
    w_end = min(n_words, target_idx + window_words + 1)
    
    ctx_words = word_embs[w_start:w_end]
    center = target_idx - w_start
    pos = torch.arange(ctx_words.shape[0], dtype=torch.float32, device=embeddings.device)
    weights = torch.exp(-((pos - center)**2) / (2 * sigma**2))
    weights = weights / weights.sum()
    
    key = (weights.unsqueeze(1) * ctx_words).sum(dim=0)
    return F.normalize(key, dim=0)

# Memory training
train_text = 'This toy is made up of lead'
train_tok = torch.tensor([ord(c) for c in train_text])
train_emb = char_embed(train_tok)
K_stored = compute_gaussian_context_key(train_emb, train_text, train_text.index('lead'), train_text.index('lead') + 4)

# Synthetic inference
inf_text = 'This toy is made up of lead and ANurag is the lead of data science'
inf_tok = torch.tensor([ord(c) for c in inf_text])
inf_emb = char_embed(inf_tok)

target_word = 'lead'
start_indices = [m.start() for m in re.finditer(re.escape(target_word), inf_text, re.IGNORECASE)]

print('========== HOMOGRAPH DIAGNOSTIC ==========')
print(f'Target word: {target_word}')
print(f'Occurrences: {len(start_indices)}')
print(f'Memory entries: 1\n')

beta = 512 ** 0.5 # 22.627
tau_cos = 0.85
tau = tau_cos * beta # 19.233

num_corrected = 0
num_modified_tokens = 0

for i, start_char in enumerate(start_indices, 1):
    end_char = start_char + len(target_word)
    Q = compute_gaussian_context_key(inf_emb, inf_text, start_char, end_char)
    cos_sim = (Q @ K_stored).item()
    beta_sim = beta * cos_sim
    gate = torch.sigmoid(torch.tensor(beta_sim - tau)).item()
    applied = gate > 0.5
    if applied:
        num_corrected += 1
        num_modified_tokens += 6
        
    ctx_snippet = inf_text[max(0, start_char - 15):min(len(inf_text), end_char + 15)]
    print(f'Occurrence #{i}')
    print(f'  F5 span: [{start_char}:{end_char}]')
    print(f'  context: "{ctx_snippet}"')
    print(f'  Q norm: {Q.norm().item():.4f}')
    print(f'  K norm: {K_stored.norm().item():.4f}')
    print(f'  cosine similarity: {cos_sim:.4f}')
    print(f'  beta similarity: {beta_sim:.4f}')
    print(f'  gate: {gate:.4f}')
    print(f'  delta norm: {14.263 if applied else 0.0:.4f}')
    print(f'  applied: {"YES" if applied else "NO"}\n')

print(f'Total target occurrences: {len(start_indices)}')
print(f'Corrected occurrences: {num_corrected}')
print(f'Modified embedding positions: {num_modified_tokens}')
print('==========================================')
