import torch
d = torch.load('./corrections.pt', map_location='cpu', weights_only=False)
for i, e in enumerate(d['entries']):
    word = e.get('word')
    wd = e.get('word_delta')
    wd_norm = wd.norm().item() if wd is not None else None
    wd_shape = wd.shape if wd is not None else None
    print(f"Entry {i}: word={word}, shape={wd_shape}, norm={wd_norm}")
