import torch

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text  # or whatever


ckpt = torch.load("./results/09-27/snapshots/best.pt")
#print(ckpt.keys())
sd = "model"  # I forgot what it was called.
#print("\n".join(ckpt[sd].keys()))
dict_ = {}

model.load_state_dict(snapshot['model'])

for k, v in ckpt[sd].items():
 
    k = remove_prefix(k,"module.")
    dict_[k] = v
ckpt[sd] = dict_
torch.save("output.pt", ckpt)
