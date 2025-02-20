
    
import tiktoken
import torch
import torch.nn.functional as F

enc = tiktoken.get_encoding('gpt2')


torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
    
def generate(model, prompt="Hello, I'm a language model", length=20):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    with torch.no_grad():
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(prompt)
        #tokens = [220] * (target_length - len(tokens)) + tokens
        x = torch.tensor(tokens).to('cuda')
        x = x.view(1, len(tokens))
        

        for i in range(length):
            if i % 10 == 0: print(f"[generate] token: {i}")
            y = model(x)
            probs = F.softmax(y[0, -1, :], dim=0)
            topk_probs, topk_ixs = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs.cpu(), 1).to(topk_probs.device)
            y= topk_ixs[ix]
            prompt += enc.decode([y[0].item()])
            x = torch.cat([x, y.view(1, 1)], dim=1)[:, 1:]
            #print(prompt)
        
    print(f"[generate] generated: {prompt}")
    return(prompt)
    