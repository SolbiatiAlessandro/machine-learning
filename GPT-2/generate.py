

import torch
import torch.nn.functional as F

    
def generate(model, tokenizer, config, prompt="Hello, I'm a language model", device='cpu', length=20):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    with torch.no_grad():
        tokens = tokenizer.encode(prompt)
        #tokens = [220] * (target_length - len(tokens)) + tokens
        x = torch.tensor(tokens[:config.block_size], device=device).view(1, -1)
        to_decode = x.tolist()[0]
        for _ in range(length):

            x = x[-config.block_size:]

            logits = model(x.view(1, -1))

            v, ixs = logits[0, -1, :].topk(20)
            ix = torch.multinomial(F.softmax(v, dim=0), 1)
            new_token = ixs[ix]
            
            new_token_value = new_token.view(-1).item()
            if new_token_value >= tokenizer.n_vocab:
                new_token_value = new_token_value % tokenizer.n_vocab
                
            to_decode.append(new_token.view(-1).item())
            x = torch.cat([x.view(-1), new_token])

        output = tokenizer.decode(to_decode)
        
    print(f"[generate] generated: {output}")
    return(output)



def sample_generations(model, tokenizer, config, device, wandb_obj=None, iteration=None):
    prompts = [
        "When he went",
        "After the other",
        "Hello world",
        "Often it was",
        "2+2=",
        "The capital of France is ",
        "The pen is on the ",
    ]
    outputs = [
        generate(model, tokenizer, config, device=device, prompt=prompt, length=30)
        for prompt in prompts
    ]

    if wandb_obj:
        gen_table = wandb_obj.Table(columns=["Prompt", "Generation", "Iteration"])
        for i, prompt in enumerate(prompts):
            gen_table.add_data(prompt, outputs[i], iteration)
        wandb_obj.log({"Generations": gen_table})


    