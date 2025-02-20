from dataclasses import dataclass
import torch.nn as nn
import torch
import torch.nn.functional as F
import math

from dataclasses import dataclass
import torch.nn as nn
import torch
import torch.nn.functional as F
import math

from GPT import GPT, GPTConfig

# from generate import generate

def test_model_sanity():
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    config = GPTConfig
    gpt = GPT.from_pretrained('gpt2')

    assert gpt, "model failed to load"
    print("TEST PASSED: model loading")

    B,T,V = config.batch_size, config.block_size, config.vocab_size

    x = torch.randint(0, V, (B, T))
    x = gpt(x)
    assert torch.tensor(x.shape).tolist() == [B,T,V], "wrong prediction shape"
    print("TEST PASSED: model prediction with T = config.block_Size")

    x = torch.randint(0, V, (B, T-1))
    x = gpt(x)
    assert torch.tensor(x.shape).tolist() == [B,T-1,V], "wrong prediction shape"
    print("TEST PASSED: model prediction with T != config.block_Size")

from generate import generate

def test_generate():
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    config = GPTConfig
    gpt = GPT.from_pretrained('gpt2')
    gpt.to(device)
    gpt.eval()
    result = generate(gpt)
    assert len(result) > len("Hello, I'm a language model")
    print("TEST PASSED: generation")

    result2 = generate(gpt)
    assert result == result2
    print("TEST PASSED: generation deterministic")
    
    gpt = GPT(GPTConfig())
    gpt.to(device)
    gpt.eval()
    result3 = generate(gpt)
    assert result3
    print("TEST PASSED: generation from non initialized model")
    
def test_model_implementation():
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    
    from transformers import GPT2LMHeadModel
    hfmodel = GPT2LMHeadModel.from_pretrained('gpt2')
    hfmodel.to(device)
    hfmodel.eval()
    hfmodel_logits = lambda x: hfmodel(x).logits
    
    hf_result = generate(hfmodel_logits)
    assert hf_result, "hugginface model broken"
    
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    config = GPTConfig
    localmodel = GPT.from_pretrained('gpt2')
    localmodel.to(device)
    localmodel.eval()
    
    local_result = generate(localmodel)
    assert local_result, "local model not generating results"
    assert local_result == hf_result, "models generating different output"
    print("TEST PASSED: local model generate same inference as hugging face model")
    
    
if __name__ == "__main__":
    #test_model_sanity()
    test_generate()
    #test_model_implementation()

