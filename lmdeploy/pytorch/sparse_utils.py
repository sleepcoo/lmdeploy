# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.pytorch.layers.sparse_mlp import LlamaSparseMLP


def replace_mlp(model: torch.nn.Module, model_path: str, torch_dtype):
    for i, layer in enumerate(model.model.layers):
        llama_mlp = LlamaSparseMLP(layer.mlp.config, model_path, torch_dtype,
                                   i, layer.mlp)
        layer.mlp = llama_mlp
    return model


if __name__ == '__main__':
    import time

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_path = 'Llama'
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    input_text = 'Hello, my dog is cute'

    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    start = time.time()
    output = hf_model.generate(input_ids, max_length=50, temperature=0.7)
    end = time.time()
    print('ori model time ', (end - start) * 1e6)
    hf_model = replace_mlp(hf_model, model_path, torch.float16)
    hf_model.to(device)
    print(hf_model)
    output = hf_model.generate(input_ids, max_length=50, temperature=0.7)
    start = time.time()
    output = hf_model.generate(input_ids, max_length=50, temperature=0.7)
    end = time.time()
    print('model time ', (end - start) * 1e6)
