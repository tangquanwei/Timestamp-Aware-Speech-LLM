from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tokenizers import AddedToken

if __name__ == "__main__":
    # in_path
    in_path = (
        "/aistor/aispeech/hpc_stor01/home/tangquanwei00sx/proj/model/Qwen2-7B-Instruct"
    )
    # out_path
    out_path = in_path + "-TS"
    relative = True

    tokenizer = AutoTokenizer.from_pretrained(in_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        in_path,
        device_map="cpu",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    tokens = [
        "<|sil|>",
    ]
    if relative:
        for i in range(500):
            tokens.append(f"<|{i / 100:.2f}|>")
    else:
        for i in range(3001):
            tokens.append(f"<|{i / 100:.2f}|>")
    

    specials = []
    for i in tokens:
        specials.append(
            AddedToken(
                i, lstrip=False, rstrip=False, single_word=False, normalized=False
            ),
        )

    num_added = tokenizer.add_special_tokens({"additional_special_tokens": specials})
    print("special added:", num_added)

    model.resize_token_embeddings(len(tokenizer))

    tokenizer.save_pretrained(out_path)
    print("model save", out_path)
