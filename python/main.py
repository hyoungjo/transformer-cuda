import os
import struct

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def load_texts(file_path):
    with open(file_path, "r") as f:
        texts = f.read().splitlines()
    return texts


def save_to_binary(file_path, tensor_dict):
    """
    Given a dictionary of tensors, converts to binary and saves as
    C-style struct objects.

    Args:
        file_path (str): The path to save the binary file.
        tensor_dict (dict): The dictionary of tensors to save.
            The tensors are saved in the following format.
            [int: len(name)][char*: name][int: len(dims)][int*: dims][float*: data]
    """

    print(f"[PYTHON][INFO] Saving to {file_path}")

    with open(file_path, "wb") as f:
        f.write(struct.pack("i", len(tensor_dict)))  # "i" is int

        for name, tensor in tensor_dict.items():
            name = name.encode("utf-8")  # binary
            data = tensor.detach().cpu().float().numpy()  # float32

            f.write(struct.pack("i", len(name)))
            f.write(name)

            dims = data.shape
            f.write(struct.pack("i", len(dims)))
            for dim in dims:
                f.write(struct.pack("i", dim))

            f.write(data.tobytes())


def inference(device, model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    print(f"[PYTHON][TRACE] {text = }")
    print(f"[PYTHON][TRACE] {inputs = }")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        last_token_logits = logits[0, -1, :]
        probs = torch.softmax(last_token_logits, dim=0)

    top_prob, top_idx = torch.topk(probs, 1)
    predicted_token = tokenizer.decode(top_idx)
    print(f"[PYTHON][TRACE] {predicted_token = }")

    return inputs["input_ids"], probs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[PYTHON][INFO] Loading tokenizer and model")

    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model.eval()

    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    print("[PYTHON][INFO] Saving model weights")

    save_to_binary(f"{data_dir}/weights.bin", model.state_dict())

    print("[PYTHON][INFO] Loading texts")

    texts = load_texts(f"{data_dir}/texts.txt")

    print("[PYTHON][INFO] Running inference")

    for idx, text in enumerate(texts):
        input_ids, probs = inference(device, model, tokenizer, text)

        os.makedirs(f"{data_dir}/{idx}", exist_ok=True)
        save_to_binary(f"{data_dir}/{idx}/input_ids.bin", {"input_ids": input_ids})
        save_to_binary(f"{data_dir}/{idx}/probs.bin", {"probs": probs})


if __name__ == "__main__":
    main()
