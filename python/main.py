import argparse
import os

import torch
from helper import load_texts, save_to_binary
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt2")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[PYTHON][INFO] Loading tokenizer and model")

    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    data_dir = f"data/{model_name}"
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
