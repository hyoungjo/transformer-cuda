import torch
import torch.utils.benchmark as benchmark
from helper import load_texts
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def inference_isolated(model, inputs):
    """
    The strictly isolated function to benchmark.
    """
    with torch.no_grad():
        output = model(**inputs)
    return output


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[PYTHON][INFO] Running on Device: {device}")

    print("[PYTHON][INFO] Loading tokenizer and model")

    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model.eval()

    print("[PYTHON][INFO] Loading texts")

    data_dir = "data"
    texts = load_texts(f"{data_dir}/texts.txt")

    print("[PYTHON][INFO] Running torch.utils.benchmark...")

    for idx, text in enumerate(texts):
        inputs = tokenizer(text, return_tensors="pt").to(device)
        _, seq_len = inputs["input_ids"].size()

        # runs a warm up period and sets block size
        timer = benchmark.Timer(
            stmt="inference_isolated(model, inputs)",
            globals={
                "inference_isolated": inference_isolated,
                "model": model,
                "inputs": inputs,
            },
            num_threads=1,
            label="Inference",
            sub_label=f"{seq_len = }",
            description=f"Text {idx}",
        )

        # blocked_autorange automatically decides the number of loops to run
        # to get a statistically significant result (in seconds)
        measurement = timer.blocked_autorange(min_run_time=0.5)
        print(measurement)


if __name__ == "__main__":
    main()
