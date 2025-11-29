import struct


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
