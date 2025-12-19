#include "gpt2.hpp"
#include "operations.hpp"
#include "tensor.hpp"
#include "utils.hpp"
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

int main() {
  std::cout << "[CUDA][INFO] Loading model weights..." << std::endl;

  GPT2 model("data/weights.bin");

  int total = 15;
  int passed = 0;

  std::cout << "[CUDA][INFO] Starting inference loop [0, " << total << ")"
            << std::endl;

  for (int i = 0; i < total; ++i) {
    std::string data_path = "data/" + std::to_string(i);

    auto input_data = utils::load_data(data_path + "/input_ids.bin", "cpu");
    std::vector<int> input_ids;
    for (float id : input_data["input_ids"]) {
      input_ids.push_back(static_cast<int>(id));
    }

    int seq_len = input_ids.size();
    std::cout << "[CUDA][TRACE] Input size: " << seq_len << " tokens"
              << std::endl;

    int *d_input_ids;
    cudaMalloc(&d_input_ids, seq_len * sizeof(int));
    cudaMemcpy(d_input_ids, input_ids.data(), seq_len * sizeof(int),
               cudaMemcpyHostToDevice);
    Tensor logits = model.forward(d_input_ids, seq_len);
    cudaFree(d_input_ids);

    Tensor probs = std::move(logits);
    operations::softmax(probs);
    probs.to("cpu");

    int64_t token_id = 0;
    float max_prob = -1e9;
    for (int64_t v = 0; v < probs.numel(); ++v) {
      if (probs(v) > max_prob) {
        max_prob = probs(v);
        token_id = v;
      }
    }

    std::cout << "[CUDA][TRACE] Predicted token_id: " << token_id << std::endl;
    std::cout << "[CUDA][TRACE] Max probability: " << max_prob << std::endl;

    auto expected_data = utils::load_data(data_path + "/probs.bin", "cpu");
    Tensor &expected_probs = expected_data["probs"];

    float max_err = 0.0f;
    for (int64_t v = 0; v < probs.numel(); ++v) {
      float err = std::abs(probs(v) - expected_probs(v));
      max_err = std::max(max_err, err);
    }

    if (max_err < 1e-2f) {
      passed++;
      std::cout << "[CUDA][INFO] Sample " << i
                << " Passed | Max Error: " << max_err << std::endl;
    } else {
      std::cout << "[CUDA][INFO] Sample " << i
                << " Failed | Max Error: " << max_err << std::endl;
    }
  }

  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "[CUDA][INFO] Final Result: " << passed << "/" << total
            << " passed." << std::endl;

  return 0;
}
