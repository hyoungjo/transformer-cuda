#include "gpt2.hpp"
#include "operations.hpp"
#include "tensor.hpp"
#include "utils.hpp"
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

int main() {
  std::cout << "[CPP][INFO] Loading model weights..." << std::endl;

  GPT2 model("data/weights.bin");

  int total = 15;
  int passed = 0;

  std::cout << "[CPP][INFO] Starting inference loop [0, " << total << ")"
            << std::endl;

  for (int i = 0; i < total; ++i) {
    std::string data_path = "data/" + std::to_string(i);

    auto input_data = utils::load_data(data_path + "/input_ids.bin");
    std::vector<int> input_ids;
    for (float id : input_data["input_ids"]) {
      input_ids.push_back(static_cast<int>(id));
    }

    std::cout << "[CPP][TRACE] Input size: " << input_ids.size() << " tokens"
              << std::endl;

    Tensor logits = model.forward(input_ids);
    Tensor probs = logits; // shallow copy, no copy constructor
    operations::softmax(probs);

    int64_t token_id = 0;
    float max_prob = -1e9;
    for (int64_t v = 0; v < probs.numel(); ++v) {
      if (probs(v) > max_prob) {
        max_prob = probs(v);
        token_id = v;
      }
    }

    std::cout << "[CPP][TRACE] Predicted token_id: " << token_id << std::endl;
    std::cout << "[CPP][TRACE] Max probability: " << max_prob << std::endl;

    auto expected_data = utils::load_data(data_path + "/probs.bin");
    Tensor &expected_probs = expected_data["probs"];

    float max_err = 0.0f;
    for (int64_t v = 0; v < probs.numel(); ++v) {
      float err = std::abs(probs(v) - expected_probs(v));
      max_err = std::max(max_err, err);
    }

    if (max_err < 1e-3f) {
      passed++;
      std::cout << "[CPP][INFO] Sample " << i
                << " Passed | Max Error: " << max_err << std::endl;
    } else {
      std::cout << "[CPP][INFO] Sample " << i
                << " Failed | Max Error: " << max_err << std::endl;
    }
  }

  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "[CPP][INFO] Final Result: " << passed << "/" << total
            << " passed." << std::endl;

  return 0;
}