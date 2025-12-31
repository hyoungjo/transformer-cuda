#include "model.hpp"
#include "tensor.hpp"
#include "utils.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#define WARMUPS 5
#define ITERATIONS 20

using namespace std::chrono;

int main(int, char *argv[]) {
  std::cout << "[CUDA][INFO] Loading model weights..." << std::endl;

  std::string model_name = std::string(argv[1]);
  std::string data_dir = "data/" + model_name + "/";
  std::unique_ptr<Model> model = utils::load_model(model_name, data_dir);

  int total = 15;
  std::cout << "[CUDA][INFO] Starting inference loop [0, " << total << ")"
            << std::endl;

  for (int i = 0; i < total; ++i) {
    std::string data_path = data_dir + std::to_string(i) + "/";

    auto input_data = utils::load_data(data_path + "input_ids.bin", "cpu");
    std::vector<int> input_ids;
    for (float id : input_data["input_ids"]) {
      input_ids.push_back(static_cast<int>(id));
    }

    int seq_len = input_ids.size();
    int *d_input_ids;
    cudaMalloc(&d_input_ids, seq_len * sizeof(int));
    cudaMemcpy(d_input_ids, input_ids.data(), seq_len * sizeof(int),
               cudaMemcpyHostToDevice);

    // 'volatile' prevents the compiler from deleting the lines
    for (int w = 0; w < WARMUPS; ++w) {
      Tensor logits = model->forward(d_input_ids, seq_len);
      asm volatile("" : : "g"(logits.d_data) : "memory");
    }

    std::vector<double> durations;
    durations.reserve(ITERATIONS);

    for (int b = 0; b < ITERATIONS; ++b) {
      auto start = high_resolution_clock::now();
      Tensor logits = model->forward(d_input_ids, seq_len);
      auto end = high_resolution_clock::now();
      auto execution_time = duration_cast<microseconds>(end - start).count();
      durations.push_back(execution_time);
    }
    cudaFree(d_input_ids);

    size_t n = durations.size();
    std::sort(durations.begin(), durations.end());

    auto get_percentile = [&](double p) {
      double pos = (n - 1) * p; // e.g., 10 * 0.5 = 5
      size_t idx = static_cast<size_t>(pos);
      double frac = pos - idx;
      if (frac == 0) {
        return durations[idx];
      }
      return durations[idx] * (1.0 - frac) + durations[idx + 1] * frac;
    };

    double median = get_percentile(0.5);
    double q1 = get_percentile(0.25);
    double q3 = get_percentile(0.75);
    double iqr = q3 - q1;

    std::cout << "Text " << i << " | Seq Len: " << input_ids.size()
              << "\n   Median: " << median << " μs"
              << "\n   IQR:    " << iqr << " μs (" << q1 << " to " << q3 << ")"
              << "\n   " << WARMUPS << " warmups, " << ITERATIONS
              << " iterations" << std::endl;

    // Other statistics
    // double sum = std::accumulate(durations.begin(), durations.end(), 0.0);
    // double mean = sum / n;
    // double squared_sum =
    //     std::accumulate(durations.begin(), durations.end(), 0.0,
    //                     [](double sum, double e) { return sum + e * e; });
    // double stdev = std::sqrt(squared_sum / n - mean * mean);
  }

  return 0;
}
