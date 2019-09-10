#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct TItem {
    int price;
    int weight;
    int index;
    bool operator<(const TItem& other) const {
        return (double)price / weight > (double)other.price / other.weight;
    }
};

const int BLOCK_SIZE = 32;

const int THREADS_PER_BLOCK = 192;

void BranchCPU(ssize_t e, int* w, int* p, int* s, int* U_old, uint32_t* X_old, int block_count, int k, int* weight, int* price) {
    int s_e = s[e];
    if (k < s_e) {
        w[e] -= weight[k];
        p[e] -= price[k];
        X_old[e * block_count + k / BLOCK_SIZE] |= (1 << (k % BLOCK_SIZE));
    } else {
        ++s[e];
        U_old[e] = 0;
    }
}

__global__ void BranchGPU(int* w, int* p, int* s, int* U_old, uint32_t* X_old, int block_count, int k, int* weight, int* price, ssize_t q) {
    ssize_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= q) {
    	return;
    }
    int s_e = s[e];
    if (k < s_e) {
        w[e] -= weight[k];
        p[e] -= price[k];
        X_old[e * block_count + k / BLOCK_SIZE] |= (1 << (k % BLOCK_SIZE));
    } else {
        ++s[e];
        U_old[e] = 0;
    }
}

void BoundCPU(ssize_t e, int* w, int* p, int* s, int* L, int* U, uint32_t* L_set, int block_count, int k, int n, int W, int* weight, int* price) {
    int i = s[e], w_e = w[e], p_e = p[e], weight_i = 0, price_i = 0;
    for (; ; ++i) {
        weight_i = weight[i];
        price_i = price[i];
        if (i < n && w_e + weight_i <= W) {
            w_e += weight_i;
            p_e += price_i;
            L_set[e * block_count + i / BLOCK_SIZE] |= (1 << (i % BLOCK_SIZE));
        } else {
            break;
        }
    }
    U[e] = p_e + (weight_i ? (W - w_e) * price_i / weight_i : 0);
    w[e] = w_e;
    p[e] = p_e;
    s[e] = i;

    for (; i < n; ++i) {
        weight_i = weight[i];
        price_i = price[i];
        if (w_e + weight_i <= W) {
            w_e += weight_i;
            p_e += price_i;
            L_set[e * block_count + i / BLOCK_SIZE] |= (1 << (i % BLOCK_SIZE));
        }
    }
    L[e] = p_e;
}

__global__ void BoundGPU(int* w, int* p, int* s, int* L, int* U, uint32_t* L_set, int block_count, int k, int n, int W, int* weight, int* price, ssize_t q) {
    ssize_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= q) {
    	return;
    }
    int i = s[e], w_e = w[e], p_e = p[e], weight_i = 0, price_i = 0;
    for (; ; ++i) {
        weight_i = weight[i];
        price_i = price[i];
        if (i < n && w_e + weight_i <= W) {
            w_e += weight_i;
            p_e += price_i;
            L_set[e * block_count + i / BLOCK_SIZE] |= (1 << (i % BLOCK_SIZE));
        } else {
            break;
        }
    }
    U[e] = p_e + (weight_i ? (W - w_e) * price_i / weight_i : 0);
    w[e] = w_e;
    p[e] = p_e;
    s[e] = i;

    for (; i < n; ++i) {
        weight_i = weight[i];
        price_i = price[i];
        if (w_e + weight_i <= W) {
            w_e += weight_i;
            p_e += price_i;
            L_set[e * block_count + i / BLOCK_SIZE] |= (1 << (i % BLOCK_SIZE));
        }
    }
    L[e] = p_e;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " input_file output_file" << std::endl;
        return 0;
    }
    std::ifstream fin(argv[1]);
    std::ofstream fout(argv[2]);
    int n, W;
    fin >> n >> W;
    std::vector<TItem> items(n);
    for (int i = 0; i < n; ++i) {
        fin >> items[i].price >> items[i].weight;
        items[i].index = i + 1;
    }
    std::sort(items.begin(), items.end());

    int* weight = (int*)malloc((n + 1) * sizeof(*weight));
    int* price = (int*)malloc((n + 1) * sizeof(*price));
    for (int i = 0; i < n; ++i) {
        weight[i] = items[i].weight;
        price[i] = items[i].price;
    }
    weight[n] = price[n] = 0;

    std::chrono::high_resolution_clock::time_point total_start = std::chrono::high_resolution_clock::now();

    int *cuda_weight = nullptr, *cuda_price = nullptr;
    ssize_t q = 1;
    int* w = (int*)malloc(q * sizeof(*w));
    int* p = (int*)malloc(q * sizeof(*p));
    int* s = (int*)malloc(q * sizeof(*s));
    int* L = (int*)malloc(q * sizeof(*L));
    int* U = (int*)malloc(q * sizeof(*U));

    const int block_count = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    uint32_t* X = (uint32_t*)calloc(q * block_count, sizeof(*X));
    w[0] = p[0] = s[0] = 0;

    uint32_t* record_set = (uint32_t*)calloc(block_count, sizeof(*X));
    BoundCPU(0, w, p, s, L, U, record_set, block_count, 0, n, W, weight, price);
    int record = L[0];
    free(L);

    for (int k = 0; k < n; ++k) {
        std::cout << "Step " << k + 1 << ", q = " << q << std::endl;
        if (q > 5000000) {
            if (cuda_weight == nullptr) {
                cudaMalloc(&cuda_weight, (n + 1) * sizeof(*cuda_weight));
                cudaMalloc(&cuda_price, (n + 1) * sizeof(*cuda_price));
                cudaMemcpy(cuda_weight, weight, (n + 1) * sizeof(*cuda_weight), cudaMemcpyHostToDevice);
                cudaMemcpy(cuda_price, price, (n + 1) * sizeof(*cuda_price), cudaMemcpyHostToDevice);
            }
            int *w_new, *p_new, *s_new, *L_new, *U_new, *U_old;
            uint32_t *X_old, *L_new_set;
            cudaMalloc(&w_new, q * sizeof(*w_new));
            cudaMalloc(&p_new, q * sizeof(*p_new));
            cudaMalloc(&s_new, q * sizeof(*s_new));
            cudaMalloc(&U_old, q * sizeof(*U_old));
            cudaMalloc(&X_old, q * block_count * sizeof(*X_old));
            cudaMemcpy(w_new, w, q * sizeof(*w), cudaMemcpyHostToDevice);
            cudaMemcpy(p_new, p, q * sizeof(*p), cudaMemcpyHostToDevice);
            cudaMemcpy(s_new, s, q * sizeof(*s), cudaMemcpyHostToDevice);
            cudaMemcpy(U_old, U, q * sizeof(*U), cudaMemcpyHostToDevice);
            cudaMemcpy(X_old, X, q * block_count * sizeof(*X), cudaMemcpyHostToDevice);

            const ssize_t q_block = (q + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            BranchGPU<<<q_block, THREADS_PER_BLOCK>>>(w_new, p_new, s_new, U_old,
                                                    X_old, block_count, k, cuda_weight, cuda_price, q);
            cudaDeviceSynchronize();
            
            cudaMemcpy(U, U_old, q * sizeof(*U), cudaMemcpyDeviceToHost);
            cudaFree(U_old);
            
            cudaMalloc(&L_new, q * sizeof(*L_new));
            cudaMalloc(&U_new, q * sizeof(*U_new));
            cudaMalloc(&L_new_set, q * block_count * sizeof(*X));
            cudaMemcpy(L_new_set, X, q * block_count * sizeof(*X), cudaMemcpyHostToDevice);

            BoundGPU<<<q_block, THREADS_PER_BLOCK>>>(w_new, p_new, s_new, L_new, U_new,
                                                    L_new_set, block_count, k, n, W, cuda_weight, cuda_price, q);
            cudaDeviceSynchronize();

            w = (int*)realloc(w, 2 * q * sizeof(*w));
            p = (int*)realloc(p, 2 * q * sizeof(*p));
            s = (int*)realloc(s, 2 * q * sizeof(*s));
            U = (int*)realloc(U, 2 * q * sizeof(*U));
            X = (uint32_t*)realloc(X, 2 * q * block_count * sizeof(*X));
            memcpy(X + q * block_count, X, q * block_count * sizeof(*X));
            cudaMemcpy(w + q, w_new, q * sizeof(*w), cudaMemcpyDeviceToHost);
            cudaMemcpy(p + q, p_new, q * sizeof(*p), cudaMemcpyDeviceToHost);
            cudaMemcpy(s + q, s_new, q * sizeof(*s), cudaMemcpyDeviceToHost);
            cudaMemcpy(U + q, U_new, q * sizeof(*U), cudaMemcpyDeviceToHost);
            cudaMemcpy(X, X_old, q * block_count * sizeof(*X), cudaMemcpyDeviceToHost);
            cudaFree(w_new);
            cudaFree(p_new);
            cudaFree(s_new);
            cudaFree(U_new);
            cudaFree(X_old);

            int *L_new_CPU = (int*)malloc(q * sizeof(*L_new_CPU));
            uint32_t* L_new_set_CPU = (uint32_t*)malloc(q * block_count * sizeof(*X));
            cudaMemcpy(L_new_CPU, L_new, q * sizeof(*L_new), cudaMemcpyDeviceToHost);
            cudaMemcpy(L_new_set_CPU, L_new_set, q * block_count * sizeof(*X), cudaMemcpyDeviceToHost);
            cudaFree(L_new);
            cudaFree(L_new_set);
            for (ssize_t e = 0; e < q; ++e) {
                if (L_new_CPU[e] > record) {
                    record = L_new_CPU[e];
                    memcpy(record_set, L_new_set_CPU + e * block_count, block_count * sizeof(*X));
                    for (int i = k + 1; i < s[q + e]; ++i) {
                        record_set[i / BLOCK_SIZE] |= (1 << (i % BLOCK_SIZE));
                    }
                }
            }
            free(L_new_CPU);
            free(L_new_set_CPU);

        } else {

            w = (int*)realloc(w, 2 * q * sizeof(*w));
            p = (int*)realloc(p, 2 * q * sizeof(*p));
            s = (int*)realloc(s, 2 * q * sizeof(*s));
            X = (uint32_t*)realloc(X, 2 * q * block_count * sizeof(*X));
            memcpy(w + q, w, q * sizeof(*w));
            memcpy(p + q, p, q * sizeof(*p));
            memcpy(s + q, s, q * sizeof(*s));
            memcpy(X + q * block_count, X, q * block_count * sizeof(*X));
            for (ssize_t e = 0; e < q; ++e) {
                BranchCPU(e, w + q, p + q, s + q, U, X, block_count, k, weight, price);
            }

            U = (int*)realloc(U, 2 * q * sizeof(*U));
            int* L_new = (int*)malloc(q * sizeof(*L_new));
            uint32_t* L_new_set = (uint32_t*)malloc(q * block_count * sizeof(*X));
            memcpy(L_new_set, X + q * block_count, q * block_count * sizeof(*X));
            for (ssize_t e = 0; e < q; ++e) {
                BoundCPU(e, w + q, p + q, s + q, L_new, U + q, L_new_set, block_count, k, n, W, weight, price);
                if (L_new[e] > record) {
                    record = L_new[e];
                    memcpy(record_set, L_new_set + e * block_count, block_count * sizeof(*X));
                    for (int i = k + 1; i < s[q + e]; ++i) {
                        record_set[i / BLOCK_SIZE] |= (1 << (i % BLOCK_SIZE));
                    }
                }
            }
            free(L_new);
            free(L_new_set);
        }

        for (ssize_t i = 0, j = 2 * q - 1; ;) {
            while (i < 2 * q && U[i] > record) {
                ++i;
            }
            while (j >= 0 && U[j] <= record) {
                --j;
            }
            if (i >= j) {
                q = j + 1;
                break;
            }
            w[i] = w[j];
            p[i] = p[j];
            s[i] = s[j];
            std::swap(U[i], U[j]);
            memcpy(X + i * block_count, X + j * block_count, block_count * sizeof(*X));
        }
        if (q == 0) {
            break;
        }
    }

    free(w);
    free(p);
    free(s);
    free(U);
    free(weight);
    free(price);
    if (cuda_weight != nullptr) {
        cudaFree(cuda_weight);
        cudaFree(cuda_price);
    }

    std::chrono::high_resolution_clock::time_point total_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration_cast<std::chrono::duration<double>>(total_end - total_start).count();
    std::cout << "Total time: " << total_time << std::endl;

    fout << record << std::endl;
    std::vector<int> record_ind;
    for (int i = 0; i < n; ++i) {
        if (record_set[i / BLOCK_SIZE] & (1 << (i % BLOCK_SIZE))) {
            record_ind.push_back(items[i].index);
        }
    }
    std::sort(record_ind.begin(), record_ind.end());
    for (auto ind : record_ind) {
        fout << ind << " ";
    }
    fout << std::endl;

    free(record_set);

    return 0;
}
