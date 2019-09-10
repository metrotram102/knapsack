#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct TItem {
    int price;
    int weight;
    bool operator<(const TItem& other) const {
        return (double)price / weight > (double)other.price / other.weight;
    }
};

const int THREADS_PER_BLOCK = 192;

void BranchCPU(ssize_t e, int* w, int* p, int* s, int* U_old, int k, int* weight, int* price) {
    int s_e = s[e];
    if (k < s_e) {
        w[e] -= weight[k];
        p[e] -= price[k];
    } else {
        ++s[e];
        U_old[e] = 0;
    }
}

__global__ void BranchGPU(int* w, int* p, int* s, int* U_old, int k, int* weight, int* price, ssize_t q) {
    ssize_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= q) {
    	return;
    }
    int s_e = s[e];
    if (k < s_e) {
        w[e] -= weight[k];
        p[e] -= price[k];
    } else {
        ++s[e];
        U_old[e] = 0;
    }
}

void BoundCPU(ssize_t e, int* w, int* p, int* s, int* L, int* U, int k, int n, int W, int* weight, int* price) {
    int i = s[e], w_e = w[e], p_e = p[e], weight_i = 0, price_i = 0;
    for (; i <= n; ++i) {
        weight_i = weight[i];
        price_i = price[i];
        if (w_e + weight_i <= W) {
            w_e += weight_i;
            p_e += price_i;
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
        }
    }
    L[e] = p_e;
}

__global__ void BoundGPU(int* w, int* p, int* s, int* L, int* U, int k, int n, int W, int* weight, int* price, ssize_t q) {
    ssize_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= q) {
    	return;
    }
    int i = s[e], w_e = w[e], p_e = p[e], weight_i = 0, price_i = 0;
    for (; i <= n; ++i) {
        weight_i = weight[i];
        price_i = price[i];
        if (w_e + weight_i <= W) {
            w_e += weight_i;
            p_e += price_i;
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
    w[0] = p[0] = s[0]= 0;

    BoundCPU(0, w, p, s, L, U, 0, n, W, weight, price);
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
            cudaMalloc(&w_new, q * sizeof(*w_new));
            cudaMalloc(&p_new, q * sizeof(*p_new));
            cudaMalloc(&s_new, q * sizeof(*s_new));
            cudaMalloc(&U_old, q * sizeof(*U_old));
            cudaMemcpy(w_new, w, q * sizeof(*w), cudaMemcpyHostToDevice);
            cudaMemcpy(p_new, p, q * sizeof(*p), cudaMemcpyHostToDevice);
            cudaMemcpy(s_new, s, q * sizeof(*s), cudaMemcpyHostToDevice);
            cudaMemcpy(U_old, U, q * sizeof(*U), cudaMemcpyHostToDevice);

            const ssize_t q_block = (q + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            BranchGPU<<<q_block, THREADS_PER_BLOCK>>>(w_new, p_new, s_new, U_old, k, cuda_weight, cuda_price, q);
            cudaDeviceSynchronize();
            
            cudaMemcpy(U, U_old, q * sizeof(*U), cudaMemcpyDeviceToHost);
            cudaFree(U_old);
            cudaMalloc(&L_new, q * sizeof(*L_new));
            cudaMalloc(&U_new, q * sizeof(*U_new));

            BoundGPU<<<q_block, THREADS_PER_BLOCK>>>(w_new, p_new, s_new, L_new, U_new, k, n, W, cuda_weight, cuda_price, q);
            cudaDeviceSynchronize();

            int *L_new_CPU = (int*)malloc(q * sizeof(*L_new_CPU));
            cudaMemcpy(L_new_CPU, L_new, q * sizeof(*L_new), cudaMemcpyDeviceToHost);
            cudaFree(L_new);
            for (ssize_t e = 0; e < q; ++e) {
                record = std::max(record, L_new_CPU[e]);
            }
            free(L_new_CPU);

            w = (int*)realloc(w, 2 * q * sizeof(*w));
            p = (int*)realloc(p, 2 * q * sizeof(*p));
            s = (int*)realloc(s, 2 * q * sizeof(*s));
            U = (int*)realloc(U, 2 * q * sizeof(*U));
            cudaMemcpy(w + q, w_new, q * sizeof(*w), cudaMemcpyDeviceToHost);
            cudaMemcpy(p + q, p_new, q * sizeof(*p), cudaMemcpyDeviceToHost);
            cudaMemcpy(s + q, s_new, q * sizeof(*s), cudaMemcpyDeviceToHost);
            cudaMemcpy(U + q, U_new, q * sizeof(*U), cudaMemcpyDeviceToHost);
            cudaFree(w_new);
            cudaFree(p_new);
            cudaFree(s_new);
            cudaFree(U_new);

        } else {

            w = (int*)realloc(w, 2 * q * sizeof(*w));
            p = (int*)realloc(p, 2 * q * sizeof(*p));
            s = (int*)realloc(s, 2 * q * sizeof(*s));
            memcpy(w + q, w, q * sizeof(*w));
            memcpy(p + q, p, q * sizeof(*p));
            memcpy(s + q, s, q * sizeof(*s));
            for (ssize_t e = 0; e < q; ++e) {
                BranchCPU(e, w + q, p + q, s + q, U, k, weight, price);
            }

            U = (int*)realloc(U, 2 * q * sizeof(*U));
            int* L_new = (int*)malloc(q * sizeof(*L_new));
            for (ssize_t e = 0; e < q; ++e) {
                BoundCPU(e, w + q, p + q, s + q, L_new, U + q, k, n, W, weight, price);
                record = std::max(record, L_new[e]);
            }
            free(L_new);
        }

        for (ssize_t i = 0, j = 2 * q - 1; ;) {
            while (i < 2 * q && U[i] >= record) {
                ++i;
            }
            while (j >= 0 && U[j] < record) {
                --j;
            }
            if (i >= j) {
                q = i;
                break;
            }
            w[i] = w[j];
            p[i] = p[j];
            s[i] = s[j];
            std::swap(U[i], U[j]);
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

    return 0;
}
