#include <algorithm>
#include <cassert>
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
    ssize_t index;
    bool operator<(const TItem& other) const {
        return (double)price / weight > (double)other.price / other.weight;
    }
};

const ssize_t BLOCK_SIZE = 32;

void DFS(ssize_t blockCount, ssize_t n, int W, int* weight, int* price, int& record, uint32_t* recordSet) {
    int* currentSet = (int*)malloc(n * sizeof(*currentSet));
    for (ssize_t i = 0; i < n; ++i) {
        currentSet[i] = -1;
    }
    int upperBound = 0;
    int weightNode = 0;
    int priceNode = 0;
    uint32_t* lowerBoundAdditionalSet = (uint32_t*)malloc(blockCount * sizeof(*lowerBoundAdditionalSet));

    for (ssize_t i = 0; i >= 0;) {
        if (i == n) {
            --i;
        } else if (currentSet[i] == 1) {
            currentSet[i] = 0;
            weightNode -= weight[i];
            priceNode -= price[i];
            ++i;
        } else if (currentSet[i] == 0) {
            currentSet[i] = -1;
            --i;
        } else {
            if (i == 0 || currentSet[i - 1] != 1) {
                int weightForBounds = weightNode;
                int priceForBounds = priceNode;
                ssize_t j = i;
                memset(lowerBoundAdditionalSet, 0, blockCount * sizeof(*lowerBoundAdditionalSet));
                for (; j < n; ++j) {
                    if (weightForBounds + weight[j] > W) {
                        break;
                    }
                    weightForBounds += weight[j];
                    priceForBounds += price[j];
                    lowerBoundAdditionalSet[j / BLOCK_SIZE] |= (((uint32_t)1) << ((uint32_t)(j % BLOCK_SIZE)));
                }
                upperBound = priceForBounds + (j == n ? 0 : ((W - weightForBounds) * price[j] / weight[j]));
                if (upperBound > record) {
                    for (; j < n; ++j) {
                        if (weightForBounds + weight[j] <= W) {
                            weightForBounds += weight[j];
                            priceForBounds += price[j];
                            lowerBoundAdditionalSet[j / BLOCK_SIZE] |= (((uint32_t)1) << ((uint32_t)(j % BLOCK_SIZE)));
                        }
                    }
                    if (priceForBounds > record) {
                        record = priceForBounds;
                        memcpy(recordSet, lowerBoundAdditionalSet, blockCount * sizeof(*recordSet));
                        for (ssize_t k = 0; k < n; ++k) {
                            if (currentSet[k] == 1) {
                                recordSet[k / BLOCK_SIZE] |= (((uint32_t)1) << ((uint32_t)(k % BLOCK_SIZE)));
                            }
                        }
                    }
                }
            }
            if (upperBound <= record) {
                --i;
            } else {
                if (weightNode + weight[i] <= W) {
                    weightNode += weight[i];
                    priceNode += price[i];
                    currentSet[i] = 1;
                } else {
                    currentSet[i] = 0;
                }
                ++i;
            }
        }
    }
    free(currentSet);
    free(lowerBoundAdditionalSet);
}

int main(int argc, char* argv[]) {
    std::chrono::high_resolution_clock::time_point totalStart = std::chrono::high_resolution_clock::now();
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " input_file output_file" << std::endl;
        return 0;
    }

    std::ifstream fin(argv[1]);
    std::ofstream fout(argv[2]);

    int W;
    ssize_t n;
    fin >> n >> W;
    std::vector<TItem> items(n);
    for (ssize_t i = 0; i < n; ++i) {
        fin >> items[i].price >> items[i].weight;
        items[i].index = i + 1;
    }
    std::sort(items.begin(), items.end());

    int* weight = (int*)malloc((n + 1) * sizeof(*weight));
    int* price = (int*)malloc((n + 1) * sizeof(*price));
    for (ssize_t i = 0; i < n; ++i) {
        weight[i] = items[i].weight;
        price[i] = items[i].price;
    }
    weight[n] = price[n] = 0;

    const ssize_t blockCount = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int record = 0;
    uint32_t* recordSet = (uint32_t*)calloc(blockCount, sizeof(*recordSet));
    DFS(blockCount, n, W, weight, price, record, recordSet);

    fout << record << std::endl;
    std::vector<ssize_t> recordInd;
    for (ssize_t i = 0; i < n; ++i) {
        if (recordSet[i / BLOCK_SIZE] & (((uint32_t)1) << ((uint32_t)(i % BLOCK_SIZE)))) {
            recordInd.push_back(items[i].index);
        }
    }
    std::sort(recordInd.begin(), recordInd.end());
    for (auto ind : recordInd) {
        fout << ind << " ";
    }
    fout << std::endl;

    free(recordSet);
    free(weight);
    free(price);

    fin.close();
    fout.close();
    
    std::chrono::high_resolution_clock::time_point totalEnd = std::chrono::high_resolution_clock::now();
    double totalTime = std::chrono::duration_cast<std::chrono::duration<double>>(totalEnd - totalStart).count();
    std::cout << "Total time: " << totalTime << std::endl;
    return 0;
}
