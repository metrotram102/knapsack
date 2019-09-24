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

void CopyRecordSet(uint32_t* recordSet, const uint32_t* set, const uint32_t* lowerBoundAdditionalSet, ssize_t blockCount) {
    memcpy(recordSet, set, blockCount * sizeof(*set));
    for (ssize_t i = 0; i < blockCount; ++i) {
        recordSet[i] |= lowerBoundAdditionalSet[i];
    }
}

void BranchCPU(ssize_t e, const ssize_t* ind,
                const int* weightNode, const int* priceNode, const ssize_t* step, const ssize_t* last, const uint32_t* set,
                int* newWeightNode, int* newPriceNode, ssize_t* newStep, ssize_t* newLast, uint32_t* newSet,
                ssize_t stepCount, ssize_t blockCount, ssize_t n, int W, const int* weight, const int* price) {
    ssize_t step_e = step[e], last_e = last[e], k = 0;
    ssize_t mayBeStepCount = last_e - step_e + 1;
    ssize_t realStepCount = mayBeStepCount < stepCount ? mayBeStepCount : stepCount;
    for (; k < realStepCount; ++k) {
        memcpy(newSet + (ind[e] + k) * blockCount, set + e * blockCount, blockCount * sizeof(*set));
        ssize_t newStepInd = step_e + k + 1;
        if (newStepInd - 1 == last_e) {
            newWeightNode[ind[e] + k] = weightNode[e];
            newPriceNode[ind[e] + k] = priceNode[e];
            for (; newStepInd < n && weightNode[e] + weight[newStepInd] > W; ++newStepInd);
            newStep[ind[e] + k] = newStepInd;
            newLast[ind[e] + k] = newStepInd;
            break;    
        } else {
            if (k == stepCount - 1) {
                newWeightNode[ind[e] + k] = weightNode[e];
                newPriceNode[ind[e] + k] = priceNode[e];
                newStep[ind[e] + k] = newStepInd - 1;
                newLast[ind[e] + k] = last_e;
            } else {
                newWeightNode[ind[e] + k] = weightNode[e] - weight[newStepInd - 1];
                newPriceNode[ind[e] + k] = priceNode[e] - price[newStepInd - 1];
                newStep[ind[e] + k] = newStepInd;
                newLast[ind[e] + k] = last_e;
                newSet[(ind[e] + k) * blockCount + (newStepInd - 1) / BLOCK_SIZE] ^= (((uint32_t)1) << ((uint32_t)((newStepInd - 1) % BLOCK_SIZE)));                
            }
        }
    }
}

void BoundCPU(ssize_t e, int* weightNode, int* priceNode, ssize_t* last, uint32_t* set, int* lowerBound, int* upperBound, uint32_t* lowerBoundAdditionalSet,
		ssize_t blockCount, ssize_t n, int W, const int* weight, const int* price) {
    ssize_t i = last[e];
    int w_e = weightNode[e], p_e = priceNode[e], weight_i = 0, price_i = 0;
    for (; ; ++i) {
        weight_i = weight[i];
        price_i = price[i];
        if (i < n && w_e + weight_i <= W) {
            w_e += weight_i;
            p_e += price_i;
            set[e * blockCount + i / BLOCK_SIZE] |= (((uint32_t)1) << ((uint32_t)(i % BLOCK_SIZE)));
        } else {
            break;
        }
    }
    upperBound[e] = p_e + (weight_i ? (W - w_e) * price_i / weight_i : 0);
    weightNode[e] = w_e;
    priceNode[e] = p_e;
    last[e] = i;

    for (; i < n; ++i) {
        weight_i = weight[i];
        price_i = price[i];
        if (w_e + weight_i <= W) {
            w_e += weight_i;
            p_e += price_i;
            lowerBoundAdditionalSet[e * blockCount + i / BLOCK_SIZE] |= (((uint32_t)1) << ((uint32_t)(i % BLOCK_SIZE)));
        }
    }
    lowerBound[e] = p_e;    
}

int main(int argc, char* argv[]) {
    std::chrono::high_resolution_clock::time_point totalStart = std::chrono::high_resolution_clock::now();
 
    if (argc != 3 && argc != 4) {
        std::cerr << "Usage: " << argv[0] << " input_file output_file" << std::endl;
        std::cerr << "Or: " << argv[0] << " input_file output_file count_of_steps_in_branching" << std::endl;
        return 0;
    }

    ssize_t stepCount = 2;
    if (argc == 4) {
        stepCount = (ssize_t)atoi(argv[3]);
        if (stepCount < 2 || stepCount > 10) {
            std::cerr << "count_of_steps_in_branching must be in [2, 10]" << std::endl;
            return 0;
        }
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
    ssize_t q = 1;
    
    int* weightNode = (int*)calloc(q, sizeof(*weightNode));
    int* priceNode = (int*)calloc(q, sizeof(*priceNode));
    ssize_t* step = (ssize_t*)calloc(q, sizeof(*step));	
    ssize_t* last = (ssize_t*)calloc(q, sizeof(*last));
    uint32_t* set = (uint32_t*)calloc(q * blockCount, sizeof(*set));
    int* lowerBound = (int*)calloc(q, sizeof(*lowerBound));
    int* upperBound = (int*)calloc(q, sizeof(*upperBound));
    uint32_t* lowerBoundAdditionalSet = (uint32_t*)calloc(q * blockCount, sizeof(*lowerBoundAdditionalSet));
    
    BoundCPU(0, weightNode, priceNode, last, set, lowerBound, upperBound, lowerBoundAdditionalSet, blockCount, n, W, weight, price);

    int record = lowerBound[0];
    uint32_t* recordSet = (uint32_t*)calloc(blockCount, sizeof(*recordSet));
    CopyRecordSet(recordSet, set, lowerBoundAdditionalSet, blockCount);
 
    free(lowerBound);
    free(upperBound);
    free(lowerBoundAdditionalSet);

    ssize_t stepNum = 0;
    while (q > 0) {
        std::cout << "Step " << stepNum + 1 << ", q = " << q << std::endl;
        ++stepNum;
        
        ssize_t* ind = (ssize_t*)malloc((q + 1) * sizeof(*ind));
        ind[0] = 0;
        for (ssize_t e = 1; e <= q; ++e) {
            ind[e] = ind[e - 1] + std::min(last[e - 1] - step[e - 1] + 1, stepCount);
        }

        int* newWeightNode = (int*)calloc(ind[q], sizeof(*newWeightNode));
        int* newPriceNode = (int*)calloc(ind[q], sizeof(*newPriceNode));
        ssize_t* newStep = (ssize_t*)calloc(ind[q], sizeof(*newStep));		
        ssize_t* newLast = (ssize_t*)calloc(ind[q], sizeof(*newLast));
        uint32_t* newSet = (uint32_t*)calloc(ind[q] * blockCount, sizeof(*newSet));
        
        for (ssize_t e = 0; e < q; ++e) {
            BranchCPU(e, ind, weightNode, priceNode, step, last, set,
                        newWeightNode, newPriceNode, newStep, newLast, newSet,
                        stepCount, blockCount, n, W, weight, price);
        }
        q = ind[q];
        free(ind);

        weightNode = (int*)realloc(weightNode, q * sizeof(*weightNode));
        priceNode = (int*)realloc(priceNode, q * sizeof(*priceNode));
        step = (ssize_t*)realloc(step, q * sizeof(*step));		
        last = (ssize_t*)realloc(last, q * sizeof(*last));
        set = (uint32_t*)realloc(set, q * blockCount * sizeof(*set));
        
        memcpy(weightNode, newWeightNode, q * sizeof(*weightNode));
        memcpy(priceNode, newPriceNode, q * sizeof(*priceNode));
        memcpy(step, newStep, q * sizeof(*step));		
        memcpy(last, newLast, q * sizeof(*last));
        memcpy(set, newSet, q * blockCount * sizeof(*set));
        
        free(newWeightNode);
        free(newPriceNode);
        free(newStep);		
        free(newLast);
        free(newSet);
        
        lowerBound = (int*)calloc(q, sizeof(*lowerBound));
        upperBound = (int*)calloc(q, sizeof(*upperBound));
        lowerBoundAdditionalSet = (uint32_t*)calloc(q * blockCount, sizeof(*lowerBoundAdditionalSet));
        
        for (ssize_t e = 0; e < q; ++e) {
            BoundCPU(e, weightNode, priceNode, last, set, lowerBound, upperBound, lowerBoundAdditionalSet, blockCount, n, W, weight, price);
            if (lowerBound[e] > record) {
                record = lowerBound[e];
                CopyRecordSet(recordSet, set + e * blockCount, lowerBoundAdditionalSet + e * blockCount, blockCount);
            }
        }

        ssize_t i = 0, j = q - 1;
        while (true) {
            while (i < q && upperBound[i] > record) {
                ++i;
            }
            while (j >= 0 && upperBound[j] <= record) {
                --j;
            }
            if (i >= j) {
                q = i;
                break;
            }
            weightNode[i] = weightNode[j];
            priceNode[i] = priceNode[j];
            step[i] = step[j];			
            last[i] = last[j];
            memcpy(set + i * blockCount, set + j * blockCount, blockCount * sizeof(*set));
            std::swap(upperBound[i], upperBound[j]);
        }
        
        weightNode = (int*)realloc(weightNode, q * sizeof(*weightNode));
        priceNode = (int*)realloc(priceNode, q * sizeof(*priceNode));
        step = (ssize_t*)realloc(step, q * sizeof(*step));		
        last = (ssize_t*)realloc(last, q * sizeof(*last));
        set = (uint32_t*)realloc(set, q * blockCount * sizeof(*set));

        free(lowerBound);
        free(upperBound);
        free(lowerBoundAdditionalSet);
    }

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

    free(weightNode);
    free(priceNode);
    free(step);	
    free(last);
    free(set);
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
