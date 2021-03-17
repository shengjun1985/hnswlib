#include "hnswalg.h"
#include "space_hamming.h"
#include "space_jaccard.h"

#include <stdio.h>
#include <memory>
#include <sys/time.h>
#include <unistd.h>
using namespace std;

#define Hamming

const uint8_t lookup8bit[256] = {
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8
};

long int getTime(timeval end, timeval start) {
	return 1000*(end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1000;
}

int M = 24;
int EF_B = 128;
int EF_Q = 128;

size_t code_size;
size_t d;                          // dimension
size_t nb;                        // database size
size_t nq;                          // nb of queries
uint8_t *xb;
uint8_t *xq;

void LoadData(){
    FILE *fi = fopen("../../00.txt","rb");

    // fread(&code_size, sizeof(code_size), 1, fi);
    // fread(&nb, sizeof(nb), 1, fi);
    // fread(&nq, sizeof(nq), 1, fi);
    code_size = 256;
    nb = 100000;
    nq = 100;

    d = code_size * 8;
    xb = new uint8_t[code_size * nb];
    xq = new uint8_t[code_size * nq];
    fread(xb, code_size * nb, 1, fi);
    fread(xq, code_size * nq, 1, fi);

    fclose(fi);
}

int main() {
    freopen("hnsw.txt", "w", stdout);
    LoadData();

#ifdef Hamming
    std::shared_ptr<hnswlib::HierarchicalNSW<int>> index;
    hnswlib::SpaceInterface<int>* space = new hnswlib::HammingSpace(d);
    index = std::make_shared<hnswlib::HierarchicalNSW<int>>(space, nb, M, EF_B);

    std::vector<std::vector<std::pair<int, size_t>>> rst;
#else
    std::shared_ptr<hnswlib::HierarchicalNSW<float>> index;
    hnswlib::SpaceInterface<float>* space = new hnswlib::JaccardSpace(d);
    index = std::make_shared<hnswlib::HierarchicalNSW<float>>(space, nb, M, EF_B);

    std::vector<std::vector<std::pair<float, size_t>>> rst;
#endif

    timeval b1;
    gettimeofday(&b1, 0);

    index->addPoint(xb, 0);
#pragma omp parallel for
    for (int i = 1; i < nb; ++i) {
        index->addPoint(xb + i * code_size, i);
    }

    timeval b2;
    gettimeofday(&b2, 0);

    delete xb;
    printf("ntotal = %ld d = %ld build time %zu\n", nb, d, getTime(b2,b1));

    index->setEf(EF_Q);

    rst.resize(nq);

    timeval t0;
    gettimeofday(&t0, 0);

#pragma omp parallel for
    for (int i=0; i<nq; i++){
        auto ret = index->searchKnnCloserFirst(xq + i * code_size, 10);
        rst[i].swap(ret);
    }

    timeval t1;
    gettimeofday(&t1, 0);
    printf("search nq %zu topk %d time %ldms\n", nq, 10, getTime(t1,t0));

    for (size_t i=0;i<nq;i++){
        for (auto &it : rst[i]) {
#ifdef Hamming
            printf("%d %zu\n", it.first, it.second);
#else
            printf("%f %zu\n", it.first, it.second);
#endif
        }
        printf("\n");
    }

    delete xq;
    return 0;
}
