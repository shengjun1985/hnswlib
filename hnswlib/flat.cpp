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

template <typename T> inline
void heap_swap_top (size_t k,
                    T * bh_val, int * bh_ids,
                    T val, int ids)
{
    bh_val--; /* Use 1-based indexing for easier node->child translation */
    bh_ids--;
    size_t i = 1, i1, i2;
    while (1) {
        i1 = i << 1;
        i2 = i1 + 1;
        if (i1 > k)
            break;
        if (i2 == k + 1 || (bh_val[i1] > bh_val[i2])) {
            if (val > bh_val[i1])
                break;
            bh_val[i] = bh_val[i1];
            bh_ids[i] = bh_ids[i1];
            i = i1;
        }
        else {
            if (val > bh_val[i2])
                break;
            bh_val[i] = bh_val[i2];
            bh_ids[i] = bh_ids[i2];
            i = i2;
        }
    }
    bh_val[i] = val;
    bh_ids[i] = ids;
}

template <typename T> inline
void heap_pop (size_t k, T* bh_val, int* bh_ids)
{
    bh_val--; /* Use 1-based indexing for easier node->child translation */
    bh_ids--;
    T val = bh_val[k];
    size_t i = 1, i1, i2;
    while (1) {
        i1 = i << 1;
        i2 = i1 + 1;
        if (i1 > k)
            break;
        if (i2 == k + 1 || (bh_val[i1] > bh_val[i2])) {
            if ((val > bh_val[i1]))
                break;
            bh_val[i] = bh_val[i1];
            bh_ids[i] = bh_ids[i1];
            i = i1;
        }
        else {
            if ((val > bh_val[i2]))
                break;
            bh_val[i] = bh_val[i2];
            bh_ids[i] = bh_ids[i2];
            i = i2;
        }
    }
    bh_val[i] = bh_val[k];
    bh_ids[i] = bh_ids[k];
}

template <typename T> inline
void Flat(const size_t nb, const uint8_t* xb,
          const size_t nq, const uint8_t* xq,
          const int k, int *id, T *dis) {

#pragma omp parallel for
    for(int i=0;i<nq;i++){
        int *rst_id = id + i*k;
        T *rst_dis = dis + i*k;
        const uint8_t *xq_i = xq + i * code_size;
        for(int j=0;j<k;j++){
            rst_id[j] = -1;
            rst_dis[j] = (typeid(T) == typeid(float)) ? (1.0 / 0.0) : 0x7fffffff;
        }
        for(int j=0;j<nb;j++){
#ifdef Hamming
            int dis_ij = hnswlib::xor_popcnt(xq_i, xb + j * code_size, code_size);
#else
  
#endif
            if (dis_ij < rst_dis[0]) {
                heap_swap_top(k, rst_id, rst_dis, i, dis_ij);
            }
        }
        // id miss
        std::sort(rst_dis, rst_dis+k);
    }

}

int main() {
    freopen("flat.txt", "w", stdout);
    LoadData();


    printf("ntotal = %ld d = %ld build time %zu\n", nb, d, 0);

    int k=10;

    timeval t0;
    gettimeofday(&t0, 0);

#ifdef Hamming
    int *id = new int[nq*k];
    int *dis = new int[nq*k];
    Flat(nb,xb,nq,xq,k,id,dis);
#else

#endif

    timeval t1;
    gettimeofday(&t1, 0);
    printf("search nq %zu topk %d time %ldms\n", nq, 10, getTime(t1,t0));

    for (size_t i=0;i<nq;i++){
        for (size_t j=0;j<k;j++) {
#ifdef Hamming
            printf("%d %zu\n", dis[i*k+j], id[i*k+j]);
#else

#endif
        }
        printf("\n");
    }

    delete xq;
    return 0;
}
