#include <stdio.h>
#include <vector>

const char* flat = "flat.txt";
const char* hnsw = "hnsw.txt";

int main(){
    FILE *f = fopen(flat,"r");
    FILE *h = fopen(hnsw,"r");

    int nq=10, k=10;
    fscanf(f,"%*[^\n]%*c");
    fscanf(f,"search nq %d topk %d time %*dms\n", &nq, &k);
    fscanf(h,"%*[^\n]%*c");
    fscanf(h,"%*[^\n]%*c");

    int correct=0;

    printf("%d %d\n", nq, k);

    std::vector<float> dis_f;
    std::vector<float> dis_h;
    dis_f.resize(k);
    dis_h.resize(k);
    for(int i=0;i<nq;i++){
        for(int j=0;j<k;j++)
            fscanf(f,"%f %*d", &dis_f[j]);
        for(int j=0;j<k;j++)
            fscanf(h,"%f %*d", &dis_h[j]);

        float max_dis = dis_f.back();
        int max_dis_num = 1;
        for(int j=k-2;j>=0;j--){
            if (dis_f[j] == max_dis) max_dis_num++;
            else break;
        }

        int ac;
        for(ac=0;ac<k;ac++){
            if(dis_h[ac]>max_dis) break;
            if(dis_h[ac]==max_dis){
                if(max_dis_num > 0) max_dis_num--;
                else break;
            }
        }

        correct += ac;
    }

    printf("recall %lf\n", ((double)correct)/(nq*k));

    return 0;
}
