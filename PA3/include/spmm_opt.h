#ifndef SpMM_OPT_H
#define SpMM_OPT_H
#include "spmm_base.h"
#include <string>

class SpMMOpt : public SpMM
{
public:
    SpMMOpt(int *dev_out_ptr, int *dev_out_idx, int out_num_v, int out_num_e, int out_feat_in) : SpMM(dev_out_ptr, dev_out_idx, out_num_v, out_num_e, out_feat_in) {}
    SpMMOpt(CSR *g, int out_feat_in) : SpMM(g, out_feat_in) {}
    ~SpMMOpt() {
        if (target) checkCudaErrors(cudaFree(target));
        if (ptr_scheduled) checkCudaErrors(cudaFree(ptr_scheduled));
    }
     
    virtual void preprocess(float *vin, float *vout);

    virtual void run(float *vin, float *vout);

    void edgesort();

    void neighbor_grouping(int neighbor_num);

private:
    int num_target;
    int * d_row;

    int task_max;
    int task_width;
    // int last_size;
    int * all_width;
    int * all_index;
    int * task_index;

    int merge_divide;

    std::string task_name;
    int * task_choice;

    int *target, *ptr_scheduled;
};
#endif