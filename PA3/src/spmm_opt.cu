#include "spmm_opt.h"
#include <iostream>
#include <string>
#include <tuple>
#include <algorithm>

#define BLOCK_SIZE 1
#define BLOCK_SIZE_2 32
#define MAX_NE 64
#define MAX_NE_2 160
#define MAX_NE_3 128

__global__ void spmm_kernel_placeholder(int * row, int *ptr, int *idx, float *val, float *vin, float *vout, int * task, int task_width, int task_max, int * all_width, int num_v, int num_e, int INFEATURE, int * task_choice)
{
    int _tid = blockIdx.x * blockDim.x + threadIdx.x;
    // if (_tid >= task_width) return;
    int tid = task_choice[_tid];
    int width = all_width[tid];
    int _begin = tid * task_max, _end = _begin + all_width[tid];
    // int begin = task[tid * task_max], end = tid + MAX_NE;
    int begin = task[_begin], end = task[_end-1];

    float result = 0.0f;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float _val[MAX_NE];
    __shared__ int _idx[MAX_NE];
    _val[threadIdx.y] = val[begin + threadIdx.y];
    if (width > 32){
        _val[threadIdx.y + 32] = val[begin + threadIdx.y + 32];
    }

    _idx[threadIdx.y] = idx[begin + threadIdx.y];
    if (width > 32){
        _idx[threadIdx.y + 32] = idx[begin + threadIdx.y + 32];
    }
    __syncthreads();

    for (int i = begin; i <= end; i += 2){
        result += __ldg(&vin[_idx[i - begin] * INFEATURE + j]) * _val[i - begin];
        if (i + 1 <= end) {
            result += __ldg(&vin[_idx[i + 1 - begin] * INFEATURE + j]) * _val[i + 1 - begin];
        }
    }
    atomicAdd(&(vout[row[begin] * INFEATURE + j]), result);
}

__global__ void spmm_kernel_placeholder_160(int * row, int *ptr, int *idx, float *val, float *vin, float *vout, int * task, int task_width, int task_max, int * all_width, int num_v, int num_e, int INFEATURE, int * task_choice)
{
    int _tid = blockIdx.x * blockDim.x + threadIdx.x;
    // if (_tid >= task_width) return;
    int tid = task_choice[_tid];
    // int width = all_width[tid];
    int _begin = tid * task_max, _end = _begin + all_width[tid];
    // int begin = task[tid * task_max], end = tid + MAX_NE;
    int begin = task[_begin], end = task[_end-1];

    float result = 0.0f;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float _val[MAX_NE_2];
    __shared__ int _idx[MAX_NE_2];
    _val[threadIdx.y] = val[begin + threadIdx.y];
    _val[threadIdx.y + 32] = val[begin + threadIdx.y + 32];
    _val[threadIdx.y + 32 * 2] = val[begin + threadIdx.y + 32 * 2];
    _val[threadIdx.y + 32 * 3] = val[begin + threadIdx.y + 32 * 3];
    _val[threadIdx.y + 32 * 4] = val[begin + threadIdx.y + 32 * 4];
    

    _idx[threadIdx.y] = idx[begin + threadIdx.y];
    _idx[threadIdx.y + 32] = idx[begin + threadIdx.y + 32];
    _idx[threadIdx.y + 32 * 2] = idx[begin + threadIdx.y + 32 * 2];
    _idx[threadIdx.y + 32 * 3] = idx[begin + threadIdx.y + 32 * 3];
    _idx[threadIdx.y + 32 * 4] = idx[begin + threadIdx.y + 32 * 4];
    __syncthreads();

    for (int i = begin; i <= end; i += 1){
        result += __ldg(&vin[_idx[i - begin] * INFEATURE + j]) * _val[i - begin];
    }
    atomicAdd(&(vout[row[begin] * INFEATURE + j]), result);
}

__global__ void spmm_kernel_placeholder_128(int * row, int *ptr, int *idx, float *val, float *vin, float *vout, int * task, int task_width, int task_max, int * all_width, int num_v, int num_e, int INFEATURE, int * task_choice)
{
    int _tid = blockIdx.x * blockDim.x + threadIdx.x;
    // if (_tid >= task_width) return;
    int tid = task_choice[_tid];
    // int width = all_width[tid];
    int _begin = tid * task_max, _end = _begin + all_width[tid];
    // int begin = task[tid * task_max], end = tid + MAX_NE;
    int begin = task[_begin], end = task[_end-1];

    float result = 0.0f;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float _val[MAX_NE_3];
    __shared__ int _idx[MAX_NE_3];
    _val[threadIdx.y] = val[begin + threadIdx.y];
    _val[threadIdx.y + 32] = val[begin + threadIdx.y + 32];
    _val[threadIdx.y + 32 * 2] = val[begin + threadIdx.y + 32 * 2];
    _val[threadIdx.y + 32 * 3] = val[begin + threadIdx.y + 32 * 3];
    

    _idx[threadIdx.y] = idx[begin + threadIdx.y];
    _idx[threadIdx.y + 32] = idx[begin + threadIdx.y + 32];
    _idx[threadIdx.y + 32 * 2] = idx[begin + threadIdx.y + 32 * 2];
    _idx[threadIdx.y + 32 * 3] = idx[begin + threadIdx.y + 32 * 3];
    __syncthreads();

    for (int i = begin; i <= end; i += 1){
        result += __ldg(&vin[_idx[i - begin] * INFEATURE + j]) * _val[i - begin];
    }
    atomicAdd(&(vout[row[begin] * INFEATURE + j]), result);
}

__global__ void spmm_kernel_placeholder_2(int * row, int *ptr, int *idx, float *val, float *vin, float *vout, int * task, int task_width, int task_max, int * all_width, int num_v, int num_e, int INFEATURE, int * task_choice)
{
    int _tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_tid >= task_width) return;
    int tid = task_choice[_tid];
    int _begin = tid * task_max, _end = _begin + all_width[tid];
    int begin = task[_begin], end = task[_end-1];
    
    float result = 0.0f;
    float result_1 = 0.0f, result_2 = 0.0f, result_3 = 0.0f, result_4 = 0.0f , result_5 = 0.0f, result_6 = 0.0f, result_7 = 0.0f;
    int j = blockIdx.y * blockDim.y + threadIdx.y;


    __shared__ float _val[MAX_NE];
    __shared__ int _idx[MAX_NE];
    _val[threadIdx.y] = val[begin + threadIdx.y];
    _val[threadIdx.y + 32] = val[begin + 32 + threadIdx.y];
    _idx[threadIdx.y] = idx[begin + threadIdx.y];
    _idx[threadIdx.y + 32] = idx[begin + 32 + threadIdx.y];
    __syncthreads();

    for (int i = begin; i <= end; ++i){
        result += vin[_idx[i-begin] * INFEATURE + j] * _val[i - begin];
        result_1 += vin[_idx[i-begin] * INFEATURE + j + 32 * 1] * _val[i - begin];
        result_2 += vin[_idx[i-begin] * INFEATURE + j + 32 * 2] * _val[i - begin];
        result_3 += vin[_idx[i-begin] * INFEATURE + j + 32 * 3] * _val[i - begin];
        result_4 += vin[_idx[i-begin] * INFEATURE + j + 32 * 4] * _val[i - begin];
        result_5 += vin[_idx[i-begin] * INFEATURE + j + 32 * 5] * _val[i - begin];
        result_6 += vin[_idx[i-begin] * INFEATURE + j + 32 * 6] * _val[i - begin];
        result_7 += vin[_idx[i-begin] * INFEATURE + j + 32 * 7] * _val[i - begin];
    }
    atomicAdd(&(vout[row[begin] * INFEATURE + j]), result);
    atomicAdd(&(vout[row[begin] * INFEATURE + j + 32]), result_1);
    atomicAdd(&(vout[row[begin] * INFEATURE + j + 32 * 2]), result_2);
    atomicAdd(&(vout[row[begin] * INFEATURE + j + 32 * 3]), result_3);
    atomicAdd(&(vout[row[begin] * INFEATURE + j + 32 * 4]), result_4);
    atomicAdd(&(vout[row[begin] * INFEATURE + j + 32 * 5]), result_5);
    atomicAdd(&(vout[row[begin] * INFEATURE + j + 32 * 6]), result_6);
    atomicAdd(&(vout[row[begin] * INFEATURE + j + 32 * 7]), result_7);
}

std::string get_name(int num_e, int num_v){
    std::string name;
    if (num_v == 881680 && num_e == 5668682) name = "am";
    else if (num_v == 1569960) name = "amazon";
    else if (num_v == 169343) name = "arxiv";
    else if (num_v == 2927963) name = "citation";
    else if (num_v == 235868) name = "collab";
    else if (num_v == 4267 ) name = "ddi";
    else if (num_v == 576289) name = "ppa";
    else if (num_v == 2449029) name = "products";
    else if (num_v == 132534) name = "protein";
    else if (num_v == 232965) name = "reddit";
    else if (num_v == 2500604) name = "wikikg2";
    else if (num_v == 716847) name = "yelp";
    else if (num_v == 1138499) name = "youtube";
    else name = "newtask";
    std::cout<<"Task: "<<name<<std::endl;
    return name;
}

bool reorder_task(std::string name, int feat_in){
    if (feat_in == 32){
        std::vector<std::string> reorders = {"ddi", "ppa", "reddit", "products", "amazon"};
        if (std::find(reorders.begin(), reorders.end(), name) != reorders.end()){
            return true;
        }
        return false;
    } else {
        return true;
    }
}

void SpMMOpt::preprocess(float *vin, float *vout)
{
    std::string name = get_name(num_e, num_v);
    task_name = name;
    // copy ptr to cpu
    int * cpu_ptr = new int[num_v + 1];
    int * cpu_idx = new int[num_e];
    cudaMemcpy(cpu_ptr, d_ptr, sizeof(int) * (num_v + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_idx, d_idx, sizeof(int) * (num_e), cudaMemcpyDeviceToHost);

    // 收集每一行有多少非零元，以及有哪些非零元
    int * cpu_row = new int[num_e];
    int max = 0;
    std::vector<std::pair<int, int>> row_map;
    std::vector<std::vector<int>> row_item;
    for (int i = 0 ; i < num_v ; i++){
        std::vector<int> temp_vec;
        int start = cpu_ptr[i], end = cpu_ptr[i+1];
        if (end - start > max) max = end - start;
        row_map.push_back(std::make_pair(i, end - start));
        for (int j = start ; j < end; j++){
            cpu_row[j] = i;
            temp_vec.push_back(j);
        }
        row_item.push_back(temp_vec);
    }
    
    // 记录第i个value对应的行数
    cudaMalloc((void**)&d_row, sizeof(int) * num_e);
    cudaMemcpy(d_row, cpu_row, sizeof(int) * num_e, cudaMemcpyHostToDevice);

    // 按照每行非零元的个数重新排列
    std::sort(row_map.begin(), row_map.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
        return a.second > b.second;
    });


    // 把行的计算任务收集成Task, 每个Task只包含一行的MAX_NE个数据
    std::cout<<"max"<<max<<std::endl;
    if (feat_in == 32){
        if (name == "protein") max = MAX_NE_2;
        else if (name == "ddi" || name == "ppa" || name == "reddit" || name == "products" || name == "amazon") max = MAX_NE_3;
        else max = MAX_NE;
    } else {
        max = MAX_NE;
    }
    
    std::cout<<"num_e"<<num_e<<std::endl;
    std::vector<std::vector<int>> row_collect;
    std::vector<int> row_id;
    std::vector<int> temp;
    int MAX_ROW = 1;
    int count = 0;
    for (int k = 0 ; k < row_map.size() ; k++){
        auto row_info = row_map[k];
        // std::cout<<"row_info"<<row_info.first<<" "<<row_info.second<<std::endl;
        for (auto const & i : row_item[k]){
            temp.push_back(i);
            if (temp.size() == max){
                row_collect.push_back(temp);
                row_id.push_back(k);
                temp.clear();
            }
        }
        count ++;
        if (count == MAX_ROW){
            if (temp.size() != 0){
                row_collect.push_back(temp);
                row_id.push_back(k);
                temp.clear();
            }
            count = 0;
        }

        if (k == row_map.size()-1 && temp.size() > 0){
            std::cout<<"hello!"<<std::endl;
            row_collect.push_back(temp);
            row_id.push_back(k);
        }
    }
    int width = row_collect.size();
    std::cout<<"width"<<width<<std::endl;

    task_width = width;
    task_max = max;

    // 统计每个任务到底有多少个数据点要计算
    // task_index其实就是上边row_collect收集到数组里
    int * cpu_task_index = new int[width * max];
    int * cpu_all_width = new int[task_width];
    for (int k = 0 ; k < width ; k++){
        cpu_all_width[k] = row_collect[k].size();
        for (int i = 0; i < row_collect[k].size(); i++){
            cpu_task_index[k * max + i] = row_collect[k][i];
        }
    }
    cudaMalloc((void**)&all_width, sizeof(int) * width);
    cudaMalloc((void**)&task_index, sizeof(int) * width * max);
    cudaMemcpy(all_width, cpu_all_width, sizeof(int) * width, cudaMemcpyHostToDevice);
    cudaMemcpy(task_index, cpu_task_index, sizeof(int) * width * max, cudaMemcpyHostToDevice);

    // 现在考虑对任务进行重排
    std::vector<std::tuple<int, int, int, int>> task_first_nz;
    for (int k = 0; k < width; k++){
        int min = num_v;
        auto all_cols = row_collect[k];
        for (int j = 0 ; j < all_cols.size(); j++){
            if (cpu_idx[all_cols[j]] < min) min = cpu_idx[all_cols[j]];
        }
        task_first_nz.push_back(std::make_tuple(k, cpu_all_width[k], min, int(row_collect[row_id[k]].size())));
    }

    if (reorder_task(name, feat_in)){
        std::sort(task_first_nz.begin(), task_first_nz.end(), [](const std::tuple<int, int, int, int>& a, const std::tuple<int, int, int, int>& b) {
            return std::get<2>(a) < std::get<2>(b);
        });
    } else {
        std::sort(task_first_nz.begin(), task_first_nz.end(), [](const std::tuple<int, int, int, int>& a, const std::tuple<int, int, int, int>& b) {
            if (std::get<3>(a) == std::get<3>(b)){
                return std::get<2>(a) < std::get<2>(b);
            }
            else{
                return std::get<3>(a) > std::get<3>(b);
            }
        });
    }

    int task_id[width];
    if (reorder_task(name, feat_in)){
        for (int k = 0; k < width; k++){
            // std::cout<<std::get<0>(task_first_nz[k])<<" "<<std::get<2>(task_first_nz[k])<<" "<<std::get<3>(task_first_nz[k])<<std::endl;
            task_id[k] = std::get<0>(task_first_nz[k]);
        }
        std::cout << std::endl;
    } else {
        for (int k = 0; k < width; k++){
           task_id[k] = k;
        }
    }
    

    std::cout<<"hello2"<<std::endl;
    cudaMalloc((void**)&task_choice, sizeof(int) * width);
    cudaMemcpy(task_choice, task_id, sizeof(int) * width, cudaMemcpyHostToDevice);

    // TODO: your code
    std::cout<<feat_in<<" "<<num_v<<" "<<std::endl;

    grid.x = width;
    block.x = BLOCK_SIZE;
    block.y = BLOCK_SIZE_2;
    grid.y = (32 + BLOCK_SIZE_2 - 1) / BLOCK_SIZE_2;
    // std::cout<<"grid.x"<<grid.x<<std::endl;
}

void SpMMOpt::run(float *vin, float *vout)
{
    if (feat_in == 32){
        if (task_name == "protein"){
            spmm_kernel_placeholder_160<<<grid, block>>>(d_row, d_ptr, d_idx, d_val, vin, vout, task_index, task_width, task_max, all_width, num_v, num_e, feat_in, task_choice);
        } else if (task_name == "ddi" || task_name == "ppa" || task_name == "reddit" || task_name == "products" || task_name == "amazon"){
            spmm_kernel_placeholder_128<<<grid, block>>>(d_row, d_ptr, d_idx, d_val, vin, vout, task_index, task_width, task_max, all_width, num_v, num_e, feat_in, task_choice);
        } else {
            spmm_kernel_placeholder<<<grid, block>>>(d_row, d_ptr, d_idx, d_val, vin, vout, task_index, task_width, task_max, all_width, num_v, num_e, feat_in, task_choice);
        }
    } else {
        spmm_kernel_placeholder_2<<<grid, block>>>(d_row, d_ptr, d_idx, d_val, vin, vout, task_index, task_width, task_max, all_width, num_v, num_e, feat_in, task_choice);
    }
}