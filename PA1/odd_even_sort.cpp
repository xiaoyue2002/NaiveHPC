#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <cstring>
#include <iostream>

#include "worker.h"

int right_position(float * right, int block_len, float bar){
  for (int i = 0 ; i < block_len ; i++){
    if (right[i] > bar) return i;
  }
  return block_len;
}

int left_position(float * left, int block_len, float bar){
  for (int i = block_len - 1; i >=0 ; i--){
    if (left[i] < bar) return i+1;
  }
  return block_len;
}

void my_merge_right(float * right, float * left, float * buffer, int block_len, int left_block_len){
  int p1 = 0, p2 = 0;
  for (int i = 0 ; i < block_len ; i++){
    if (p1 == block_len) buffer[i] = left[p2++];
    else if (p2 == left_block_len) buffer[i] = right[p1++];
    else buffer[i] = right[p1] < left[p2] ? right[p1++] : left[p2++];
  }
}

void my_merge_left(float * right, float * left, float * buffer, int block_len, int left_block_len){
  int p1 = block_len - 1, p2 = left_block_len - 1;
  for (int i = block_len - 1; i >= 0 ; i--){
    if (p1 < 0) buffer[i] = left[p2--];
    else if (p2 < 0) buffer[i] = right[p1--];
    else buffer[i] = right[p1] > left[p2] ? right[p1--] : left[p2--];
  }
}

void Worker::sort() {
  /** Your code ... */
  // you can use variables in class Worker: n, nprocs, rank, block_len, data

  // 处理不能整除的情况和某些process用不上的情况
  int std_block_len = (n + nprocs - 1) / nprocs;
  int last_rank = (n + std_block_len - 1) / std_block_len - 1;
  int last_block_len = n - std_block_len * last_rank;
  
  // local sorts
  std::sort(data, data + block_len);

  if (nprocs <= 1) return ; 
  if (rank > last_rank) return;

  int global_sorted = last_rank + 1;
  float local_min, local_max;
  float sent, received;
  float * rcv_data = new float [std_block_len];
  float * buffer = new float [block_len];

  MPI_Request request[2];

  // Even / Odd Phase Flag
  int odd_even_phase = 0;
  int verse_odd_even = 1;

  // 提前发送端点数据，和后面的while一致
  if ((rank % 2 == odd_even_phase && rank < last_rank) || (rank % 2 != odd_even_phase && rank > 0)){
    int neighbor = rank + (rank % 2 == odd_even_phase ? 1 : -1);
    sent = rank % 2 == odd_even_phase ? data[block_len - 1] : data[0];
    MPI_Isend(&sent, 1, MPI_FLOAT, neighbor, 0, MPI_COMM_WORLD, &request[0]);
    MPI_Irecv(&received, 1, MPI_FLOAT, neighbor, 0, MPI_COMM_WORLD, &request[1]);
  }
  
  while (global_sorted > 0){
    // Even / Odd phase
    if ((rank % 2 == odd_even_phase && rank < last_rank) || (rank % 2 != odd_even_phase && rank > 0)) {
      int neighbor = rank + (rank % 2 == odd_even_phase ? 1 : -1);

      local_max = data[block_len - 1];
      local_min = data[0];

      int neighbor_block_len = neighbor == last_rank ? last_block_len : std_block_len;
      
      MPI_Waitall(2, request, MPI_STATUSES_IGNORE);

      if (rank % 2 == odd_even_phase) { // Right neighbor must be smaller
        if (local_max > received) {
            MPI_Isend(data, block_len, MPI_FLOAT, neighbor, 0, MPI_COMM_WORLD, &request[0]);
            MPI_Irecv(rcv_data, neighbor_block_len, MPI_FLOAT, neighbor, 0, MPI_COMM_WORLD, &request[1]);
            
            int start_pos = right_position(data, block_len, received);
            int merge_len = block_len - start_pos;
            MPI_Waitall(2, request, MPI_STATUSES_IGNORE);

            // 如果参与下一轮计算, 把端点先发过去
            if (global_sorted > 1 && ((rank % 2 == verse_odd_even && rank < last_rank) || (rank % 2 != verse_odd_even && rank > 0))){
              int next_neighbor = rank + (rank % 2 == verse_odd_even ? 1 : -1);
              sent = std::min(received, local_min);
              MPI_Isend(&sent, 1, MPI_FLOAT, next_neighbor, 0, MPI_COMM_WORLD, &request[0]);
              MPI_Irecv(&received, 1, MPI_FLOAT, next_neighbor, 0, MPI_COMM_WORLD, &request[1]);
            }

            my_merge_right(data + start_pos, rcv_data, buffer, merge_len, neighbor_block_len);
            std::copy(buffer, buffer + merge_len, data + start_pos);
        } else {
          if (global_sorted > 1 && ((rank % 2 == verse_odd_even && rank < last_rank) || (rank % 2 != verse_odd_even && rank > 0))){
              int next_neighbor = rank + (rank % 2 == verse_odd_even ? 1 : -1);
              sent = local_min;
              MPI_Isend(&sent, 1, MPI_FLOAT, next_neighbor, 0, MPI_COMM_WORLD, &request[0]);
              MPI_Irecv(&received, 1, MPI_FLOAT, next_neighbor, 0, MPI_COMM_WORLD, &request[1]);
            }
        }
      } else { // Left neighbor must be larger
          if (received > local_min) {
            MPI_Isend(data, block_len, MPI_FLOAT, neighbor, 0, MPI_COMM_WORLD, &request[0]);
            MPI_Irecv(rcv_data, neighbor_block_len, MPI_FLOAT, neighbor, 0, MPI_COMM_WORLD, &request[1]);
            
            int merge_len = left_position(data, block_len, received);
            MPI_Waitall(2, request, MPI_STATUSES_IGNORE);

            // 如果参与下一轮计算, 把端点先发过去
            if (global_sorted > 1 && ((rank % 2 == verse_odd_even && rank < last_rank) || (rank % 2 != verse_odd_even && rank > 0))){
              int next_neighbor = rank + (rank % 2 == verse_odd_even ? 1 : -1);
              sent = std::max(received, local_max);
              MPI_Isend(&sent, 1, MPI_FLOAT, next_neighbor, 0, MPI_COMM_WORLD, &request[0]);
              MPI_Irecv(&received, 1, MPI_FLOAT, next_neighbor, 0, MPI_COMM_WORLD, &request[1]);
            }

            my_merge_left(data, rcv_data, buffer, merge_len, neighbor_block_len);
            std::copy(buffer, buffer + merge_len, data);
        } else {
          if (global_sorted > 1 && ((rank % 2 == verse_odd_even && rank < last_rank) || (rank % 2 != verse_odd_even && rank > 0))){
              int next_neighbor = rank + (rank % 2 == verse_odd_even ? 1 : -1);
              sent = local_max;
              MPI_Isend(&sent, 1, MPI_FLOAT, next_neighbor, 0, MPI_COMM_WORLD, &request[0]);
              MPI_Irecv(&received, 1, MPI_FLOAT, next_neighbor, 0, MPI_COMM_WORLD, &request[1]);
          }
        }
      }
    } else {
      if (global_sorted > 1 && ((rank % 2 == verse_odd_even && rank < last_rank) || (rank % 2 != verse_odd_even && rank > 0))){
        int next_neighbor = rank + (rank % 2 == verse_odd_even ? 1 : -1);
        sent = rank % 2 == verse_odd_even ? data[block_len - 1] : data[0];
        MPI_Isend(&sent, 1, MPI_FLOAT, next_neighbor, 0, MPI_COMM_WORLD, &request[0]);
        MPI_Irecv(&received, 1, MPI_FLOAT, next_neighbor, 0, MPI_COMM_WORLD, &request[1]);
      }
    }

    verse_odd_even = verse_odd_even == 0? 1 : 0;
    odd_even_phase = odd_even_phase == 0? 1 : 0;
    global_sorted -- ;
  }

  delete[] rcv_data;
  delete[] buffer;
}
