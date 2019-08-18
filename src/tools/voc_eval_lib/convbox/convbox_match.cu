#include "convbox_match.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#include <cfloat>

// TODO make it in a common file
#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

template <typename T>
__device__ T overlap(T x1, T w1, T x2, T w2){
  T l1 = x1 - w1 / 2;
  T l2 = x2 - w2 / 2;
  T left = l1 > l2 ? l1 : l2;
  T r1 = x1 + w1 / 2;
  T r2 = x2 + w2 / 2;
  T right = r1 < r2 ? r1 : r2;
  return right - left;
}

template <typename T>
__device__ T box_intersection(const T* a, const T* b){
  T w = overlap(a[0], a[2], b[0], b[2]);
  T h = overlap(a[1], a[3], b[1], b[3]);
  if (w < 0 || h < 0) return 0;
  T area = w * h;
  return area;
}

template <typename T>
__device__ T box_iou(const T* a, const T* b){
    return box_intersection(a, b) / box_union(a, b);
}

template <typename T>
__device__ T box_rmse(const T* a, const T* b){
  return sqrt(pow(a[0] - b[0], 2) + 
              pow(a[1] - b[1], 2) + 
              pow(a[2] - b[2], 2) + 
              pow(a[3] - b[3], 2));
}

template <typename T>
__device__ T box_union(const T* a, const T* b){
    T i = box_intersection(a, b);
    T u = a[2] * a[3] + b[2] * b[3] - i;
    return u;
}

template <typename T>
__global__ void ConvboxMatchForward(int nthreads, const T* pred, const T* label, const int num_box, const int num_class, const int batch_size, const int rows, const int cols, const int channels, const T* pred_mask_output, const T* label_mask_output, const T* update_label_output) {
    CUDA_1D_KERNEL_LOOP(i, nthreads) {
/*
        T avg_dist = 0;
        T avg_cat = 0;
        T avg_allcat = 0;
        T avg_conn = 0;
        T avg_allconn = 0;
        T avg_obj = 0;
        T avg_anyobj = 0;
        int count = 0;

      // assert(rows == 14);
      // assert(cols == 14);
      for (int batch = 0; batch < batch_size; ++batch){
        for (int rowm = 0; rowm < rows; ++rowm){
          for (int colm = 0; colm < cols; ++colm){
            for (int c = 0; c < channels; ++c){
              for (int i = 0; i < num_box; ++i)
                pred_mask_output(batch, rowm, colm, c*num_box+i) = false;
              label_mask_output(batch, rowm, colm, c) = false;
              update_label_output(batch, rowm, colm, c) = label(batch, rowm, colm, c);
            }
          }
        }
          
        for (int rowm = 0; rowm < rows; ++rowm){
          for (int colm = 0; colm < cols; ++colm){
                        
            for (int i = 0; i < 2*num_box; ++i)
              avg_anyobj += pred(batch, rowm, colm, i);
            
            if (label(batch, rowm, colm, 0)){
              // match by overlap
              int best_index = -1;
              int rowc = -1, colc = -1;
              for (int i = 0; i < rows; ++i){
                if (label(batch, rowm, colm, 6+i) == 1){
                  rowc = i; break;
                }
              }
              for (int i = 0; i < cols; ++i){
                if (label(batch, rowm, colm, 6+rows+i) == 1){
                  colc = i; break;
                }
              }
              
              T xm = (label(batch, rowm, colm, 2) + colm) / cols;
              T ym = (label(batch, rowm, colm, 3) + rowm) / rows;
              T xc = (label(batch, rowc, colc, 4) + colc) / cols;
              T yc = (label(batch, rowc, colc, 5) + rowc) / rows;
              T* truth;
              truth[0] = xm; truth[1] = ym;
              truth[2] = fabs(xm - xc) * 2.0;
              truth[3] = fabs(ym - yc) * 2.0;
      
              T best_match = -100;
              for (int i = 0; i < num_box; ++i){
                
                T pred_cat = 0.0;
                for (int j = 0; j < num_class; ++j){
                  T  pred_cat_m =  pred(batch, rowm, colm, (6+2*(rows+cols))*num_box+i*num_class+j);
                  T  pred_cat_c =  pred(batch, rowc, colc, (6+2*(rows+cols)+num_class)*num_box+i*num_class+j);
                  T label_cat_m = label(batch, rowm, colm, 6+2*(rows+cols)+j);
                  T label_cat_c = label(batch, rowc, colc, 6+2*(rows+cols)+num_class+j);
                  pred_cat += pow(pred_cat_m - label_cat_m, 2);
                  pred_cat += pow(pred_cat_c - label_cat_c, 2); 
                }

                T pred_conn_m = pred(batch, rowm, colm, 6*num_box+i*(rows+cols)+rowc)
                              * pred(batch, rowm, colm, 6*num_box+i*(rows+cols)+rows+colc);
                T pred_conn_c = pred(batch, rowc, colc, (6+rows+cols)*num_box+i*(rows+cols)+rowm)
                              * pred(batch, rowc, colc, (6+rows+cols)*num_box+i*(rows+cols)+rows+colm);
                T pred_conn = (pred_conn_m + pred_conn_c) / 2.0;
                
                xm = (pred(batch, rowm, colm, (num_box+i)*2+0) + colm) / cols;
                ym = (pred(batch, rowm, colm, (num_box+i)*2+1) + rowm) / rows;
                xc = (pred(batch, rowc, colc, (num_box*2+i)*2+0) + colc) / cols;
                yc = (pred(batch, rowc, colc, (num_box*2+i)*2+1) + rowc) / rows;
                T out[4] = {};
                out[0] = xm; out[1] = ym;
                out[2] = fabs(xm - xc) * 2.0;
                out[3] = fabs(ym - yc) * 2.0;
                T iou = box_iou(out, truth);
                T rmse = box_rmse(out, truth);
                
                T cur_match = pred_conn * (iou - rmse + 0.1) + 0.1 * (2 - pred_cat);
                
                if (cur_match > best_match){
                  best_match = cur_match;
                  best_index = i;
                }
              }

              assert(best_index != -1);
              assert(rowc != -1);
              assert(colc != -1);
              
              int row[2] = {rowm, rowc};
              int col[2] = {colm, colc};
              for (int n = 0; n < 2; ++n) {
                pred_mask_output(batch, row[n], col[n], n*num_box+best_index) = true;
                label_mask_output(batch, row[n], col[n], n) = true;
                avg_obj += pred(batch, row[n], col[n], n*num_box+best_index);
                
                pred_mask_output(batch, row[n], col[n], ((1+n)*num_box+best_index)*2+0) = true;
                pred_mask_output(batch, row[n], col[n], ((1+n)*num_box+best_index)*2+1) = true;
                label_mask_output(batch, row[n], col[n], 2*(1+n)+0) = true;
                label_mask_output(batch, row[n], col[n], 2*(1+n)+1) = true;
                avg_dist += (pow(pred(batch, row[n], col[n], ((1+n)*num_box+best_index)*2+0) - label(batch, row[n], col[n], 2*(1+n)+0), 2)
                           + pow(pred(batch, row[n], col[n], ((1+n)*num_box+best_index)*2+1) - label(batch, row[n], col[n], 2*(1+n)+1), 2));
                
                for (int i = 0; i < (rows+cols); ++i){
                  pred_mask_output(batch, row[n], col[n], (6+n*(rows+cols))*num_box+best_index*(rows+cols)+i) = true;
                  label_mask_output(batch, row[n], col[n], 6+n*(rows+cols)+i) = true;
                  if (label(batch, row[n], col[n], 6+n*(rows+cols)+i) == 1)
                    avg_conn += pred(batch, row[n], col[n], (6+n*(rows+cols))*num_box+best_index*(rows+cols)+i);
                  avg_allconn += pred(batch, row[n], col[n], (6+n*(rows+cols))*num_box+best_index*(rows+cols)+i);
                }
                
                for (int i = 0; i < num_class; ++i){
                  pred_mask_output(batch, row[n], col[n], (6+2*(rows+cols)+n*num_class)*num_box+best_index*num_class+i) = true;
                  label_mask_output(batch, row[n], col[n], 6+2*(rows+cols)+n*num_class+i) = true;
                  if (label(batch, row[n], col[n], 6+2*(rows+cols)+n*num_class+i) == 1)
                    avg_cat += pred(batch, row[n], col[n], (6+2*(rows+cols)+n*num_class)*num_box+best_index*num_class+i);
                  avg_allcat += pred(batch, row[n], col[n], (6+2*(rows+cols)+n*num_class)*num_box+best_index*num_class+i);
                }
                count += 1;
              }
            } // has obj
          } // colm
        } // rowm
      } // batch
   */
 } // CUDA_1D_KERNEL_LOOP
} // ConvboxMatchForward


template <typename T>
__global__ void ConvboxMatchBackward( const int nthreads, T* d_logits) {
  CUDA_1D_KERNEL_LOOP(i, nthreads) {
    d_logits[i] = 0.0;
  } // CUDA_1D_KERNEL_LOOP
} // ConvboxMatchBackward


std::tuple<at::Tensor, at::Tensor, at::Tensor> ConvboxMatch_forward_cuda(
		const at::Tensor& pred,
                const at::Tensor& label,
                const int num_box,
		const int num_class,
                const int batch_size,
                const int rows,
                const int cols,
		const int channels){
  AT_ASSERTM(pred.type().is_cuda(), "pred must be a CUDA tensor");
  AT_ASSERTM(label.type().is_cuda(), "label must be a CUDA tensor");

  auto pred_mask_output = at::empty({batch_size, rows, cols, channels * num_box}, pred.options());
  auto label_mask_output = at::empty({batch_size, rows, cols, channels}, pred.options());
  auto update_label_output = at::empty({batch_size, rows, cols, channels}, pred.options());
  auto output_size = batch_size * rows * cols * channels;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(THCCeilDiv((long)output_size, 512L), 4096L));

  dim3 block(512);

  //if (losses.numel() == 0) {
  //  THCudaCheck(cudaGetLastError());
  //  return losses;
  //}

  
  AT_DISPATCH_FLOATING_TYPES(pred.type(), "ConvboxMatch_forward", [&] {
    ConvboxMatchForward<scalar_t><<<grid, block, 0, stream>>>(
         output_size,
         pred.contiguous().data<scalar_t>(),
         label.contiguous().data<scalar_t>(),
	 num_box,
	 num_class,
	 batch_size,
	 rows,
	 cols,
	 channels,
         pred_mask_output.data<scalar_t>(),
         label_mask_output.data<scalar_t>(),
         update_label_output.data<scalar_t>());
  }); 

  THCudaCheck(cudaGetLastError());
  return std::make_tuple(pred_mask_output, label_mask_output, label_mask_output);
}



at::Tensor ConvboxMatch_backward_cuda() {
  auto d_logits = at::zeros({1});
  return d_logits;
}


