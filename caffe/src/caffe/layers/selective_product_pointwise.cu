#include "caffe/layers/selective_product_pointwise.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe { 

template <typename Dtype>
__global__ void SelectiveProductPointwiseForward(const int nthreads, 
   const Dtype* _param_data, const Dtype* _outer_data,
   const int* _input, const int* _param, 
   const Dtype* _input_data, const Dtype* _bias_data,
   Dtype* output,
   int I_, int N_, int T_, int D_, int C_, bool dobias){
   CUDA_KERNEL_LOOP(index, nthreads){
     //index = b * T_ + k;
     int k = index % T_;
     int b = index / T_;
 
     int i = _input[k];
     int p = _param[k];
     int i_offset = b*I_*C_*D_ + i*C_*D_;
     int p_offset = p*D_;
     Dtype total = 0;
     for (int c = 0; c < C_; c++){
       for(int d1 = 0; d1 < D_; d1++){
         Dtype v1 = _input_data[i_offset + c*D_ + d1];
         Dtype v2 = _param_data[p_offset + d1];
         Dtype w = _outer_data[i*D_ + d1];
	 total += w * v1 * v2;
       }
     }
     output[index] = total + (dobias ? _bias_data[index] : 0);
   }
} 

 template <typename Dtype>
__global__ void SelectiveProductPointwiseInputBackward(const int nthreads, 
   const Dtype* _param_data, const Dtype* _input_data, const Dtype* _outer_data,  const Dtype* _outputGrad,
   const int* _input, const int* _param, const int* index, const int* offsets, const int* lengths,  
   Dtype* _inputGrad, 
   int I_, int N_, int T_, int D_, int C_){
   CUDA_KERNEL_LOOP(iindex, nthreads){
     //index = b*I_ + i
     int i = iindex % I_;
     int b = iindex / I_;
     
     //loop through all output indexes that have this input
     for(int c = 0; c < C_ ; c++){
       for(int d = 0; d < D_; d++){
         int begin = offsets[i];
         int end = begin + lengths[i];
        
 	 Dtype w = _outer_data[i*D_ + d];
         int i_offset = b*I_*C_*D_ + i*C_*D_ + c*D_ + d;
         Dtype total = 0.f;	
         for(int x = begin; x < end; x++){
           int k = index[x];
           Dtype v2 = _param_data[D_*_param[k] + d];
           Dtype outgrad = _outputGrad[b*T_ + k];
           total += v2 * outgrad;
         }
         _inputGrad[i_offset] = total * w;
       }
     } 
   }
}

template <typename Dtype>
__global__ void SelectiveProductPointwiseWieghtBackward(const int nthreads, 
   const Dtype* _param_data, const Dtype* _input_data, const Dtype* _outer_data, const Dtype* _outputGrad,
   const int* _input, const int* _param, const int* index, const int* offsets, const int* lengths,  
   Dtype* _weightGrad, 
   int I_, int N_, int T_, int D_, int C_, int B_){
   CUDA_KERNEL_LOOP(iindex, nthreads){
     //index = D_*i + d
     int d = iindex % D_;
     int i = iindex / D_;
     
     //loop through all output indexes that have this input
     int begin = offsets[i];
     int end = begin + lengths[i];
     Dtype total = 0.f;
     for(int b = 0; b < B_ ; b++){
       for(int x = begin; x < end; x++){
         int k = index[x];
	 int p = _param[k];
          
         Dtype outgrad = _outputGrad[b*T_ + k];
         Dtype v2 = _param_data[p*D_ + d];
	 int input_offset = b*I_*D_*C_ + i*D_*C_ + d;
	 Dtype w = v2 * outgrad;
         for( int c = 0; c < C_; c++){
           Dtype d1 = _input_data[input_offset + c*D_];
           total += d1 * w;
         }
       }
     }
     //this should be a sum
     _weightGrad[iindex] = total;
   }
}

 template <typename Dtype>
__global__ void SelectiveProductPointwiseParamBackward(const int nthreads, 
   const Dtype* _param_data, const Dtype* _input_data, const Dtype* _outer, const Dtype* _outputGrad,
   const int* _input, const int* _param, const int* index, const int* offsets, const int* lengths,  
   Dtype* _paramGrad, 
   int I_, int N_, int T_, int D_, int C_, int B_){
   CUDA_KERNEL_LOOP(iindex, nthreads){
     //index = D_*p + d
     int d = iindex % D_;
     int p = iindex / D_;
     
     //loop through all output indexes that have this input
     int begin = offsets[p];
     int end = begin + lengths[p];
     Dtype total = 0.f;
     for(int b = 0; b < B_ ; b++){
       for(int x = begin; x < end; x++){
         int k = index[x];
	 int i = _input[k];
          
         Dtype outgrad = _outputGrad[b*T_ + k];
	 int input_offset = b*I_*D_*C_ + i*D_*C_ + d;
	 Dtype w = _outer[i*D_ + d] * outgrad;
         for( int c = 0; c < C_; c++){
           Dtype d1 = _input_data[input_offset + c*D_];
           total += d1 * w;
         }
       }
     }
     //this should be a sum
     _paramGrad[iindex] = total;
   }
}

template <typename Dtype>
void SelectiveProductPointwiseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*> & top){
  LOG(INFO) << "Selective Forward"; 
  const Dtype* w = this->blobs_[w_index_]->gpu_data(); 
  const Dtype* o = this->blobs_[o_index_]->gpu_data(); 
 
  const Dtype* input_data = bottom[0]->gpu_data();
  const Dtype* bias_data;
  if ( bottom.size() > 1 ) bias_data = bottom[1]->gpu_data();

  const int* input = this->i_input_.gpu_data();
  const int* param = this->i_param_.gpu_data();

  Dtype* output = top[0]->mutable_gpu_data();
  
  int batch_size = bottom[0]->shape(0);
  int jobs = batch_size *  T_ ;
  
  SelectiveProductPointwiseForward<Dtype><<<CAFFE_GET_BLOCKS(jobs), CAFFE_CUDA_NUM_THREADS>>>( jobs, 
    w, o, 
    input, param, 
    input_data, bias_data, 
    output, 
    I_, N_, T_, D_, C_, bottom.size() > 1);

  CUDA_POST_KERNEL_CHECK;
 /* 
  int index = 12;
  int batch = 11;
  const Dtype* save = top[0]->cpu_data();
  Dtype t = save[index + T_ * batch]; 
  this->Forward_cpu(bottom, top);
  const Dtype* save2 =  top[0]->cpu_data();
  LOG(INFO) << "CPU DIFF : " << t - save2[index + T_ * batch] << " " << t; 
*/
  LOG(INFO) << "Selective Forward Done";
}

template <typename Dtype>
void SelectiveProductPointwiseLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  const Dtype* w = this->blobs_[w_index_]->gpu_data(); 
  const Dtype* o = this->blobs_[o_index_]->gpu_data(); 
 
  const Dtype* input_data = bottom[0]->gpu_data();
  //const Dtype* bias_data;
  //if ( bottom.size() > 1 ) bias_data = bottom[1]->gpu_data();

  const Dtype* topGrad = top[0]->gpu_diff();   
  Dtype* inputGrad = bottom[0]->mutable_gpu_diff();
  Dtype* biasGrad;
  if( bottom.size() > 1 ) biasGrad = bottom[1]->mutable_gpu_diff();
  Dtype* paramGrad = this->blobs_[w_index_]->mutable_gpu_diff();
  Dtype* outerGrad = this->blobs_[o_index_]->mutable_gpu_diff();

  const int* input = this->i_input_.gpu_data();
  const int* param = this->i_param_.gpu_data();
  const int* input_index = i_input_index_.gpu_data(); 
  const int* input_length = i_input_length_index_.gpu_data();
  const int* input_offset = i_input_offset_index_.gpu_data();
  const int* param_index = i_param_index_.gpu_data();
  const int* param_length = i_param_length_index_.gpu_data();
  const int* param_offset = i_param_offset_index_.gpu_data();
 
  const int* param_index_cpu = i_param_index_.cpu_data();
  const int* param_cpu = i_param_.cpu_data();
  const int* input_cpu = i_input_.cpu_data();
//  for(int i = 0; i < 20; i++) LOG(INFO) << param_index_cpu[i] << " " << param_cpu[param_index_cpu[i]] << " " << input_cpu[param_index_cpu[i]];
 
  int batch_size = bottom[0]->shape(0);
  
  LOG(INFO) << "input backward"; 
  int ijobs = batch_size*I_;
  SelectiveProductPointwiseInputBackward<Dtype><<<CAFFE_GET_BLOCKS(ijobs), CAFFE_CUDA_NUM_THREADS>>>( ijobs, 
    w, input_data, o, topGrad, 
    input, param, input_index, input_offset, input_length,  
    inputGrad,
    I_, N_, T_, D_, C_); 

  CUDA_POST_KERNEL_CHECK;

  LOG(INFO) << "param backward";
  int pjobs = N_ * D_;
  SelectiveProductPointwiseParamBackward<Dtype><<<CAFFE_GET_BLOCKS(pjobs), CAFFE_CUDA_NUM_THREADS>>>( pjobs, 
    w, input_data, o, topGrad, 
    input, param, param_index, param_offset, param_length,  
    paramGrad,
    I_, N_, T_, D_, C_, batch_size); 

  CUDA_POST_KERNEL_CHECK;

  if(this->scalar_){
  LOG(INFO) << "w backward";
  int wjobs = I_ * D_;
  SelectiveProductPointwiseWieghtBackward<Dtype><<<CAFFE_GET_BLOCKS(wjobs), CAFFE_CUDA_NUM_THREADS>>>( wjobs, 
    w, input_data, o, topGrad, 
    input, param, input_index, input_offset, input_length,  
    outerGrad,
    I_, N_, T_, D_, C_, batch_size); 

  CUDA_POST_KERNEL_CHECK;
  }
  else{
    LOG(INFO) << "IGNORING SCALAR";
  }
  LOG(INFO) << "bias copy " << T_ << " " << batch_size;
  //do the bias grad
  if(bottom.size() > 1){
    //its just a copy of the top
    caffe_copy(T_*batch_size, topGrad, bottom[1]->mutable_gpu_diff());
  }

  //LOG(INFO) << "bias copy done.";
/*  int testindex = 10000;
  int batch = 11;
  const Dtype* paramGradCPU = this->blobs_[w_index_]->cpu_diff(); 
  const Dtype* inputGradCPU = bottom[0]->cpu_diff();
  const Dtype* inputBiasCPU = bottom[1]->cpu_diff();
  const Dtype t1 = paramGradCPU[testindex];
  const Dtype t2 = inputGradCPU[testindex + I_*C_*D_*batch];
  const Dtype t3 = inputBiasCPU[testindex + T_*batch];
  this->Backward_cpu(top, propagate_down, bottom); 
  const Dtype* paramGradCPU2 = this->blobs_[w_index_]->cpu_diff(); 
  const Dtype* inputGradCPU2 = bottom[0]->cpu_diff();
  const Dtype* inputBiasCPU2 = bottom[1]->cpu_diff();
  LOG(INFO) << " grad param diff" << t1 - paramGradCPU2[testindex]<< " grad input diff " << t2 - inputGradCPU[testindex+ I_*C_*D_*batch] << " bias input diff " << t3 - inputBiasCPU2[testindex + T_*batch];
;
*/
}
  
INSTANTIATE_LAYER_GPU_FUNCS(SelectiveProductPointwiseLayer);
//namespace
} 
