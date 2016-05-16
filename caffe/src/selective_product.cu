template <typename Dtype>
__global__ void SelectiveProductForward(const int nthreads,
   const Dtype* param, const Dtype* input, const Dtype* scalar, const Dtype* s_bias, const Dtype* d_bias, Dtype* product_sratch, Dtype* output, 
   const int* o_param, const int* o_input, const int* o_index,
   int T_, int N_, int D_, int I_,  int max_o, bool static_bias, bool dynamic_bias, bool scalar){
   CUDA_KERNEL_LOOP(index, nthreads) {
     //index = b*T_ + o
     int b = index / T_;
     int o = index % T_;
     int offset = b*T_*max_o + o*max_o;
     Dtype rv = 0.;

     if(static_bias) rv += s_bias[o];
     if(dynamic_bias) rv += d_bias[index];
     
     for(int l = 0; l < max_items; l++){
       int i = o_input[offset + l];
       int p = o_param[offset + l];
       int k = o_index[offset + l];
       if( i == -1) return rv;
       int input_offset = b*D_*I_ + D_*i;
       int param_offset = D_*p;
       Dtype tot = 0.f;
       for(int d = 0; d < D_; d++) tot += param[param_offset+d] * input[input_offset+d];
       if(scalar) tot = tot * scalar[k]; 
       //store this for the backward
       product_scratch[offset + l] = tot;
       
       rv += tot;
     }
     return rv;
   }
} 

template <typename Dtype>
__global__ void SelectiveProductBackwardParameter(const int nthreads,
   const Dtype* param, const Dtype* input, const Dtype* scalar, const Dtype* top_grad,  Dtype* p_grad, 
   const int* p_output, const int* p_input, const int* p_index,
   int B_, int T_, int N_, int D_, int I_, int max_p, bool static_bias, bool dynamic_bias, bool scalar){
   CUDA_KERNEL_LOOP(index, nthreads) {
     int p = index;
     int p_offset = p*max_p;
     int grad_offset = D_*p;

     //we need to be paralell over batch
     for(int d = 0; d < D_; d++){
       Dtype grad = 0.f;
       for(int b = 0; b < B_; b++){
         int input_offset = b*I_*D_;
         int top_offset = b*T_;
         for(int l = 0; l < max_p; l++){
           int o = p_output[p_offset+l];
           if(o  != -1){
             int i = p_input[p_offset+l];
             int k = p_index[p_offset+l];
             //the output is scalar 
             grad += top_grad[top_offset + o]*input[input_offset + i*D_]*scalar[k];
           }
         }
       }
       p_grad[grad_offset + d] = grad;      
     }
  }
}

template <typename Dtype>
__global__ void SelectiveProductBackwardInput(const int nthreads,
   const Dtype* param, const Dtype* input, const Dtype* scalar, Dtype* top_grad, Dtype* i_grad, 
   const int* i_param, const int* i_output, const int* i_index, 
   int B_ , int T_, int N_, int D_, int I_,  int max_i, bool static_bias, bool dynamic_bias, bool scalar){
   CUDA_KERNEL_LOOP(index, nthreads) {
     //index = b*I_ + i
     int b = index / I_;
     int i = index % I_;
     int i_offset = i*max_i;
     int input_offset = b*I_*D_ + i*D_;
 
     int top_offset = b*T_;

     //we need to be paralell over batch
     for(int d = 0; d < D_; d++){
       Dtype grad = 0.f;
       for(int l = 0; l < max_i; l++){
         int o = i_output[i_offset+l];
         if(o  != -1){
           int p = i_input[i_offset+l];
           int k = i_index[i_offset+l];
           grad += top_grad[top_offset + o]*param[p*D_ + d]*scalar[k];
         }
       }
       p_grad[grad_offset + d] = grad;      
     }
  }
}

template <typename Dtype>
__global__ void SelectiveProductBackwardOutput(const int nthreads,
   const Dtype* param, const Dtype* input, const Dtype* scalar, const Dtype* s_bias, const Dtype* d_bias, Dtype* output, 
   const int* o_param, const int* o_input,
   int T_, int N_, int D_, int I_,  int max_items, bool static_bias, bool dynamic_bias, bool scalar){
   CUDA_KERNEL_LOOP(index, nthreads) {


 
  }
}

template <typename Dtype>
void SelectiveProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
 
  Dtype* d_bias;
  if(bottom.size() == 2) d_bias = bottom[1]->gpu_data();
  int batches = bottom[0]->shape(0);
  int jobs = batches*T_;

  SelectiveProductForward<Dtype><<<CAFFE_GET_BLOCKS(jobs), CAFFE_CUDA_NUM_THREADS>>>(jobs,
    this->blobs_[w_index_]->gpu_data(), bottom[0]->gpu_data(), this->blobs_[s_index_], this->blobs_[b_index_], d_bias, top[0]->mutable_gpu_data(),
    output_pindex_.gpu_data(), output_iindex_.gpu_data(), 
    T_, N_, D_, I_, output_max_, bias_, bottom.size() == 2, scalar_);
  CUDA_POST_KERNEL_CHECK;
}
 

template <typename Dtype>
void MILFrameLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  

}
