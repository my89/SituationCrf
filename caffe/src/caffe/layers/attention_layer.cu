#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void AttentionForward(const int nthreads,
   const Dtype* spatial, const Dtype* attention, Dtype* output,  
   int args, int dim, int height, int width){ 
   CUDA_KERNEL_LOOP(index, nthreads) {
     //index = i*args*dim + a*dim + d
     int i = index / (args*dim);
     int a = (index / dim) % args;
     int d = index % dim;
     Dtype avg = 0;
     for(int h = 0; h < height; h++){
       int offset_space = i*dim*height*width + d*height*width + h*width;
       int offset_attention = i*args*height*width +a*height*width + h*width;
       for(int w = 0; w < width; w++){
         avg += spatial[offset_space + w]*attention[offset_attention + w];
       }
     }
     output[index] = avg/(height*width);
   }
}

template <typename Dtype>
__global__ void SpatialDiff(const int nthreads,
   const Dtype* spatial, const Dtype* attention, const Dtype* top_diff, Dtype* spatial_diff,  
   int args, int dim, int height, int width){ 
   CUDA_KERNEL_LOOP(index, nthreads) {
     //index = i*dim*height*width + d*height*width + h*width + w
     //index = i*dim + d
     int i = index / (dim*height*width);
     int d = (index / (height*width)) % dim;
     int h = (index / width) % height;
     int w = index % width;

     int diff_offset = i*args*dim + d;
     int map = height*width;     
     int attention_offset = i*args*map + h*width + w;
     Dtype total = 0.f;
     for(int a = 0; a < args; a++){ 
        Dtype _top_diff = top_diff[diff_offset + a*dim];
        Dtype _attention = attention[attention_offset + a*map];
        total+= _top_diff * _attention;
     }
     spatial_diff[index] = total/(height*width);
  /*for(int h = 0; h < height; h++){
        int offset_space = i*dim*height*width + d*height*width + h*width;
        for(int w = 0; w < width; w++){
          spatial_diff[offset_space + w] = _top_diff*attention[offset_attention + w];
        }             
      }
   */
  } 
}

template <typename Dtype> 
__global__ void AttentionDiff(const int nthreads,
   const Dtype* spatial, const Dtype* attention, const Dtype* top_diff, Dtype* attention_diff,  
   int args, int dim, int height, int width){ 
   CUDA_KERNEL_LOOP(index, nthreads) {
     //index = i*args*height*width + a*height*width + h*width + w
     int i = index / (args*height*width);
     int a = (index / (height*width)) % args;
     int h = (index / width) % height;
     int w = index % width;
     Dtype val = 0.f;
     int top_offset = i*args*dim + a*dim;
     int spatial_offset = i*dim*height*width + h*width + w;
     for(int d = 0; d < dim; d++){
       val += top_diff[top_offset +d]*spatial[spatial_offset + d*height*width];
     }
     attention_diff[index] = val/(height*width);
     //index = i*height*width + h*width + w
/*     int i = index / (height*width);
     int h = (index / width) % height;
     int w = index % width;
     Dtype val = 0.f;
     int top_offset = i*dim;
     int spatial_offset = i*height*width*dim + h*width + w;
     for(int d = 0; d < dim; d++){
       val += top_diff[top_offset +d]*spatial[spatial_offset + d*height*width];
     }
     attention_diff[index] = val;
*/
  }
}

template <typename Dtype>
void AttentionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    int jobs = batch * args * dim;
    AttentionForward<Dtype><<<CAFFE_GET_BLOCKS(jobs), CAFFE_CUDA_NUM_THREADS>>>(jobs, 
      bottom[spatial_index]->gpu_data(), bottom[attention_index]->gpu_data(), top[0]->mutable_gpu_data(),
      args, dim, height, width);
    CUDA_POST_KERNEL_CHECK;
    /*if(itt > 503) itt = 0;
    int to = 3*itt*height*width;
    const Dtype* att = bottom[attention_index]->cpu_data();
    for(int k = 0 ; k < 5; k++){
    int o = to + k*args*height*width;
    for(int i = 0; i < height; i++){
      LOG(INFO) << att[o + i*width + 0] << " " << att[o + i*width + 1]<< " " << att[o + i*width + 2] << " " << att[o + i*width + 3] << " " << att[o + i*width + 4] << " " << att[o + i*width + 5] << " " << att[o + i*width + 6] ;
    }
    LOG(INFO) << " --- ";
  }
  itt++;
  */
} 


template <typename Dtype>
void AttentionLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
   
 //   const Dtype* att = bottom[attention_index]->cpu_data();
  //  LOG(INFO)<< att[0] << "," << att[1];
 
    int spatial_jobs = batch*dim*height*width;
    SpatialDiff<Dtype><<<CAFFE_GET_BLOCKS(spatial_jobs), CAFFE_CUDA_NUM_THREADS>>>(spatial_jobs, 
      bottom[spatial_index]->gpu_data(), bottom[attention_index]->gpu_data(), top[0]->gpu_diff(),bottom[spatial_index]->mutable_gpu_diff(),
      args,dim, height, width);
    CUDA_POST_KERNEL_CHECK;
 
    int attention_jobs = batch*args*height*width;
    AttentionDiff<Dtype><<<CAFFE_GET_BLOCKS(attention_jobs), CAFFE_CUDA_NUM_THREADS>>>(attention_jobs, 
      bottom[spatial_index]->gpu_data(), bottom[attention_index]->gpu_data(), top[0]->gpu_diff(),bottom[attention_index]->mutable_gpu_diff(),
      args,dim, height, width);
    CUDA_POST_KERNEL_CHECK;
 //   LOG(INFO) << spatial_index << "," << bottom[spatial_index]->shape(0) << "," << bottom[spatial_index]->shape(1);
 
//    const Dtype* v = bottom[spatial_index]->cpu_data();
//    LOG(INFO) << v[0] << "," << v[1];
} 

INSTANTIATE_LAYER_GPU_FUNCS(AttentionLayer);

}
