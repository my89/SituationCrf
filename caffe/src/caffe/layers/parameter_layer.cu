#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/parameter_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ParameterLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  caffe_copy(this->blobs_[0].get()->count(), this->blobs_[0].get()->gpu_data(), top[0]->mutable_gpu_data());
  //const Dtype*  data = this->blobs_[0].get()->cpu_data();
  //for(int i = 0; i < 10; i++) LOG(INFO) << "i ( " << i << ") = " << data[i]; 

}

template <typename Dtype>
void ParameterLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
 // const Dtype*  data = top[0]->cpu_diff();
 // const Dtype*  data2 = this->blobs_[0].get()->cpu_data();
  caffe_copy(this->blobs_[0].get()->count(), top[0]->gpu_diff(), this->blobs_[0].get()->mutable_gpu_diff());
 
//  for(int i = 0; i < 32; i++) LOG(INFO) << "i diff ( " << i << ") = " << data[i] << " val " << data2[i]; 
 //LOG(INFO) << 

}

INSTANTIATE_LAYER_GPU_FUNCS(ParameterLayer);

}  // namespace caffe
