#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/parameter_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ParameterLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const ParameterParameter2& param = this->layer_param_.parameter_param2();  
  int total_blobs_ = 1;
  D_ = param.d();
  N_ = param.n();

  if( this->blobs_.size() == total_blobs_){
    LOG(INFO) << "Skipping param init";
  } 
  else{
    this->blobs_.resize(total_blobs_);
    this->param_propagate_down_.resize(this->blobs_.size(), true);
    vector<int> shape(2);
    shape[0] = N_;
    shape[1] = D_;
    LOG(INFO) << "init blob 0 with N=" << N_ << " D= " << D_;
    this->blobs_[0].reset(new Blob<Dtype>(shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(param.weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    LOG(INFO) << "done init";
  }
}

template <typename Dtype>
void ParameterLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    
    vector<int> shape(2);
    shape[0] = this->blobs_[0]->shape()[0];
    shape[1] = this->blobs_[0]->shape()[1];
    top[0]->Reshape(shape);
}

template <typename Dtype>
void ParameterLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  caffe_copy(this->blobs_[0].get()->count(), this->blobs_[0].get()->cpu_data(), top[0]->mutable_cpu_data());
}

template <typename Dtype>
void ParameterLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  caffe_copy(this->blobs_[0].get()->count(), top[0]->cpu_diff(), this->blobs_[0].get()->mutable_cpu_diff());
}

#ifdef CPU_ONLY
STUB_GPU(ParameterLayer);
#endif

INSTANTIATE_CLASS(ParameterLayer);
REGISTER_LAYER_CLASS(Parameter);

}  // namespace caffe
