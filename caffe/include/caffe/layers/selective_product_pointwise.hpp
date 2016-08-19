#ifndef CAFFE_SELECTIVE_PRODUCT_LAYER_HPP_
#define CAFFE_SELECTIVE_PRODUCT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
namespace caffe{
template <typename Dtype>
class SelectiveProductPointwiseLayer : public Layer<Dtype>{
 public:
  explicit SelectiveProductPointwiseLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SelectiveProductPointwise"; }
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int T_; //total number assignments to make
  int I_; //total number of input vectors
  int N_; //total number of layer's vectors
  int D_; //dim of vectors
  int C_; //clusters comming from the bottom  
  int P_;

  int total_blobs_;
  int w_index_;
  int o_index_;

  int scalar_;
  //this holds the unique inner products we need to compute and assigns them an id
  // i_inner_products[k] = input_[i]'*param_[i]
  Blob<int> i_input_;
  Blob<int> i_param_;
  Blob<int> output_;

  Blob<int> i_input_index_;
  Blob<int> i_param_index_;
  Blob<int> i_input_length_index_;
  Blob<int> i_param_length_index_;
  Blob<int> i_input_offset_index_;
  Blob<int> i_param_offset_index_;

  int output_max_;
  int param_max_;
  int input_max_;

  Blob<Dtype> i_inner_products_;
  Blob<int> maxes_;
  //this holds the mapping to the unique inner products
  // output_[i] = i_inner_products[output_id[i]]
  Blob<int> output_id_;
};
}
#endif
