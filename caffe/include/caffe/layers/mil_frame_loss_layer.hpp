#ifndef CAFFE_FRAME_LOSS_LAYER_HPP_
#define CAFFE_FRAME_LOSS_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"


namespace caffe {

template <typename Dtype> 
class MILFrameLossLayer : public LossLayer<Dtype> {
 public:
  explicit MILFrameLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MILFrameLoss"; }
  virtual inline int ExactNumTopBlobs() const { return 3; }
  virtual inline int ExactNumBottomBlobs() const { return -1;}

 protected:
  virtual void PredictMarginalObject(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void PredictMarginalFrame(const vector<Blob<Dtype>*>& top, const Dtype* verb, std::vector<const Dtype*> args, const Dtype* bottom_label);
  virtual void PredictMarginalFrameGPU(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void PredictMaxFrameGPU(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual Dtype log_sum_exp(int length, int offset, int increment, const Dtype* data);
  virtual void toBlob(vector<int> in , Blob<Dtype> * out);
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
   virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  //set which inference problem is solved in the forward
  // 0: frame prediction (outputs a frame for each verb)
  // 1: object prediction (outputs a frame and a score) 
  int mode;

  Blob<Dtype> norms;
  Blob<Dtype> verb_marginal;
  Blob<Dtype> verb_marginal_scratch;
  Blob<Dtype> scratch;
  Blob<Dtype> max_scratch;
  Blob<Dtype> arg_marginal;
  Blob<Dtype> ref_scores;
  Blob<Dtype> pos_scores;
  Blob<Dtype> total_scores;
  Blob<Dtype> verb_max;
  Blob<Dtype> arg_max;
  Blob<Dtype> b_verb_length;
  Blob<Dtype> b_verb_start;
  Blob<Dtype> b_arg_structure;
  Blob<Dtype> b_arg_verb;
  Blob<Dtype> b_arg_index;
  vector<int> verb_length;
  vector<int> verb_start;
  vector<int> arg_structure;
  vector<int> arg_verb;
  vector<int> arg_index;
  vector< vector< pair< int,int > > > syn_arg_value;
  int max_value;
  int verb_index;
  int label_index;
  int maxlabel;
  int references;
  float lambda;
  int total_args;
  int total_frames;
  int total_syns;
  int total_bbs;
  int curbatch;
  int maxbatch;
  bool predict_only;
};
}
#endif

