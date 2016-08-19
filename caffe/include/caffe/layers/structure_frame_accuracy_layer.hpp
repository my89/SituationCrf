#ifndef CAFFE_STRUCTURE_ACCURACY_LAYER_HPP_
#define CAFFE_STRUCTURE_ACCURACY_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
namespace caffe {
template <typename Dtype>
class StructureFrameAccuracyLayer : public Layer<Dtype> {
 public:
  explicit StructureFrameAccuracyLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "StructureFrameAccuracy"; }
  //takes in as many structures and outputs scores for each structure
  //take in a references from the data layer
  virtual inline int ExactNumBottomBlobs() const { return 3; }
  //return top-n verb, top-n arg, top-n any-match, top-n full-match, gold verb arg, gold verb any-match, gold verb full-match
  virtual inline int ExactNumTopBlobs() const { return 7; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /// @brief Not implemented -- AccuracyLayer cannot be used as a loss.
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }  }

  int total_frames;
  int references;
  int maxlabel;
  int topk;
  int mode;
  int curbatch;
  int maxbatch;
};

}  // namespace caffe

#endif  

