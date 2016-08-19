#ifndef CAFFE_MULTILABEL_LAYER_HPP_
#define CAFFE_MULTILABEL_LAYER_HPP_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe{

template <typename Dtype>
class MultiLabelImageDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit MultiLabelImageDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~MultiLabelImageDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MultiLabelImageData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int mTopBlobs() const { return -1; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void load_batch(Batch<Dtype>* batch);

  vector<std::pair<std::string, std::pair< vector<int>, vector<int> > > > lines_;
  int lines_id_;
  float dropout;
  bool ignore_zero;
};

}
#endif
