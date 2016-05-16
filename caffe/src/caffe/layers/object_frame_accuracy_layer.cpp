#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ObjectFrameAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  this->topk = this->layer_param_.object_frame_accuracy_param().topk();
}

template <typename Dtype>
void ObjectFrameAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void ObjectFrameAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  
  const Dtype* output_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int batch_size = bottom[1]->shape(0);
  int ref_size = bottom[1]->shape(3);   
  int label_space = bottom[0]->shape(1);

  //this is the real loss
  Dtype score = 0.;
  for (int i = 0; i < batch_size; i++){    
    int data_offset = i*label_space;
    int ref_offset = i*ref_size;
    std::vector<std::pair<Dtype, int> > _scores;
    for(int j = 0; j < label_space; j++){ 
      _scores.push_back(std::make_pair(output_data[data_offset+j],j));
    }
   
    std::partial_sort(
          _scores.begin(), _scores.begin() + this->topk,
          _scores.end(), std::greater<std::pair<Dtype, int> >());

    int found = 0;
    for( int k = 0 ; k < this->topk; k++){
      if(found) break;
      //find the index of the matching verb, if its in there. 
      int guessed_object = _scores[k].second;

      for(int j = 0; j < ref_size; j++){
        if( guessed_object == bottom_label[ref_offset + j]){
          found = 1;
          score ++;
          break;
        }
      }  
    }
  } 
  top[0]->mutable_cpu_data()[0] = score/batch_size;
}

INSTANTIATE_CLASS(ObjectFrameAccuracyLayer);
REGISTER_LAYER_CLASS(ObjectFrameAccuracy);

}
