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
void FullRowFrameAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  this->maxlabel = this->layer_param_.frame_loss_param().maxlabels(); 
  this->references = this->layer_param_.frame_loss_param().references();   
  const string& source = this->layer_param_.frame_loss_param().structure();
  std::ifstream infile(source.c_str());
  LOG(INFO) << "Opening File " << source.c_str();
  string line;
  while( std::getline(infile, line)){
    std::istringstream iss(line);
    vector<int> ref;
    int v;
    while( iss >> v) ref.push_back(v);
    if( ref.size() != maxlabel){
      LOG(INFO) << "structure file doesn't correspond to promised references!!!";
    }
    outputs.push_back(ref);
  }
  this->total_verbs = outputs.size() / this->references;
}

template <typename Dtype>
void FullRowFrameAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->shape(0), 1 , 1 , this->total_verbs*this->maxlabel); 
  top[1]->Reshape(bottom[0]->shape(0), 1, 1, this->total_verbs);

}

template <typename Dtype>
void FullRowFrameAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* output_data = bottom[0]->cpu_data();
  Dtype* output_structure = top[0]->mutable_cpu_data();
  Dtype* output_weights = top[1]->mutable_cpu_data();
  int batch_size = bottom[0]->shape(0);

  for (int i = 0; i < batch_size; i++){
    for( int v = 0; v < this->total_verbs; v++){    
      int input_offset = i*outputs.size() + v*this->references;
      int mi = 0;
      Dtype mv = output_data[input_offset];
      for(int j = 1; j < this->references; j++){ 
        Dtype tv = output_data[input_offset + j]; 
        if( tv  > mv ){
          mi = j;
          mv = tv;
        }
      }
      vector<int> output = outputs[v*this->references + mi];
      int output_offset0 = i*this->total_verbs*this->maxlabel + v*this->maxlabel;
      //copy a frame into the output
      for(int j = 0; j < this->maxlabel; j++){
        output_structure[output_offset0+j] = output[j];
      }
      //copy the weight for that frame
      output_weights[i*this->total_verbs+v] = mv;
    }
  }
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(FullRowFrameAccuracyLayer);
REGISTER_LAYER_CLASS(FullRowFrameAccuracy);

}  // namespace caffe
