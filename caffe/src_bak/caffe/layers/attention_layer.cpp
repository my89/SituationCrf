#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void AttentionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  AttentionParameter attention_param = this->layer_param_.attention_param();
  this->spatial_index = 0;
  this->attention_index = 1;
  this->batch = bottom[spatial_index]->shape(0);
  this->dim = bottom[spatial_index]->shape(1);
  this->height = bottom[spatial_index]->shape(2);
  this->width = bottom[spatial_index]->shape(3);
  this->map = attention_param.map();
  this->args = attention_param.args();
 
  CHECK_EQ(batch, bottom[attention_index]->shape(0)) << "Batch dim mismatch";
  CHECK_EQ(height*width*args, bottom[attention_index]->shape(1)) << "Attention image dim mismatch";
}

template <typename Dtype>
void AttentionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  this->batch = bottom[spatial_index]->shape(0);
  this->dim = bottom[spatial_index]->shape(1);
  this->height = bottom[spatial_index]->shape(2);
  this->width = bottom[spatial_index]->shape(3);

  CHECK_EQ(batch, bottom[attention_index]->shape(0)) << "Batch dim mismatch";
  CHECK_EQ(height*width*args, bottom[attention_index]->shape(1)) << "Attention image dim mismatch";

  top[0]->Reshape(batch,args*dim, 1, 1);
}

template <typename Dtype>
void AttentionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* spatial = bottom[spatial_index]->cpu_data();
  const Dtype* attention = bottom[attention_index]->cpu_data();
  Dtype* output = top[0]->mutable_cpu_data();  
  for (int i = 0; i < batch; i++) {
    for(int a = 0; a < args; a++){
      for( int d = 0; d < dim; d++){
        Dtype avg = 0;
        for(int h = 0; h < height; h++){
          int offset_space = i*dim*height*width + d*height*width + h*width;
          int offset_attention = i*args*height*width + args*height*width + h*width;
          for(int w = 0; w < width; w++){
            avg += spatial[offset_space + w]*attention[offset_attention + w];
          }
        }
        output[i*dim + d] = avg;
      }
    }
  }
}

template <typename Dtype>
void AttentionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* spatial = bottom[spatial_index]->cpu_data();
  const Dtype* attention = bottom[attention_index]->cpu_data();
  
  Dtype* spatial_diff = bottom[spatial_index]->mutable_cpu_diff();
  Dtype* attention_diff = bottom[attention_index]->mutable_cpu_diff();

  //spatial diff
  for(int i = 0; i < batch; i++){
    for(int a = 0; a < args; a++){
      for(int d = 0; d < dim; d++){
        Dtype _top_diff = top_diff[i*args*dim + a*dim + d];
        for(int h = 0; h < height; h++){
          int offset_space = i*dim*height*width + d*height*width + h*width;
          int offset_attention = i*args*height*width + args*height*width + h*width;
          for(int w = 0; w < width; w++){
            spatial_diff[offset_space + w] = _top_diff*attention[offset_attention + w];
          }             
        }
      }
    } 
  }
  //attention diff
  for(int i = 0; i < batch; i++){
    for(int a = 0; a < args; a++){
      for(int h = 0; h < height; h++){
        for(int w = 0; w < width; w++){
          Dtype val = 0.f;
          int top_offset = i*args*dim + a*dim;
          int spatial_offset = i*dim*height*width + h*width + w;
          for(int d = 0; d < dim; d++){
            val += top_diff[top_offset +d]*spatial[spatial_offset + d*height*width];
          }
          attention_diff[i*args*height*width + a*height*width + h*width + w] = val;
        }
      }
    }
  }
}


#ifdef CPU_ONLY
//STUB_GPU(PoolingLayer);
#endif

INSTANTIATE_CLASS(AttentionLayer);
REGISTER_LAYER_CLASS(Attention);
}  // namespace caffe
