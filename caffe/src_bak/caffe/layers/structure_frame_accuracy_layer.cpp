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
void StructureFrameAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  this->maxlabel = this->layer_param_.structure_frame_accuracy_param().maxlabels(); 
  this->references = this->layer_param_.structure_frame_accuracy_param().references();
  this->total_frames = this->layer_param_.structure_frame_accuracy_param().total_verbs();
  this->topk = this->layer_param_.structure_frame_accuracy_param().topk();
  this->mode = this->layer_param_.structure_frame_accuracy_param().mode();
  this->maxbatch = this->layer_param_.structure_frame_accuracy_param().batch();
  this->curbatch = 0;
}

template <typename Dtype>
void StructureFrameAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  for(int i = 0; i < 7; i++) top[i]->Reshape(top_shape);
}

template <typename Dtype>
void StructureFrameAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  
  const Dtype* bottom_refs = bottom[0]->cpu_data();
  const Dtype* bottom_outputs = bottom[1]->cpu_data();
  const Dtype* bottom_scores = bottom[2]->cpu_data();
  int batch_size = bottom[1]->shape(0);
  //this is the real loss
  Dtype verb_score = 0.;
  Dtype arg_score = 0.;
  Dtype any_score = 0.;
  Dtype full_score = 0.;
  Dtype vstar_arg_score = 0.;
  Dtype vstar_any_score = 0.;
  Dtype vstar_full_score = 0.;
  std::ofstream outfile;
  outfile.open(this->layer_param_.structure_frame_accuracy_param().outfile().c_str(), std::ios_base::app);
  Dtype arg_count = 0.;
  //LOG(INFO) << "batch_size" << batch_size;
  //LOG(INFO) << "label_size" << bottom[1]->shape(3);
  for (int i = 0; i < batch_size; i++){    
    int ref_offset = i*this->maxlabel*this->references;
    int scores_offset = i*this->total_frames;
    int output_offset = i*this->total_frames*this->maxlabel;
    std::vector<std::pair<Dtype, int> > _scores; 
    for(int j = 0; j < this->total_frames; j++){\
      _scores.push_back(std::make_pair(bottom_scores[scores_offset+j],j));
    }
    std::partial_sort(
          _scores.begin(), _scores.begin() + this->topk,
          _scores.end(), std::greater<std::pair<Dtype, int> >());
    int reference_verb = bottom_refs[ref_offset];
    int found = 0;
    for( int k = 0 ; k < this->topk; k++){
      //find the index of the matching verb, if its in there. 
      int guessed_verb = bottom_outputs[output_offset + _scores[k].second*this->maxlabel];
      if( reference_verb == guessed_verb ) { found = 1; break; }
    }
    if(mode == 1){
      std::stringstream buffer;
      int guess_offset = output_offset + _scores[0].second*this->maxlabel;
      int guessed_verb = bottom_outputs[guess_offset];
      buffer << reference_verb << "\t" <<  guessed_verb;
      for(int i = 1; i < this->maxlabel; i++){
        buffer << "\t" << bottom_outputs[guess_offset +i] ;
      }
      outfile << buffer.str() << "\n"; 
    }
    
    if(found) verb_score++;
    int total_args = 0;
    for(int j = 1; j < this->maxlabel; j++){
      if(bottom_refs[ref_offset+j] != -1) total_args++;
    }
    arg_count += total_args;
    //find the reference corresponding to the verb and compute the arg score, which gets into the vstar only if found
    int output_verb_offset = i*this->total_frames*this->maxlabel + reference_verb*this->maxlabel;
    int found_args = 0;
    for(int j = 1; j < this->maxlabel; j++){
      int predicted_arg = bottom_outputs[output_verb_offset + j];
      if(predicted_arg == -1) break;
      for(int r = 0; r < this->references; r++){
        if(bottom_refs[ref_offset+r*this->maxlabel+j] == predicted_arg){
          if(found){
            arg_score++;
          }
          vstar_arg_score++;
          found_args++;
          break;
        }
      }
    }
    if(found_args == total_args){
      if(found) any_score++;
      vstar_any_score++;
    }
    //compute the full score  
    for( int r = 0; r < this->references; r++ ){
      int nfound = 0;
      for(int j = 1; j < this->maxlabel; j++){        
        int predicted_arg = bottom_outputs[output_verb_offset + j];
        int ref_arg = bottom_refs[ref_offset+r*this->maxlabel+j];
        if(ref_arg == -1) break;
        if(ref_arg == predicted_arg){ nfound++;}
      }
      if(nfound == total_args){
        if(found) full_score++;
        vstar_full_score++;
        break;
      }
    }
  }
  if (curbatch == 0){
    LOG(INFO) << "clearing...";
    for(int i = 0; i < 7; i++) top[i]->mutable_cpu_data()[0] = 0;
  } 
  top[0]->mutable_cpu_data()[0] += verb_score/(batch_size*maxbatch);
  top[1]->mutable_cpu_data()[0] += arg_score/(arg_count*maxbatch);
  top[2]->mutable_cpu_data()[0] += any_score/(batch_size*maxbatch);
  top[3]->mutable_cpu_data()[0] += full_score/(batch_size*maxbatch);
  top[4]->mutable_cpu_data()[0] += vstar_arg_score/(arg_count*maxbatch);
  top[5]->mutable_cpu_data()[0] += vstar_any_score/(batch_size*maxbatch);
  top[6]->mutable_cpu_data()[0] += vstar_full_score/(batch_size*maxbatch);
  curbatch = (curbatch + 1) % maxbatch;
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(StructureFrameAccuracyLayer);
REGISTER_LAYER_CLASS(StructureFrameAccuracy);

}
