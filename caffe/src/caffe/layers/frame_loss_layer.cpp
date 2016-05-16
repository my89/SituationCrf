#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void FrameLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
//  LossLayer<Dtype>::LayerSetUp(bottom, top);
  this->layer_param_.add_loss_weight(Dtype(1));
  this->layer_param_.add_loss_weight(Dtype(0));
  this->layer_param_.add_loss_weight(Dtype(0));
  FrameLossParameter frame_param = this->layer_param_.frame_loss_param(); 
  const string& syn_source = frame_param.syns();
  const string& source = frame_param.structure();  
  this->mode = frame_param.mode();
  this->maxlabel = frame_param.maxlabels();
  this->references = frame_param.references();
  this->lambda = frame_param.lambda();
  this->total_syns = frame_param.total_syns();
  LOG(INFO) << "Frame DEF; Opening File " << source.c_str();
  LOG(INFO) << "Value Def; Opening File " << syn_source.c_str();

  std::ifstream synfile(syn_source.c_str());
  for(int i = 0; i < this->total_syns; i++){
    std::vector<pair<int,int> > syn;
    this->syn_arg_value.push_back(syn);
  }
  std::string line;
  LOG(INFO) << "Opening File " << source.c_str();
  int n = 0;
  while( std::getline(synfile, line)){
    std::istringstream iss(line);
    std::string verb;
    int verb_index;
    int arg_index;
    int value_total;
    iss >> verb_index >> arg_index >> value_total;
    int syn;
    int value;    
    while( iss >> syn >> value ){     
      if( syn == -1) continue; 
      this->syn_arg_value[syn].push_back(std::make_pair(n, value));
    }
    n++;
  }
  
  LOG(INFO) << "lambda=" << this->lambda; 
  LOG(INFO) << "total syn incidence=" << this->syn_arg_value[0].size();
  int verb_curr = 0;
  int verbs = 0;
  int args = 0;
  std::ifstream infile(source.c_str());

  while( std::getline(infile, line)){
    std::istringstream iss(line);
    std::string verb;
    int verb_index;
    int arg_total;
    iss >> verb >> verb_index >> arg_total;
    
    std::string arg;
    int arg_index;
    int value_total;
    while( iss >> arg >> arg_index >> value_total ){
       this->arg_structure.push_back(value_total);       
       this->arg_verb.push_back(verb_index);
       this->arg_index.push_back(arg_index);
       args++;  
    } 
    verb_start.push_back(verb_curr);
    verb_length.push_back(arg_total);
    verb_curr += arg_total; 
    verbs++;
  }

  this->total_frames = verbs;
  this->total_args = args;
  verb_index = verb_curr;
  label_index = verb_curr+1;

}

template <typename Dtype>
void FrameLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
 // LossLayer<Dtype>::Reshape(bottom, top);
  vector<int> loss_shape(0);
  top[0]->Reshape(loss_shape);
 
  if( this->mode == 0){
    top[1]->Reshape(bottom[0]->shape(0),1,1,this->maxlabel*this->total_frames);
    top[2]->Reshape(bottom[0]->shape(0),1,1,this->total_frames);
  }
  else if ( this->mode == 1){
    top[1]->Reshape(bottom[0]->shape(0),this->total_syns,1,1);
  }
  
  this->arg_marginal.Reshape(bottom[label_index]->shape(0),1,1,this->total_args);
  this->norms.Reshape(bottom[label_index]->shape(0),1,1,1);
  this->verb_marginal.Reshape(bottom[label_index]->shape(0),1,1,this->total_frames);
  this->verb_marginal_scratch.Reshape(1,1,1,this->total_frames);
  this->ref_scores.Reshape(bottom[label_index]->shape(0),1,1,this->references);
  this->pos_scores.Reshape(bottom[label_index]->shape(0),1,1,1);
  this->verb_max.Reshape(bottom[label_index]->shape(0),1,1,1);
  this->arg_max.Reshape(bottom[label_index]->shape(0),1,this->total_frames,this->maxlabel);
}

template <typename Dtype>
void FrameLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) { 

  Dtype* norms = this->norms.mutable_cpu_data();
  Dtype* arg_marginal = this->arg_marginal.mutable_cpu_data();
  Dtype* verb_marginal = this->verb_marginal.mutable_cpu_data();
  Dtype* ref_scores = this->ref_scores.mutable_cpu_data();
  Dtype* pos_scores = this->pos_scores.mutable_cpu_data();
  Dtype* verb_max = this->verb_max.mutable_cpu_data();
  Dtype* arg_max = this->arg_max.mutable_cpu_data();
  const Dtype* bottom_label = bottom[this->label_index]->cpu_data();
  const Dtype* bottom_verb = bottom[this->verb_index]->cpu_data();
  std::vector<const Dtype*> args;
  
  int batch_size = bottom[this->label_index]->shape(0);
  
  LOG(INFO) << "shape 1 " << bottom[this->verb_index]->shape(0);
  LOG(INFO) << "shape 2 " << bottom[this->verb_index]->shape(1); 
//  LOG(INFO) << "shape 3 " << bottom[this->verb_index]->shape(2);
//  LOG(INFO) << "shape 4 " << bottom[this->verb_index]->shape(3);

  //LOG(INFO) << "verb shape " << bottom[this->verb_index]->shape(0) << " " << bottom[this->verb_index]->shape(1);
  //LOG(INFO) << "label sizes " << this->maxlabel << " " << this->references ;
  //LOG(INFO) <<  "label shape " << bottom[this->label_index]->shape(0) << " " << bottom[this->label_index]->shape(3);


  for(int i = 0; i < total_args; i++){
     args.push_back(bottom[i]->cpu_data());
  }

  //compute the marginals over args
  for(int i = 0; i < this->total_args; i++){
     for(int j = 0; j < batch_size; j++){
       //get the max
       int length = arg_structure[i];
       const Dtype* values = args[i];       
       Dtype total = 0.;
       int offset = j*length;
       Dtype max = values[offset];
       Dtype maxi = 0;
       for(int k = 1; k < length; k++){
          if ( values[offset + k] > max ){
            max = values[offset + k];
            maxi = k;
          }
       }
       int frame = arg_verb[i];
       if(mode != 1){
         int maxoffset = this->total_frames*this->maxlabel*j + this->maxlabel*frame + this->arg_index[i]; 
         arg_max[maxoffset] = maxi;        
       }
       for(int k = 0; k < length; k++){
          total += std::exp(values[offset + k] - max);
       }
       total = std::log(total) + max;
       arg_marginal[this->total_args*j + i] = total;
     }
  }

  //now compute the verb marginal
  for (int i = 0; i < batch_size; i++){ 
    for(int j = 0; j < this->total_frames; j++){      
      int verb_offset = i*this->total_frames + j;
      int arg_offset = i*this->total_args + this->verb_start[j];
      int nargs = this->verb_length[j];
      Dtype total = bottom_verb[verb_offset];           
      for(int k = 0; k < nargs; k++){
        total += arg_marginal[arg_offset + k];
      }
      verb_marginal[verb_offset] = total;      
    }
  }  
  //the norm is the log sum exp of verb_marginal
  for( int i = 0; i < batch_size; i++){
    int offset = i*this->total_frames;
    Dtype max = verb_marginal[offset];
    int maxv = 0;
    for( int j = 1; j < this->total_frames; j++){
      if(verb_marginal[offset + j] > max){
         max = verb_marginal[offset + j];
         maxv = j;
      }
    }
    verb_max[i] = maxv;
 
    Dtype total = 0.;
    for( int j = 0 ; j < this->total_frames; j++){
      total += std::exp(verb_marginal[offset + j] - max);
    }
    norms[i] = std::log(total) + max;
  }

  if(mode != 1){
  //compute the positive side of the score
  for( int i = 0; i < batch_size; i++){   
    for( int j = 0 ; j < this->references; j++){     
      int label_offset = i*this->maxlabel*this->references + j*this->maxlabel;
      int verb_index = bottom_label[label_offset];
      int arg_offset = verb_start[verb_index];
      Dtype total = bottom_verb[i*this->total_frames +verb_index];
      for( int k = 0; k < this->verb_length[verb_index]; k++){
         int length = arg_structure[arg_offset + k];
         int value_index = bottom_label[label_offset+k+1];
         total += args[arg_offset + k][length*i + value_index];
      }
      ref_scores[i*this->references + j] = total;              
    }
  }
  //log sum exp the refs, subtract norm and sum them all, and thats the objective.
  Dtype total_score = 0.;
  for(int i = 0; i < batch_size; i++){
    int offset = i*this->references; 
    Dtype max = ref_scores[offset];
    for(int j = 1; j < this->references; j++){
      if(ref_scores[offset + j] > max) max = ref_scores[offset + j];
    }
    Dtype total = 0.;
    for(int j = 0; j < this->references; j++){ 
      total += std::exp(ref_scores[offset +j] - max);
    }
    //LOG(INFO) << total + max <<" " <<  norms[i];
    pos_scores[i] = std::log(total) + max; // - std::log(this->references);
    total_score += pos_scores[i] - norms[i];
  }
  
  Dtype verb_marginal_score = 0.;
  for(int i = 0; i < batch_size; i++){
    int label_offset = i*this->maxlabel*this->references;
    int verb_offset = i*this->total_frames;
    int verb_index = bottom_label[label_offset];
    verb_marginal_score += verb_marginal[verb_offset+verb_index] - norms[i];

  }
  top[0]->mutable_cpu_data()[0] = (this->lambda*verb_marginal_score + total_score)/batch_size;
  }
  LOG(INFO) << "done forward";
  if(this->mode == 0) this->PredictMarginalFrame(top);
  if(this->mode == 1) this->PredictMarginalObject(bottom,top); 

}

template <typename Dtype> 
void FrameLossLayer<Dtype>::PredictMarginalFrame(const vector<Blob<Dtype>*>& top){
  Dtype* output_structure = top[1]->mutable_cpu_data();
  Dtype* output_scores = top[2]->mutable_cpu_data();
  const Dtype* verb_marginal = this->verb_marginal.cpu_data();
  const Dtype* arg_max = this->arg_max.cpu_data();

  int batchsize = top[1]->shape(0);
  for(int i = 0; i < batchsize; i++){
    for(int v = 0; v < this->total_frames; v++){
      int score_offset = i*this->total_frames;
      int structure_offset = i*this->total_frames*this->maxlabel + v*this->maxlabel;
      output_scores[score_offset + v] = verb_marginal[score_offset + v];
      output_structure[structure_offset]  = v;      
      for(int a = 0; a < verb_length[v]; a++){
         output_structure[structure_offset + a + 1 ] = arg_max[structure_offset+a];
      }
      for(int a = 1 + verb_length[v]; a < this->maxlabel; a++){
         output_structure[structure_offset + a] = -1;
      }
    }
  }
}

template <typename Dtype>
void FrameLossLayer<Dtype>::PredictMarginalObject(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  Dtype* output = top[1]->mutable_cpu_data();
  Dtype* arg_marginal = this->arg_marginal.mutable_cpu_data();
  Dtype* verb_scratch = this->verb_marginal_scratch.mutable_cpu_data();
  Dtype* verb_marginal = this->verb_marginal.mutable_cpu_data();
  Dtype* norms = this->norms.mutable_cpu_data();

  const Dtype* bottom_verb = bottom[this->verb_index]->cpu_data();
  std::vector<const Dtype*> args;
  
  int batch_size = bottom[this->label_index]->shape(0);
  
  for(int i = 0; i < total_args; i++){
     args.push_back(bottom[i]->cpu_data());
  }
  
  for(int i = 0; i < batch_size; i++){
    Dtype prob = 1;
    for(int s = 0; s < this->total_syns; s++){
      vector< pair <int , int > >  incidence = syn_arg_value[s];
      if(incidence.size() == 0) 
        LOG(INFO) << "incidence ERROR: " << s << " " << incidence.size();
      //compute the marginal score for synset
      int hits = 0;
      for(int v = 0; v < this->total_frames; v++){
        int verb_offset = i*this->total_frames + v;
        int arg_offset = verb_start[v];
        int length = verb_length[v];
        Dtype total = bottom_verb[verb_offset];
        for(int a = 0; a < length; a++){
          int q = a + arg_offset;
          int marginal_offset = i*total_args;
          Dtype _arg_marginal = arg_marginal[marginal_offset + q];
          for( int k = 0; k < incidence.size(); k++){
            if(incidence[k].first == q){
              Dtype x = args[q][arg_structure[q]*i + incidence[k].second];             
              //Dtype _temp = _arg_marginal;
              //_arg_marginal = log(exp(_arg_marginal) - exp(x));
              _arg_marginal = std::log( 1 - std::exp( x - _arg_marginal ) ) + _arg_marginal;
              hits++;
              break;
            } 
          }
          total += _arg_marginal;
        }      
        //if(verb_marginal[verb_offset] > total) LOG(INFO) << s <<" " << incidence.size() << " " << verb_marginal[verb_offset] << " " << total;
        verb_scratch[v] = total;        
      }
      //log sum exp the verbs
      Dtype max = verb_scratch[0];
      for( int j = 1; j < this->total_frames; j++){
        if(verb_scratch[j] > max){
          max = verb_scratch[j];
        }
      }
      Dtype total = 0.;
      for( int j = 0 ; j < this->total_frames; j++){
        total += std::exp(verb_scratch[j] - max);
      }
      total = std::log(total) + max;
      //prob(s) =  1-total(s)/norm , so we rank by negative total
      if( hits != incidence.size() ) LOG(INFO) << "hit error";
      if( total > norms[i]) LOG(INFO) << incidence.size() <<" "<< hits << " "  << total << " " << norms[i];
      output[i*this->total_syns + s] = 1 - exp(total - norms[i]); //probability no frame had value i. 
      prob *= 1 - output[i*this->total_syns + s]; 
    }
    //LOG(INFO) << "prob of seeing any value: " << 1 - prob;
  } 
}

template <typename Dtype>
void FrameLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  Dtype* norms = this->norms.mutable_cpu_data();
  Dtype* arg_marginal = this->arg_marginal.mutable_cpu_data();
  Dtype* verb_marginal = this->verb_marginal.mutable_cpu_data();
  Dtype* ref_scores = this->ref_scores.mutable_cpu_data();
  Dtype* pos_scores = this->pos_scores.mutable_cpu_data();
  const Dtype* bottom_label = bottom[this->label_index]->cpu_data();
  //const Dtype* bottom_verb = bottom[this->verb_index]->cpu_data();
  int batch_size = bottom[this->label_index]->shape(0); 

  //compute the negative part of the update
  Dtype* verb_diff = bottom[this->verb_index]->mutable_cpu_diff();
  for(int i = 0; i < batch_size; i++){
    int offset = i*this->total_frames;
    Dtype norm = norms[i];
    for(int j = 0; j < this->total_frames; j++){
      verb_diff[offset + j] = (this->lambda+1)*std::exp(verb_marginal[offset + j] - norm);
    }
  }

  for(int a = 0; a < this->total_args; a++){
    Dtype* arg_diff = bottom[a]->mutable_cpu_diff();
    const Dtype* arg = bottom[a]->cpu_data();    
    int frame_id = this->arg_verb[a];
    int length = arg_structure[a];
    for(int i = 0; i < batch_size; i++){
      //we need the verb marginal for this arg, and subtract the arg marginal
      int value_offset = i*length;      
      int arg_offset = i*this->total_args + a;  
      Dtype norm = norms[i];
      Dtype _verb_marginal = verb_marginal[i*this->total_frames + frame_id];      
      Dtype _arg_marginal = arg_marginal[arg_offset];
      for(int k = 0; k < length; k++){
        arg_diff[value_offset + k ] = (this->lambda+1)*std::exp(_verb_marginal - _arg_marginal + arg[value_offset + k] - norm);
      }     
    }
  }

//now handle the positive part of the gradient.
  for(int i = 0; i < batch_size; i++){
     for( int j = 0; j < this->references; j++){
      int label_offset = i*this->maxlabel*this->references;
      int verb_index = bottom_label[label_offset];
      int arg_offset = verb_start[verb_index];
      Dtype pos_grad = std::exp(ref_scores[i*this->references + j] - pos_scores[i]);
      verb_diff[i*this->total_frames + verb_index] -= pos_grad;     
      for(int k = 0; k < this->verb_length[verb_index]; k++){
         Dtype* arg_diff = bottom[arg_offset+k]->mutable_cpu_diff();
         int length = this->arg_structure[arg_offset+k];
         int value_index = length*i + (int)bottom_label[label_offset + k + 1];
         arg_diff[value_index] -= pos_grad; 
      }
    } 
  }
//now handle the marginal positive part of the gradient

  for(int i = 0; i < batch_size; i++){
    int label_offset = i*this->maxlabel*this->references;
    int verb_index = bottom_label[label_offset];
    int verb_offset = i*this->total_frames;
    int arg_offset = verb_start[verb_index];
    verb_diff[verb_offset + verb_index] -= this->lambda;
    //update the gradient of all the arguments
    for(int k =0; k < this->verb_length[verb_index]; k++){
      Dtype* arg_diff = bottom[arg_offset+k]->mutable_cpu_diff();
      int length = this->arg_structure[arg_offset + k];
      int value_offset = i*length;
      Dtype _arg_marginal =  arg_marginal[i*this->total_args+arg_offset+k];
      const Dtype* values = bottom[arg_offset+k]->cpu_data();
      for(int v = 0; v < length; v++){
         arg_diff[value_offset+v] -= this->lambda*std::exp(values[value_offset+v] - _arg_marginal);
      }
    }
  }  
}

INSTANTIATE_CLASS(FrameLossLayer);
REGISTER_LAYER_CLASS(FrameLoss);

}  // namespace caffe
