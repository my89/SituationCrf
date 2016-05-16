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
void MILFrameLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
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
  this->total_bbs = frame_param.bounding_boxes();
  this->maxbatch = frame_param.batch();
  this->curbatch = 0;
  //this->bounding_boxes = frame_param.bounding_boxes();
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
    if( value_total > max_value) max_value = value_total;
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
 
  toBlob(this->verb_length, &this->b_verb_length);
  toBlob(this->verb_start, &this->b_verb_start);
  toBlob(this->arg_structure, &this->b_arg_structure);
  toBlob(this->arg_verb, &this->b_arg_verb);
  toBlob(this->arg_index, &this->b_arg_index);
   
}

template <typename Dtype>
void MILFrameLossLayer<Dtype>::toBlob(vector<int> in , Blob<Dtype> * out){   
  out->Reshape(1,1,1,in.size());
  Dtype* data = out->mutable_cpu_data();
  for( int i = 0; i < in.size(); i++) data[i] = (Dtype)in[i];
}

template <typename Dtype>
void MILFrameLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
 // LossLayer<Dtype>::Reshape(bottom, top);
  vector<int> loss_shape(0);
  top[0]->Reshape(loss_shape);
  
  int batches = bottom[this->label_index]->shape(0);
 
  if( this->mode == 0){
    top[1]->Reshape(batches,1,1,this->maxlabel*this->total_frames);
    top[2]->Reshape(batches,1,1,this->total_frames);
  }
  else if ( this->mode == 1){
    top[1]->Reshape(batches,this->total_syns,1,1);
  }
  
  //int box_total = batches * this->total_bbs;
  this->arg_marginal.Reshape(batches, this->total_args, this->total_bbs, 1);
  this->norms.Reshape(batches, this->total_bbs, 1, 1);
  this->verb_marginal.Reshape(batches, this->total_frames, this->total_bbs, 1);
  this->scratch.Reshape(batches,this->total_args,this->max_value,1);
  this->max_scratch.Reshape(batches, this->total_args,1,1);
  this->ref_scores.Reshape(batches,this->references,this->total_bbs, 1);
  this->pos_scores.Reshape(batches,this->references, 1, 1);
  this->total_scores.Reshape(batches, 1 , 1, 1);
  //this->verb_max.Reshape(box_total,1,1,1);
  //this->arg_max.Reshape(box_total,1,this->total_frames,this->maxlabel);
}

template <typename Dtype> 
Dtype MILFrameLossLayer<Dtype>::log_sum_exp(int length, int offset, int increment, const Dtype* data){
  int kmin = offset;
  int kmax = offset + length*increment;

  Dtype total = 0.;
  Dtype max = data[kmin];

  for(int k = kmin+increment; k < kmax; k+=increment){
    if (data[k] > max ) max = data[k];
  }
  for(int k = kmin; k < kmax; k+=increment){
    total += std::exp(data[k] - max);
  }
 
  return std::log(total) + max;
}

template <typename Dtype>
void MILFrameLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) { 

  Dtype* norms = this->norms.mutable_cpu_data();
  Dtype* arg_marginal = this->arg_marginal.mutable_cpu_data();
  Dtype* verb_marginal = this->verb_marginal.mutable_cpu_data();
  Dtype* ref_scores = this->ref_scores.mutable_cpu_data();
  Dtype* pos_scores = this->pos_scores.mutable_cpu_data();
  Dtype* total_scores = this->total_scores.mutable_cpu_data();

  const Dtype* bottom_label = bottom[this->label_index]->cpu_data();
  const Dtype* bottom_verb = bottom[this->verb_index]->cpu_data();
  std::vector<const Dtype*> args;
  
  int batch_size = bottom[this->label_index]->shape(0);
  //LOG(INFO) << "batch size " << batch_size;
  //LOG(INFO) << "arg0 shape " << bottom[0]->shape(0) << " " << bottom[0]->shape(1);
  //LOG(INFO) << "arg1 shape " << bottom[1]->shape(0) << " " << bottom[1]->shape(1);
  //LOG(INFO) << "label sizes " << this->maxlabel << " " << this->references ;
  //LOG(INFO) <<  "label shape " << bottom[this->label_index]->shape(0) << " " << bottom[this->label_index]->shape(3);

  for(int i = 0; i < total_args; i++){
     args.push_back(bottom[i]->cpu_data());
  }

  //compute the marginals over args
  for(int i = 0; i < batch_size; i++){
    for(int a = 0; a < this->total_args; a++){
      const Dtype* values = args[a];
      for(int b = 0; b < this->total_bbs; b++){
         //get the max
        int length = arg_structure[a];
        int value_offset = i*this->total_bbs*length + b*length;
        int marginal_offset = i*this->total_args*this->total_bbs + a*this->total_bbs + b;
        arg_marginal[marginal_offset] = this->log_sum_exp(length, value_offset, 1, values);
      }
    }
  }
  //now compute the verb marginal
  for (int i = 0; i < batch_size; i++){
    for(int f = 0; f < this->total_frames; f++){      
      for(int b = 0; b < this->total_bbs; b++){ 
        int verb_offset = i*this->total_bbs*this->total_frames + b*this->total_frames + f;
        int arg_offset = i*this->total_args*this->total_bbs + this->verb_start[f]*this->total_bbs + b;
        int nargs = this->verb_length[f];
        Dtype total = bottom_verb[verb_offset];           
        for(int k = 0; k < nargs; k++){
          total += arg_marginal[arg_offset + k*this->total_bbs];
        }
        int verb_marginal_offset = i*this->total_frames*this->total_bbs + f*this->total_bbs + b;
        verb_marginal[verb_marginal_offset] = total;      
      }
    }
  }  
  //the norm is the log sum exp of verb_marginal
  for( int i = 0; i < batch_size; i++){
    for( int b = 0; b < this->total_bbs; b++){
      int verb_offset = i*this->total_bbs*this->total_frames + b;
      norms[i*this->total_bbs + b] = this->log_sum_exp(this->total_frames, verb_offset , this->total_bbs, verb_marginal); 
    }
  }

  //compute the positive side of the score
  for( int i = 0; i < batch_size; i++){   
    for( int r = 0 ; r < this->references; r++){     
      for(int b = 0; b < this->total_bbs; b++){
        int label_offset = i*this->references*this->maxlabel + r*this->maxlabel;
        int verb_index = bottom_label[label_offset];
        int arg_offset = verb_start[verb_index];
        Dtype total = bottom_verb[i*this->total_bbs*this->total_frames + b*this->total_frames + verb_index];
        for( int k = 0; k < this->verb_length[verb_index]; k++){
          int arg_index = arg_offset + k;
          int arg_length = arg_structure[arg_index];
          int value_index = bottom_label[label_offset+k+1];
          total += args[arg_index][i*this->total_bbs*arg_length + b*arg_length + value_index];
        }
        ref_scores[i*this->references*this->total_bbs + r*this->total_bbs + b] = total;              
      }
    }
  }
  //log sum exp the refs, subtract norm and sum them all, and thats the objective.
  Dtype total_score = 0.;
  for(int i = 0; i < batch_size; i++){  
    //Dtype allzero = 0.;
    Dtype not1 = 0.; //sum has the probability that all of the boxes don't have any of the references
    for(int r = 0; r < this->references; r++){
      Dtype _not1 = 0.;
      for( int b = 0; b < this->total_bbs; b++ ){
        Dtype ref = ref_scores[i*this->total_bbs*this->references + r*this->total_bbs + b];
        Dtype norm = norms[i*this->total_bbs + b];       
        Dtype value = std::exp(ref-norm);
        //LOG(INFO) << "ref = " << ref << " norm=" << norm << " value " << value;
        _not1 = _not1 + value - _not1*value;
      }
      pos_scores[i*this->references + r] = _not1;
      not1 = not1 + _not1 - _not1*not1;
    }
    total_score += std::log(not1); //not1 is v in : 1-v; the total score is 1-(1-v) = v
    total_scores[i] = std::log(not1); //total_score is the probability everyone is zero
  }
  //LOG(INFO) << "done. " << batch_size;
  if(curbatch == 0)  top[0]->mutable_cpu_data()[0] = 0;
  top[0]->mutable_cpu_data()[0] += (total_score)/(batch_size*maxbatch);
  curbatch = (curbatch + 1) % maxbatch;
  if(this->mode == 0) this->PredictMarginalFrame(top, bottom_verb, args, bottom_label);
  //if(this->mode == 1) this->PredictMarginalObject(bottom,top); 

}

template <typename Dtype> 
void MILFrameLossLayer<Dtype>::PredictMarginalFrame(const vector<Blob<Dtype>*>& top, const Dtype* verb, std::vector<const Dtype*> args, const Dtype* label){
  Dtype* output_structure = top[1]->mutable_cpu_data();
  Dtype* output_scores = top[2]->mutable_cpu_data();
  const Dtype* norms = this->norms.cpu_data();
  const Dtype* verb_marginal = this->verb_marginal.cpu_data();
  const Dtype* arg_marginal = this->arg_marginal.cpu_data();
  Dtype* scratch = this->scratch.mutable_cpu_data(); 
  int batchsize = top[1]->shape(0);
  
  LOG(INFO) << "batch size " << batchsize;
  LOG(INFO) << "bbs" << this->total_bbs;
  LOG(INFO) << "frames" << this->total_frames;
  LOG(INFO) << "max_labels" << this->maxlabel; 
  for(int i = 0 ; i < batchsize; i++){
//    Dtype maxv = 0;
//    int max_verb = 0;
    for(int v = 0; v < this->total_frames; v++){    
      int score_offset = i*this->total_frames;
      int structure_offset = i*this->total_frames*this->maxlabel + v*this->maxlabel;
      int offset = i*this->total_frames*this->total_bbs + v*this->total_bbs;
      int norm_offset = i*total_bbs;
      Dtype not1 = 0;
      for(int b = 0; b < this->total_bbs; b++){
        Dtype value =  std::exp(verb_marginal[offset + b] - norms[norm_offset + b]);
        not1 = not1 + value - not1*value;       
        //not1 += value*(1-not1); 
      }  
      output_scores[score_offset+v] = not1;
/*      if( not1 > maxv ) { 
        maxv = not1;
        max_verb = v;
      }
*/
      output_structure[structure_offset] = v;
//      if( label[i*this->references*this->maxlabel] == v) LOG(INFO) << "gold p(" << v << ")" <<  not1;//1 - std::exp(sum);
      for( int a = 0; a < verb_length[v]; a++){
      //here we need the per argument marginal max
        int arg_index = this->verb_start[v] + a;
        int length = this->arg_structure[arg_index];
        //int verb_offset = i*this->total_bbs*this->total_frames;
        const Dtype* values = args[arg_index];
        Dtype max_s_value = 0.;
        int max_s = 0;
        for(int s = 0; s < length; s++){
          scratch[s] = 0;
        }
        for(int b = 0; b < this->total_bbs; b++){
          Dtype _arg_marginal = arg_marginal[i*this->total_bbs*this->total_args + arg_index*this->total_bbs + b];
          //Dtype verb_pot = verb[verb_offset + b*this->total_frames + v];
          //Dtype verb_norm = verb_marginal[verb_offset + b*this->total_frames + v];
          //Dtype verb_values = verb_pot - verb_norm;
          int value_offset = i*this->total_bbs*length + b*length;          
          for(int s = 0; s < length; s++){
            //scratch[s] += std::log(1 - std::exp(values[value_offset + s] - _arg_marginal));
            Dtype value = std::exp(values[value_offset + s] - _arg_marginal);
            scratch[s] = scratch[s] + value - value*scratch[s];
          }
        }
        for(int s = 0; s < length; s++){
          Dtype val = scratch[s];//1 - std::exp(scratch[s]);
          //LOG(INFO) << s << ":" << val;
          if( val > max_s_value){ 
            max_s_value = val;
            max_s = s;
          }
        }
        output_structure[structure_offset + a + 1] = max_s;
      }      
      for(int a = 1 + verb_length[v]; a < this->maxlabel; a++){
         output_structure[structure_offset + a ] = -1;
      }
    }
//    LOG(INFO) << "max p(" << max_verb << ")" << maxv;
  }
}

template <typename Dtype>
void MILFrameLossLayer<Dtype>::PredictMarginalObject(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
/*  Dtype* output = top[1]->mutable_cpu_data();
  Dtype* arg_marginal = this->arg_marginal.mutable_cpu_data();
  Dtype* verb_scratch = this->verb_marginal_scratch.mutable_cpu_data();
  //Dtype* verb_marginal = this->verb_marginal.mutable_cpu_data();
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
*/
}

template <typename Dtype>
void MILFrameLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  const Dtype* norms = this->norms.cpu_data();
  const Dtype* arg_marginal = this->arg_marginal.cpu_data();
  const Dtype* verb_marginal = this->verb_marginal.cpu_data();
  const Dtype* ref_scores = this->ref_scores.cpu_data();
//  const Dtype* pos_scores = this->pos_scores.cpu_data();
  const Dtype* total_scores = this->total_scores.cpu_data();
  const Dtype* bottom_label = bottom[this->label_index]->cpu_data();
  //const Dtype* bottom_verb = bottom[this->verb_index]->cpu_data();
  int batch_size = bottom[this->label_index]->shape(0); 

  Dtype* verb_grad = bottom[this->verb_index]->mutable_cpu_diff();        
  //compute the negative and positive part together
  for(int i = 0; i < batch_size; i++){
    Dtype at_least_1 = total_scores[i];
//    LOG(INFO) << "batch " << i << " total score (at least 1) " << at_least_1;
    Dtype sum_scalar = 0.;
   // total_pos_score = std::log(total_pos_score);
    for(int r = 0; r < this->references; r++){       
      for(int b = 0; b < this->total_bbs; b++){  
        Dtype score = ref_scores[i*this->total_bbs*this->references  + r*this->total_bbs + b] - norms[i*this->total_bbs +b];
        Dtype top_scalar = std::exp( log(1-std::exp(at_least_1)) - log(1 - std::exp(score)) + score); //probability that everything else is not it plus this bb and ref is it
        Dtype bottom_scalar = std::exp(at_least_1); //probabilty that something is it
//        if(top_scalar < 1e-7) top_scalar = 1e-7;
//        if(bottom_scalar < 1e-7) bottom_scalar = 1e-7;
        Dtype scalar = - top_scalar / bottom_scalar;
        sum_scalar += scalar;
//        LOG(INFO) << "scalar" << scalar << " top " << top_scalar << " bottom " << bottom_scalar;
        if(scalar > 1){
          LOG(INFO) << "scalar" << scalar << " top " << top_scalar << " bottom " << bottom_scalar;
          scalar = 1;
        }
        int verb_offset = i*this->total_bbs*this->total_frames; //+ b*this->total_frames;
        int label_offset = i*this->references*this->maxlabel + r*this->maxlabel;
        int gold_verb = bottom_label[label_offset];
        Dtype bb_norm = norms[i*this->total_bbs + b];
        for(int f = 0; f < this->total_frames; f++){
          Dtype _verb_marginal = verb_marginal[verb_offset + f*this->total_bbs + b];
          if( f == gold_verb) LOG(INFO) << f << " " << b << " " << _verb_marginal; 
          int verb_grad_offset = verb_offset + b*this->total_frames + f;
          verb_grad[verb_grad_offset] = (r == 0 ? 0 : verb_grad[verb_grad_offset]  )  +  scalar * ( (f == gold_verb ? 1 : 0) - std::exp(_verb_marginal - bb_norm));   
          int arg_start = verb_start[f];
          int nargs = verb_length[f];
          for(int arg = 0; arg < nargs; arg++){
            int arg_index = arg + arg_start;
            Dtype value_bias = _verb_marginal - arg_marginal[i*this->total_args*this->total_bbs + arg_index*this->total_bbs + b] - bb_norm;
            Dtype* arg_grad = bottom[arg_index]->mutable_cpu_diff();
            const Dtype* arg_value = bottom[arg_index]->cpu_data();
            int arg_length = arg_structure[arg_index];
            int gold_value = bottom_label[label_offset+arg+1];
            int value_offset = i*this->total_bbs*arg_length + b*arg_length;
            for(int v = 0; v < arg_length; v++){
              Dtype update = scalar * ( (f == gold_verb && v == gold_value? 1 : 0 ) - std::exp( value_bias + arg_value[value_offset + v]));
              arg_grad[value_offset + v] = ( r == 0 ? 0 : arg_grad[value_offset + v] ) + update;             
            }    
          }        
        }
      }
    }
   //LOG(INFO) << "batch " << i << " sum scalar " << sum_scalar;
  }
}

INSTANTIATE_CLASS(MILFrameLossLayer);
REGISTER_LAYER_CLASS(MILFrameLoss);

}  // namespace caffe
