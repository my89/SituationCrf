#include "caffe/loss_layers.hpp"
#include "assert.h"

namespace caffe {

template <typename Dtype> 
__device__ Dtype gpu_log_sum_exp(int length, int offset, int increment, const Dtype* data){
  int kmin = offset;
  int kmax = offset + length*increment;

  Dtype total = 0.;
  Dtype max = data[kmin];

  for(int k = kmin+increment; k < kmax; k+=increment){
    if (data[k] > max ) max = data[k];
  }
  for(int k = kmin; k < kmax; k+=increment){
    total += exp(data[k] - max);
  }
  //rv[0] = 
  return std::log(total) + max;
}

template <typename Dtype>
__global__ void MILFrameArgMarginal(const int nthreads,
   const Dtype* arg_data, Dtype* arg_marginal, 
   int total_args, int total_bbs, const int arg, const int length){ 
   CUDA_KERNEL_LOOP(index, nthreads) {
     //compute my argument index
     //each thread is tasked with computing one marginal. those are stored batch x arg x bb
     //index = (i*total_args*total_bb + a*total_bb + b)
     //int i = index / (total_args * total_bbs);
     //int a = (index / total_bbs ) % total_args;
     //int b = index % total_bbs;
     
     //index = i*total_bb + b
     int i = index / total_bbs;
     int b = index % total_bbs;
     int value_offset = i*total_bbs*length + b*length;
     int marginal_offset = i*total_args*total_bbs + arg*total_bbs + b;     

     arg_marginal[marginal_offset] = gpu_log_sum_exp(length, value_offset, 1, arg_data);
   }
} 

template <typename Dtype>
__global__ void MILFrameVerbMarginal(const int nthreads, 
   const Dtype* bottom_verb, const Dtype* arg_marginal, Dtype* verb_marginal,
   const Dtype* verb_start, const Dtype* verb_length,
   int total_frames, int total_bbs, int total_args){ 
   CUDA_KERNEL_LOOP(index, nthreads) {
     //index = i*total_frames*total_bbs + f*total_bbs + b
     int i = index / ( total_frames * total_bbs);
     int f = ( index / total_bbs ) % total_frames;
     int b = index % total_bbs;
     
     int verb_offset = i*total_bbs*total_frames + b*total_frames + f;
     int arg_offset = i*total_args*total_bbs + ((int)verb_start[f])*total_bbs + b;
     //int verb_marginal_offset = index;//i*total_frames*total_bbs + f*total_bbs + b;
     int nargs = (int)verb_length[f];     
 
     Dtype total = bottom_verb[verb_offset];
     for( int k = 0; k < nargs; k++){
       total+= arg_marginal[arg_offset + k*total_bbs];
     } 
     verb_marginal[index] = total;
  }
}

template <typename Dtype>
__global__ void MILFrameNorm(const int nthreads, 
   Dtype* norms, const Dtype* verb_marginal,
   int total_frames, int total_bbs){ 
   CUDA_KERNEL_LOOP(index, nthreads) {
     //index = i*total_bbs + b
     int i = index / total_bbs;
     int b = index % total_bbs;

     int verb_offset = i*total_bbs*total_frames + b;
     norms[index] = gpu_log_sum_exp(total_frames, verb_offset, total_bbs, verb_marginal);
   }
}

template <typename Dtype>
__global__ void MILFrameRefScore(const int nthreads, 
   const Dtype * score, Dtype* ref_scores,
   int gold_label, int i, int r, int length, int total_bbs, int references, bool clear){ 
   CUDA_KERNEL_LOOP(index, nthreads) {
     //index = i*references*total_bbs + r*total_bbs + b      
     //int i = index / (references * total_bbs);
     //int r = (index / total_bbs) % references;
     int b = index;// % total_bbs;
     int ref_offset = i*references*total_bbs + r*total_bbs + b;
      
     int value_index = i*total_bbs*length + b*length + gold_label;
     ref_scores[ref_offset] = (clear ? 0 :  ref_scores[ref_offset]) + score[value_index];
  } 
}

template <typename Dtype>
__global__ void MILFramePosScores(const int nthreads, 
   const Dtype * norms, const Dtype* ref_scores, Dtype * pos_scores,
   int total_bbs, int references){ 
   CUDA_KERNEL_LOOP(index, nthreads) {
     //index = i*references + r
     int i = index / references;
     int r = index % references;
     Dtype _not1 = log(0.);
     int ref_offset = i*references*total_bbs + r*total_bbs;
     int norm_offset = i*total_bbs;
     for( int b = 0; b < total_bbs; b++){
        Dtype ref = ref_scores[ref_offset + b];
        Dtype norm = norms[norm_offset + b];
        Dtype value = ref-norm;
        //_not1 = _not1 + value - _not1*value;
        if( _not1 > value)
           _not1 = log( 1 + exp(value - _not1) - exp(value)) + _not1;
        else
           _not1 = log( 1 + exp(_not1 - value) - exp(_not1)) + value;
     }
    //pos scores now stores log of p
    // _not1 = 1- log(v)
    pos_scores[index] = _not1;
   }
}


template <typename Dtype>
__global__ void MILFrameFillVerbMarginal(const int nthreads, 
  const Dtype* verb_marginal, const Dtype* norms, Dtype* output_scores, Dtype* output_structure,
  int maxlabel, int total_frames, int total_bbs){
  CUDA_KERNEL_LOOP(index, nthreads) {
    //index = i*total_frames + v 
    int i = index / total_frames;
    int v = index % total_frames;
    int offset = i*total_frames*total_bbs + v*total_bbs;
    int norm_offset = i*total_bbs;
    int score_offset = i*total_frames + v;
    int structure_offset = i*total_frames*maxlabel + v*maxlabel;
 
    Dtype not1 = 0.;
    for(int b = 0; b < total_bbs; b++){
      Dtype value =  exp(verb_marginal[offset + b] - norms[norm_offset + b]);
      not1 = not1 + value - not1*value;       
    }
    output_scores[score_offset] = not1;
    output_structure[structure_offset] = v;
  }
}

template <typename Dtype>
__global__ void MILFrameFillScratch(const int nthreads, 
  const Dtype * arg_marginal, const Dtype * values, Dtype* scratch,  
  int arg, int length, int total_args, int total_bbs, int max_value){
  
  CUDA_KERNEL_LOOP(index, nthreads) {
    //index = i*arg_length + v    
    int i = index / length;
    int v = index % length;
    int value_offset = i*total_bbs*length + v;    
    int marginal_offset = i*total_args*total_bbs + arg*total_bbs;
    
    Dtype not1 = 0.;
      //we need to look up the value for all bbs
    for(int b = 0; b < total_bbs; b++){
      Dtype value = exp(values[value_offset + b*length] -  arg_marginal[marginal_offset + b]);
      not1 = not1 + value - value*not1;      
    }
    scratch[i*total_args*max_value + arg*max_value + v] = not1;
  }
}

template <typename Dtype>
__global__ void MILFrameScratchMaxValue(const int nthreads, 
  const Dtype * scratch,const Dtype* arg_length,  Dtype* max_scratch, 
  int total_args, int max_value){
  CUDA_KERNEL_LOOP(index, nthreads) {
    //index = i*total_arguments + a
    int i = index / total_args;
    int a = index % total_args;

    int value_offset = i*total_args*max_value + a*max_value;
    int length = (int)arg_length[a];

    Dtype mv = 0.;
    int mi = 0;
    for(int v = 0; v < length; v++){
      Dtype value = scratch[value_offset + v];
      if( value >= mv) {
        mv = value;
        mi = v;
      }
    }   
    max_scratch[index]  = mi;
  }
} 

template <typename Dtype>
__global__ void MILFrameBackward(const int nthreads,
  const Dtype* total_scores, const Dtype * ref_scores,const Dtype* norms, const Dtype * verb_marginal, const Dtype * labels, Dtype * diff,
  int maxlabel, int total_frames, int total_bbs, int references){
  CUDA_KERNEL_LOOP(index, nthreads) {
    //index = i*total_bbs*total_frames + b*total_frames + f
    int i = index / (total_bbs * total_frames);
    int b = ( index / total_frames) % total_bbs;
    int f = index % total_frames;
    Dtype bb_norm = norms[i*total_bbs + b];    
    Dtype image_score = total_scores[i];    
    Dtype _verb_marginal = verb_marginal[i*total_bbs*total_frames + f*total_bbs + b];  
    Dtype g = 0.;
    for(int r = 0; r < references; r++){ 
      int gold_verb = labels[i*references*maxlabel + r*maxlabel];
      Dtype score = ref_scores[i*total_bbs*references + r*total_bbs + b] - bb_norm;
      //p(all zero) * (p(x = 1) / (p(x = 0))* p( at least 1))   
      //Dtype top_scalar = exp(log(1-exp(image_score)) - log(1 - exp(score)) + score);
      //Dtype bottom_scalar = exp(image_score);
      //Dtype scalar = -top_scalar/ bottom_scalar;
      Dtype scalar = - exp(log(1-exp(image_score)) - image_score + score - log(1 - exp(score)));
      if( scalar != scalar ) scalar = 0;
      if( image_score == 0 && score == 0 ) scalar = 0;  
      //if( scalar > 1 ) scalar = 1;      
      g += scalar * ( (f == gold_verb ? 1 : 0) - exp(_verb_marginal - bb_norm) );
    }
    diff[index] = g;
  }
}

template <typename Dtype>
__global__ void MILArgBackward(const int nthreads, 
  const Dtype* total_scores, const Dtype * ref_scores, const Dtype* norms, const Dtype * verb_marginal, const Dtype* arg_marginal, const Dtype * labels, const Dtype * values, Dtype * diff, 
  int maxlabel, int total_frames, int total_args, int total_bbs, int references, int arg_length, int f, int a, int arg){
  CUDA_KERNEL_LOOP(index, nthreads) {
    //index = i*total_bbs*arg_length + b*arg_length + v
    int i = index / (total_bbs * arg_length);
    int b = (index / arg_length) % total_bbs;
    int v = index % arg_length;
    
    Dtype bb_norm = norms[i*total_bbs + b];    
    Dtype image_score = total_scores[i];    
    Dtype _verb_marginal = verb_marginal[i*total_bbs*total_frames + f*total_bbs + b];  
    Dtype _arg_marginal = arg_marginal[i*total_bbs*total_args + a*total_bbs + b];
    Dtype value_score = values[index]; 
    int label_offset = i*references*maxlabel; 
    Dtype expected = exp( _verb_marginal - _arg_marginal + value_score - bb_norm);    

    Dtype g = 0.;
    for(int r = 0; r < references; r++){
      int gold_verb = labels[label_offset + r*maxlabel];
      int gold_value = labels[label_offset + r*maxlabel + arg + 1];

      Dtype score = ref_scores[i*total_bbs*references + r*total_bbs + b] - bb_norm;
      //Dtype top_scalar = exp(log(1-exp(image_score)) - log(1 - exp(score)) + score);
      //Dtype bottom_scalar = exp(image_score);
      //Dtype scalar = -top_scalar/ bottom_scalar;
      Dtype scalar = - exp(log(1-exp(image_score)) - image_score + score - log(1 - exp(score)));
      if( scalar != scalar ) scalar = 0;
      if( image_score == 0 && score == 0 ) scalar = 0;  
      //if( scalar > 1 ) scalar = 1; 
     
      g+= scalar * ( ( f == gold_verb && v == gold_value ? 1 : 0 ) - expected);
    }   
    diff[index] = g;    
  } 

}

template <typename Dtype>
void MILFrameLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int batch_size = bottom[this->label_index]->shape(0);

 // LOG(INFO) << "MIL START FORWARD " << batch_size << " " << total_bbs; 
  //compute arg marginal
  int am_jobs = batch_size*total_bbs;
  for(int a = 0; a < total_args; a++){  
    MILFrameArgMarginal<Dtype><<<CAFFE_GET_BLOCKS(am_jobs), CAFFE_CUDA_NUM_THREADS>>>(am_jobs, 
      bottom[a]->gpu_data(), arg_marginal.mutable_gpu_data(), 
      total_args, total_bbs, a, arg_structure[a]);
    CUDA_POST_KERNEL_CHECK;
  } 
  //compute verb marginal
  int vm_jobs = batch_size*total_frames*total_bbs;
  MILFrameVerbMarginal<Dtype><<<CAFFE_GET_BLOCKS(vm_jobs), CAFFE_CUDA_NUM_THREADS>>>(
      vm_jobs, bottom[verb_index]->gpu_data(), arg_marginal.gpu_data(),verb_marginal.mutable_gpu_data(), 
      b_verb_start.gpu_data(), b_verb_length.gpu_data(),
      total_frames, total_bbs, total_args);
  CUDA_POST_KERNEL_CHECK;
  //compute the norm 
  int n_jobs = batch_size*total_bbs;
  MILFrameNorm<Dtype><<<CAFFE_GET_BLOCKS(n_jobs), CAFFE_CUDA_NUM_THREADS>>>(
      n_jobs, norms.mutable_gpu_data(), verb_marginal.gpu_data(), 
      total_frames, total_bbs);
  CUDA_POST_KERNEL_CHECK;
  //compute ref scores... very irritating way to avoid sending a pointer to the args
  //honestly, for low bbs setting, this has extremely low parallism and likely
  //just avoids copying the previous layer to cpu.
  int r_jobs = total_bbs;
  const Dtype* label = bottom[label_index]->cpu_data();
  for(int i = 0; i < batch_size; i++){
    for(int r = 0; r < references; r++){
      int label_offset = i*maxlabel*references + r*maxlabel;
      int gold_verb = label[label_offset];
      MILFrameRefScore<Dtype><<<CAFFE_GET_BLOCKS(r_jobs), CAFFE_CUDA_NUM_THREADS>>>(
          r_jobs, bottom[verb_index]->gpu_data(), ref_scores.mutable_gpu_data(),
          gold_verb, i, r, total_frames, total_bbs, references, true);
      CUDA_POST_KERNEL_CHECK;
      int vlength = verb_length[gold_verb];
      int arg_offset = verb_start[gold_verb];
      for(int a = 0; a < vlength; a++){        
          int arg_index = arg_offset + a;
          int arg_length = arg_structure[arg_index];
          int gold_value = label[label_offset + a + 1];
          MILFrameRefScore<Dtype><<<CAFFE_GET_BLOCKS(r_jobs), CAFFE_CUDA_NUM_THREADS>>>(
            r_jobs, bottom[arg_index]->gpu_data(), ref_scores.mutable_gpu_data(),
            gold_value, i, r, arg_length, total_bbs, references, false);
          CUDA_POST_KERNEL_CHECK;
       }
    }
  }
  //compute positive scores
  int p_jobs = batch_size * this->references;
  MILFramePosScores<Dtype><<<CAFFE_GET_BLOCKS(p_jobs), CAFFE_CUDA_NUM_THREADS>>>(
    p_jobs,
    norms.gpu_data(), ref_scores.gpu_data(), pos_scores.mutable_gpu_data(),
    total_bbs, references);
  CUDA_POST_KERNEL_CHECK;
  
  const Dtype* pos_scores = this->pos_scores.cpu_data();
  Dtype* total_scores = this->total_scores.mutable_cpu_data();
  Dtype total_score = 0;
  for(int i = 0; i < batch_size; i++){
    Dtype not1 = log(0.);
    for(int r = 0; r < references; r++){
      Dtype value = pos_scores[i*references + r]; //pos scores [ i] = v where p(1 -prod( log(v) )
      //LOG(INFO) << i << "," << r << ":" << value;
      if(value > 0 ) LOG(INFO) << "POS SCORE PROB GREATER THAN 1:" << value;
      if( value != value) LOG(INFO) << "NAN value:" << value; 
      if( not1 > value)
          not1 = log( 1 + exp(value - not1) - exp(value)) + not1;
      else
          not1 = log( 1 + exp(not1 - value) - exp(not1)) + value;
     // not1 = not1 + value - not1*value;
    }
    //not1 = std::log(not1);
    if(not1 != not1) LOG(INFO) << "NOT 1 NAN";
    if(not1 > 0){
      LOG(INFO) << "NOT1 PROB GREATER THAN 1:" << not1;
      not1 = 0;
    }
    total_score += not1;
    total_scores[i] = not1; 
  }
  if(total_score != total_score) LOG(INFO) << "Total score nan" << total_score; 
  top[0]->mutable_cpu_data()[0] = (total_score)/batch_size;
  //LOG(INFO) << "MIL END FORWARD"; 
  if(this->mode == -1) return; //we aren't going to do inference, and settle in for just probabilty
  if(this->mode == 0) this->PredictMarginalFrameGPU(bottom, top);
}

template <typename Dtype> 
void MILFrameLossLayer<Dtype>::PredictMarginalFrameGPU(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){  
  //LOG(INFO) << "MIL START PREDICT";
  int batch_size = bottom[this->label_index]->shape(0);

  //LOG(INFO)<< "verb marginal...";
  //get the bb marginal in  
  int v_jobs = batch_size*total_frames; 
  MILFrameFillVerbMarginal<Dtype><<<CAFFE_GET_BLOCKS(v_jobs), CAFFE_CUDA_NUM_THREADS>>>( v_jobs,
    verb_marginal.gpu_data(), norms.gpu_data(), top[2]->mutable_gpu_data(), top[1]->mutable_gpu_data(), 
    maxlabel, total_frames, total_bbs);
  CUDA_POST_KERNEL_CHECK;

  //LOG(INFO)<< "done.\n" << "scratch fill...";
  //compute the pre value marginal
  for(int a = 0; a < total_args; a++){  
    int s_jobs = batch_size * arg_structure[a];  
    MILFrameFillScratch<Dtype><<<CAFFE_GET_BLOCKS(s_jobs), CAFFE_CUDA_NUM_THREADS>>>(s_jobs,
      arg_marginal.gpu_data(), bottom[a]->gpu_data(), scratch.mutable_gpu_data(),
      a, arg_structure[a], total_args, total_bbs, max_value);
    CUDA_POST_KERNEL_CHECK;
  }
  
  //LOG(INFO) << "done.\n" << "max value...";
  int m_jobs = batch_size*total_args;
  //compute the max over the marginal
  MILFrameScratchMaxValue<Dtype><<<CAFFE_GET_BLOCKS(m_jobs), CAFFE_CUDA_NUM_THREADS>>>(m_jobs,
     scratch.gpu_data(), b_arg_structure.gpu_data(), max_scratch.mutable_gpu_data(),
     total_args, max_value);
  CUDA_POST_KERNEL_CHECK;
  //LOG(INFO) << "done.";
 //this could be on gpu, but we need the actual output back anyways.
  const Dtype* max_scratch = this->max_scratch.cpu_data();
  Dtype* structure_output = top[1]->mutable_cpu_data();
  //we need to copy max data to output
  for( int i = 0; i < batch_size; i++){
    for(int f = 0; f < total_frames; f++){
      int total_arg = verb_length[f];
      int arg_offset = verb_start[f];   
      int offset = i*total_frames*maxlabel + f*maxlabel;
      int arg_max_offset = i*total_args;
      for( int a = 0; a < total_arg; a++){
        int arg_index = arg_offset + a; 
        structure_output[offset + a + 1] = max_scratch[arg_max_offset + arg_index];
      }
    }
  } 
  //LOG(INFO) << "MIL END PREDICT";
}

template <typename Dtype>
void MILFrameLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  
  //LOG(INFO) << "BACKWARD START";
  int batch_size = bottom[this->label_index]->shape(0);
  const Dtype * labels = bottom[label_index]->gpu_data();
  
  int f_jobs = batch_size * total_bbs * total_frames; 
  MILFrameBackward<Dtype><<<CAFFE_GET_BLOCKS(f_jobs), CAFFE_CUDA_NUM_THREADS>>>(f_jobs,
    total_scores.gpu_data(), ref_scores.gpu_data(), norms.gpu_data(), verb_marginal.gpu_data(), labels, bottom[verb_index]->mutable_gpu_diff(), 
    maxlabel, total_frames, total_bbs, references);
  CUDA_POST_KERNEL_CHECK;
/*
  const Dtype * vdiff = bottom[verb_index]->cpu_diff();
  for(int i = 0 ; i < f_jobs; i++){
    if(vdiff[i] > 1.0 || vdiff[i] < -1.0) 
      LOG(INFO) << "VDIFF ERROR: " << vdiff[i]; 
  } 
*/
  for(int f = 0; f < total_frames; f++){
    int arg_offset = verb_start[f];
    for( int arg = 0; arg < verb_length[f]; arg++){
      int arg_index = arg_offset + arg;
      int arg_length = arg_structure[arg_index];
      int a_jobs = batch_size * total_bbs * arg_length;
      MILArgBackward<Dtype><<<CAFFE_GET_BLOCKS(a_jobs), CAFFE_CUDA_NUM_THREADS>>>(a_jobs,
        total_scores.gpu_data(), ref_scores.gpu_data(),norms.gpu_data(), verb_marginal.gpu_data(), arg_marginal.gpu_data(), labels, bottom[arg_index]->gpu_data(), bottom[arg_index]->mutable_gpu_diff(),
        maxlabel, total_frames, total_args, total_bbs, references, arg_length, f, arg_index, arg);
      CUDA_POST_KERNEL_CHECK;  
 /*
     const Dtype * vdiff = bottom[arg_index]->cpu_diff();
     for(int i = 0 ; i < a_jobs; i++){
       if(vdiff[i] > 1.0 || vdiff[i] < -1.0) { LOG(INFO) << "ADIFF ERROR: " << arg_index  << " " << vdiff[i];}
     } 
*/
    }    
  }
 //LOG(INFO) << "BACKWARD END";
}

template void MILFrameLossLayer<float>::PredictMarginalFrameGPU( \
      const std::vector<Blob<float>*>& bottom, \
      const std::vector<Blob<float>*>& top); \
template void MILFrameLossLayer<double>::PredictMarginalFrameGPU( \
      const std::vector<Blob<double>*>& bottom, \
      const std::vector<Blob<double>*>& top);

INSTANTIATE_LAYER_GPU_FUNCS(MILFrameLossLayer);

}
