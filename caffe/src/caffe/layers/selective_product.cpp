#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SelectiveProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  SelectiveProductParameter param = this->layer_param_.selective_product_param();
  D_ = param.dim();
  const string& pair_file_name = param.pair_file();
  const string& init_file_name = param.init_file();
  const float init_scalar = param.init_file_scalar();
  bias_ = param.bias(); //if true use the bias in forward and backward 
  scalar_ = param.scalar();   
  
  //we will always allocate space.  
  w_index_ = 0;
  b_index_ = 1;
  s_index_ = 2;

  total_blobs_ = 3;
  
  std::string line;
  LOG(INFO) << "Opening File " << pair_file_name.c_str();
  std::ifstream pair_file(pair_file_name.c_str());

  std::vector<vector<int> > pairs;
  std::vector<vector<int> > triplets;
  std::vector<int> triplet_pair;

  int max_i = 0;
  int max_p = 0;;
  int max_o = 0;
  int linen = 0;  
  while( std::getline(pair_file, line)){
    //LOG(INFO) << linen;
    //LOG(INFO) << line;
    linen++;
    int i, p, o;
    std::istringstream iss(line);
    iss >> i >> p >> o;
    int x2[2] = {i,p};
    int x3[3] = {i,p,o};
    std::vector<int> _p (x2, x2 + sizeof(x2)/sizeof(int));
    std::vector<int> _t (x3, x3 + sizeof(x3)/sizeof(int));
    triplets.push_back(_t);
    if( i > max_i) max_i = i;
    if( p > max_p) max_p = p;
    if( o > max_o) max_o = o;
      //check if p is already in pairs
    bool found = false;
    int k = pairs.size();
      /*for(k = 0; k < pairs.size(); k++){
        vector<int> _v = pairs[k];
        found = true;
        for(int x = 0; x < _v.size(); x++){
          if(_v[x] != x2[x]) found = false;
        } 
        if(found) break;
      }*/
    if(!found) pairs.push_back(_p);
    triplet_pair.push_back(k);
  }
  for(int i = 0; i < max_i; i++){
    int iv = 0;
    for(int k = 0 ; k < triplets.size(); k++){
      if(triplets[k][0] == i) iv++;
    }
    LOG(INFO)<< i << " " << iv;
  }
    

  LOG(INFO) << "done reading";
  I_ = max_i+1;
  N_ = max_p+1;
  P_ = pairs.size(); 
  T_ = triplets.size();
  
  CHECK_EQ(max_o+1, T_) << "repeated output assignments are not allowed, and every output must be assigned."; 
 
  vector<int> output_shape(1,T_); 
  output_.Reshape(output_shape);
  output_id_.Reshape(output_shape);
  int* _output_id_ = output_id_.mutable_cpu_data();
  int* _output_ = output_.mutable_cpu_data();
  for(int i = 0; i < T_; i++){
    //find the correct index item in pairs
    _output_id_[i] = triplet_pair[i];
    _output_[i] = triplets[i][2];
  } 
 
  vector<int> i_shape(1, P_);
  i_input_.Reshape(i_shape);
  i_param_.Reshape(i_shape);
  i_inner_products_.Reshape(i_shape);
  int* _i_input_ = i_input_.mutable_cpu_data();
  int* _i_param_ = i_param_.mutable_cpu_data();
  for(int i = 0; i < P_; i++){
    _i_input_[i] = pairs[i][0];
    _i_param_[i] = pairs[i][1];
  }
  LOG(INFO) << "N=" << N_  << " D=" << D_ << " P= " << P_ << " T=" << T_;
  //setup the param array  
  if(this->blobs_.size() == total_blobs_){ 
     LOG(INFO) << "Skipping param init";
  }
  else{
     this->blobs_.resize(total_blobs_);
     
     vector<int> w_shape(2);
     w_shape[0] =1;// N_;
     w_shape[1] =1;//D_; 
     this->blobs_[w_index_].reset(new Blob<Dtype>(w_shape));

     vector<int> b_shape(1,T_);
     this->blobs_[b_index_].reset(new Blob<Dtype>(b_shape));
    
     vector<int> s_shape(1,P_);
     this->blobs_[s_index_].reset(new Blob<Dtype>(s_shape));

     shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(param.weight_filler()));
     weight_filler->Fill(this->blobs_[w_index_].get());

     shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(param.bias_filler()));
     bias_filler->Fill(this->blobs_[b_index_].get());

     shared_ptr<Filler<Dtype> > scalar_filler(GetFiller<Dtype>(param.scalar_filler()));
     scalar_filler->Fill(this->blobs_[s_index_].get());

     if( init_file_name.length() > 0){
       LOG(INFO) << "PRE INIT VECTORS";
       std::ifstream init_file(init_file_name.c_str());
       std::string _init;
       Dtype* vectors = this->blobs_[w_index_]->mutable_cpu_data();
       while( std::getline(init_file, _init)){
         std::istringstream iss(_init);
         int index;
         Dtype value;
         int offset = 0;
         iss >> index;
         while( iss >> value){
	   vectors[index*D_ + offset] = value * init_scalar;
           offset++;
         }
       }
     }
     
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);
/* 
  vector<int> incidenceO(T_,0);
  vector<int> incidenceP(N_,0);
  vector<int> incidenceI(I_,0);
  for(int i = 0; i < P_; i++){
    incidenceI[triplets[i][0]] += 1;
    incidenceP[triplets[i][1]] += 1;
    incidenceO[triplets[i][2]] += 1;
  }
  output_max_ = param_max_ = input_max_ = 0;
  for(int i = 0; i < P_; i++){
    if(incidenceI[i] > input_max_) input_max_ = incidenceI[i];
    if(incidenceP[i] > param_max_) param_max_ = incidenceP[i];
    if(incidenceO[i] > output_max_) output_max_ = incidenceO[i];
  }
  output_pindex_.Reshape(1,1,T_, output_max_);
  output_iindex_.Reshape(1,1,T_, output_max_);
  output_kindex_.Reshape(1,1,T_, output_max_);  

  param_oindex_.Reshape(1,1,N_,param_max_);
  param_iindex_.Reshape(1,1,N_,param_max_);
  param_kindex_.Reshape(1,1,N_,param_max_);

  input_oindex_.Reshape(1,1,I_,input_max_);
  input_pindex_.Reshape(1,1,I_,input_max_);
  input_kindex_.Reshape(1,1,I_,input_max_);
 
  int* _o_pi = output_pindex_.mutable_cpu_data();
  int* _o_ii = output_iindex_.mutable_cpu_data();
  int* _o_ki = output_kindex_.mutable_cpu_data();
  int* _i_oi = input_oindex_.mutable_cpu_data();
  int* _i_pi = input_pindex_.mutable_cpu_data();
  int* _i_ki = input_kindex_.mutable_cpu_data();
  int* _p_oi = param_oindex_.mutable_cpu_data();
  int* _p_ii = param_iindex_.mutable_cpu_data();
  int* _p_ki = param_kindex_.mutable_cpu_data();

  for(int k = 0; k < T_*output_max_; k++) _o_pi[k] = _o_ii[k] = _o_ki[k] = -1;
  for(int k = 0; k < I_*input_max_; k++) _i_oi[k] = _i_pi[k] = _i_ki[k] =  -1;
  for(int k = 0; k < N_*param_max_; k++) _p_oi[k] = _p_ii[k] = _p_ki[k] = -1;


  vector<int> i_offset(I_,0);
  vector<int> p_offset(N_,0);
  vector<int> o_offset(T_,0);

  for (int k = 0; k < P_; k++){
    int i = triplets[k][0];
    int p = triplets[k][1];
    int o = triplets[k][2];
    
    int _i = i_offset[i]++;
    int _p = p_offset[p]++;
    int _o = o_offset[o]++;

    int ii = i*input_max_ + _i;
    int pp = p*param_max_ + _p;
    int oo = o*output_max_ + _o;

    _o_pi[oo] = p;
    _o_ii[oo] = i;
    _o_ki[oo] = k;
  
    _i_oi[ii] = o;
    _i_pi[ii] = p;
    _i_ki[ii] = k;
  
    _p_oi[pp] = o;
    _p_ii[pp] = i;
    _p_ki[pp] = k;
  } 
*/
}

template <typename Dtype>
void SelectiveProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  int batches = bottom[0]->shape(0);
  int in_size = bottom[0]->shape(1);

  CHECK_EQ(in_size/D_, I_) << "input size should equal input size in file";
  LOG(INFO) << "T_ = " << T_ << " P_ = " << P_ ; 
 
  vector<int> top_shape(2);
  top_shape[0] = batches;
  top_shape[1] = T_;
  //top_shape[2] = 1;
  //top_shape[3] = 1;
  top[0]->Reshape(top_shape);   

  vector<int> inner_shape(2);
  inner_shape[0] = batches;
  inner_shape[1] = P_;
  i_inner_products_.Reshape(inner_shape);
}

template <typename Dtype>
void SelectiveProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  //precompute the inner products
  const Dtype* _input = bottom[0]->cpu_data();
  const Dtype* _vec_scalar;
  if(bottom.size() >= 3) _vec_scalar = bottom[2]->cpu_data();

  const Dtype* _param = this->blobs_[w_index_]->cpu_data();
  const Dtype* _scalar = this->blobs_[s_index_]->cpu_data();
  Dtype* _results = i_inner_products_.mutable_cpu_data();  
  Dtype* _top_results = top[0]->mutable_cpu_data();

  const int* _i_input_ = i_input_.cpu_data();
  const int* _i_param_ = i_param_.cpu_data();  
  const int* _output_ = output_.cpu_data();
  int batches = bottom[0]->shape(0);
  
  LOG(INFO) << "_i_input c " << i_input_.count();
  LOG(INFO) << "_i_param c " << i_param_.count();
  LOG(INFO) << "P_ " << P_; 
  // clear the output layer
  for(int b = 0; b < batches; b++){
    for(int k = 0; k < T_; k++){
      _top_results[b*T_ + k] = 0;
    }
  } 
 
  for(int b = 0; b < batches; b++){ 
    for(int k = 0; k < P_; k++){
      int i = _i_input_[k];
      int p = _i_param_[k];
    
      int offset_i = b*I_*D_ + i*D_;
      int offset_p = p*D_;
      Dtype total = 0;
      //for(int d = 0; d < D_; d++) total+= _input[offset_i + d]*_param[offset_p + d];
      //if(scalar_) total *= _scalar[_output_[k]];
      if(bottom.size() >= 3){
        total+= _vec_scalar[b*N_ + i];
      }
      _top_results[b*T_ + _output_[k]] += total;   
    }
  }
  /*LOG(INFO) << "inners done";
  //copy the results to the appropriate layers, adding the bias
  for(int b = 0; b < batches; b++){
    for(int k = 0; k < T_; k++){
      _top_results[b*T_ + _output_[k]] = _results[b*P_ + k]; 
    }
  }
  LOG(INFO) << "done copy";
  */
  //add in bias
  if(bottom.size() >= 2){
//    const Dtype* bias = bottom[1]->cpu_data();
//    for(int b = 0; b < batches; b++){
//      for(int k = 0; k < T_; k++){
//        _top_results[b*T_ + k] += bias[b*T_ + k];
//      }
//    }
//    LOG(INFO) << "done dynamic bias (F):" << bias[0];
  }
  
  if(bias_){
//    const Dtype* bias = this->blobs_[b_index_]->cpu_data();
//    for(int b = 0; b < batches; b++){
//      for(int k = 0; k < T_; k++){
//        _top_results[b*T_ + k] += bias[k];
//      }
//    }
//    LOG(INFO) << "done static bias (F):" << bias[0];
  }
}

template <typename Dtype>
void SelectiveProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  //make sure to reset our gradients 
  Dtype* _input_grad = bottom[0]->mutable_cpu_diff();
  Dtype* _param_grad = this->blobs_[w_index_]->mutable_cpu_diff();
  Dtype* _scalar_grad = this->blobs_[s_index_]->mutable_cpu_diff(); 
  Dtype* _bias_grad = this->blobs_[b_index_]->mutable_cpu_diff();
  
  const Dtype* _input = bottom[0]->cpu_data();
  const Dtype* _param = this->blobs_[w_index_]->cpu_data();  
  const Dtype* _scalar = this->blobs_[s_index_]->cpu_data();
  const Dtype* _bias = this->blobs_[b_index_]->cpu_data();
  const Dtype* _output_grad = top[0]->cpu_diff();
  const Dtype* _output_data = top[0]->cpu_data();
  
  const int batches = bottom[0]->shape(0);

  LOG(INFO) << "batch:" << batches; 
  for(int k = 0; k < bottom[0]->count(); k++) _input_grad[k] = 0;
  for(int k = 0; k < this->blobs_[w_index_]->count(); k++) _param_grad[k] = 0;
  for(int k = 0; k < this->blobs_[s_index_]->count(); k++) _scalar_grad[k] = 0;
  LOG(INFO) << "done clear " << bottom[0]->count();
 
  if(bias_){
 //   for(int b = 0; b < batches; b++){
    //compute the grad for each output
 //     for(int k = 0; k < T_; k++){
 //       _bias_grad[k] = _output_grad[b*T_ + k];    
 //     }  
 //   }
 //   LOG(INFO) << "done static bias";
  }
  
  if(bottom.size() >= 2){
 //   Dtype* _s_bias_grad = bottom[1]->mutable_cpu_diff();
 //   for(int b = 0; b < batches; b++){
    //compute the grad for each output
 //     for(int k = 0; k < T_; k++){
 //       _s_bias_grad[b*T_ + k] = _output_grad[b*T_ + k];    
 //     }  
 //   }
//    LOG(INFO) << "done dynamic bias:" << _s_bias_grad[0];
  }
 
  const int* _i_input_ = i_input_.cpu_data();
  const int* _i_param_ = i_param_.cpu_data();
  const int* _output_ = output_.cpu_data();

  const Dtype* _bias_input;
  if(bottom.size() >= 2) _bias_input = bottom[1]->cpu_data();
/*  for(int b = 0; b < batches; b++){
    for(int k = 0; k < P_; k++){
      int i = _i_input_[k];
      int p = _i_param_[k];
      Dtype _o_grad = _output_grad[b*T_ + _output_[k]];
      if(scalar_) _o_grad *= _scalar[_output_[k]]; 
      int offset_i = b*I_*D_ + i*D_;
      int offset_p = p*D_; 
      for(int d = 0; d < D_; d++){
        _input_grad[offset_i + d] += _o_grad*_param[offset_p + d];
        _param_grad[offset_p + d] += _o_grad*_input[offset_i + d];
      }      
    }
  }
*/
  if(bottom.size() >= 3){
    Dtype* _vec_bias = bottom[2]->mutable_cpu_diff();
    for(int k = 0; k < bottom[2]->count(); k++) _vec_bias[k] = 0;

    for(int b = 0; b < batches; b++){
      for(int k = 0; k < P_; k++){
        int i = _i_input_[k];
        _vec_bias[b*N_ + i] +=  _output_grad[b*T_ + _output_[k]];
      }
    }  
    LOG(INFO) << "vec bias done " << _vec_bias[0]; 
  }

/*  if(scalar_){
    for(int b = 0; b < batches; b++){
      for(int k = 0; k < T_; k++){
          Dtype update = _output_data[b*T_ + k]; 
          if(bias_) update -= _bias[k];
          if(bottom.size() >= 2) update -= _bias_input[b*T_ + k]; 
          _scalar_grad[k] +=  _output_grad[b*T_ + k]*update;
      }
    }
  }
*/
  LOG(INFO) << "param_grad:" << _param_grad[0] << " param:" << _param[0] << " scalar:" << _scalar[0]  << " bias:" << _bias[0] << " output:" << _output_data[0] << " output_grad:" << _output_grad[0] <<  " scalar_grad:" << _scalar_grad[0];
}

#ifdef CPU_ONLY
STUB_GPU(SelectiveProductLayer);
#endif

INSTANTIATE_CLASS(SelectiveProductLayer);
REGISTER_LAYER_CLASS(SelectiveProduct);

}//namespace caffe
