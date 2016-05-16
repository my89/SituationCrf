#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
MultiLabelImageDataLayer<Dtype>::~MultiLabelImageDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void MultiLabelImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  MultiLabelImageDataParameter image_data_param = this->layer_param_.multilabel_image_data_param();
  const int new_height = image_data_param.new_height();
  const int new_width  = image_data_param.new_width();
  const bool is_color  = image_data_param.is_color();
  const int max_labels = image_data_param.max_labels();
  const int references = image_data_param.references();
  const int appended = image_data_param.appended();
  const int expected_labels = max_labels*references;
  const int n_bounding_boxes = image_data_param.n_bounding_boxes();
  
  string bounding_box_folder = image_data_param.bounding_box_folder();
  string root_folder = image_data_param.root_folder();
  

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.multilabel_image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename;
 
  std::string line;
  int line_num = 0;
  LOG(INFO) << "Expecting " << expected_labels << " labels.";
  while (std::getline(infile, line)) {
     int label;
     std::istringstream iss(line);
     iss >> filename;
     std::vector<int> labels;
     std::vector<int> appended_labels;
     while(iss >> label) {        
        if(labels.size() < expected_labels)
          labels.push_back(label);
        else{
          appended_labels.push_back(label);
        }
     }    

     if( labels.size()+appended_labels.size() != expected_labels + appended ){
     	LOG(INFO) << "Unexpected number of labels on line " << line_num 
                  << " of file " << source.c_str() << ": (" << labels.size()
                  << " vs. " << expected_labels << ". Skipping..."; 
     }
     else {
         filename.replace(filename.find(".jpg"), 4, "");
         lines_.push_back(std::make_pair(filename, std::make_pair(labels, appended_labels)));
     }
     line_num += 1;
  }

  if (this->layer_param_.multilabel_image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "*A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first+".jpg",
                                    new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.multilabel_image_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";

  top_shape[0] = batch_size;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
    this->prefetch_[i].data_.mutable_cpu_data();
    this->prefetch_[i].data_.mutable_gpu_data();
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  top[1]->Reshape(batch_size, 1 ,1 , expected_labels);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(batch_size, 1, 1, expected_labels);
    this->prefetch_[i].label_.mutable_cpu_data();
    this->prefetch_[i].label_.mutable_gpu_data();
  }

  if( n_bounding_boxes >= 1){
    LOG(INFO) << "preallocating bbs";
    top[2]->Reshape(batch_size*n_bounding_boxes, 1 , 1, 5);
    for( int i = 0; i < this->PREFETCH_COUNT; ++i){
       this->prefetch_[i].bounding_boxes_.Reshape(batch_size*n_bounding_boxes, 1 , 1, 5);
       this->prefetch_[i].bounding_boxes_.mutable_cpu_data();
       this->prefetch_[i].bounding_boxes_.mutable_gpu_data();
    }
  }
  if(appended > 0){   
    int ix = n_bounding_boxes > 1 ? 3 : 2;
    top[ix]->Reshape(batch_size, 1, 1, appended);
    for(int i =0; i < this->PREFETCH_COUNT; ++i){
      this->prefetch_[i].append_.Reshape(batch_size, 1, 1, appended);
      this->prefetch_[i].append_.mutable_cpu_data();
      this->prefetch_[i].append_.mutable_gpu_data();
    }
  }
}

template <typename Dtype>
void MultiLabelImageDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

template <typename Dtype>
void  MultiLabelImageDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch<Dtype>* batch = this->prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
             top[0]->mutable_cpu_data());
  //DLOG(INFO) << "Prefetch copied";
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(batch->label_);
    // Copy the labels.
    caffe_copy(batch->label_.count(), batch->label_.cpu_data(),
        top[1]->mutable_cpu_data());
  }
  if(batch->bounding_boxes_.count()){
    top[2]->ReshapeLike(batch->bounding_boxes_);
    caffe_copy(batch->bounding_boxes_.count(), batch->bounding_boxes_.cpu_data(),
        top[2]->mutable_cpu_data());
  }
  if(batch->append_.count()){
    int idx = batch->bounding_boxes_.count() ? 3 : 2;
    top[idx]->ReshapeLike(batch->append_);
    caffe_copy(batch->append_.count(), batch->append_.cpu_data(), 
         top[idx]->mutable_cpu_data()); 
  }
  
  this->prefetch_free_.push(batch);
}

template <typename Dtype>
void  MultiLabelImageDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch<Dtype>* batch = this->prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
/*  const Dtype* _d = batch->data_.cpu_data();
  for(int i = 0; i < 100; i++){
    LOG(INFO) << i << ":" << _d[i];
  }
*/
  //LOG(INFO) << batch->data_.count();
  caffe_copy(batch->data_.count(), batch->data_.gpu_data(),
             top[0]->mutable_gpu_data());
  //LOG(INFO) << "Prefetch copied";
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(batch->label_);
    // Copy the labels.
    caffe_copy(batch->label_.count(), batch->label_.gpu_data(),
        top[1]->mutable_gpu_data());
  }
  if(batch->bounding_boxes_.count()){
//   LOG(INFO) << "bb..." << batch->bounding_boxes_.cpu_data()[0] << "," << batch->bounding_boxes_.cpu_data()[1] << "," << batch->bounding_boxes_.cpu_data()[2] << "," << batch->bounding_boxes_.cpu_data()[3] << "," << batch->bounding_boxes_.cpu_data()[4];
//   LOG(INFO) << "bb..." << batch->bounding_boxes_.cpu_data()[5] << "," << batch->bounding_boxes_.cpu_data()[6] << "," << batch->bounding_boxes_.cpu_data()[7] << "," << batch->bounding_boxes_.cpu_data()[8] << "," << batch->bounding_boxes_.cpu_data()[9];
//   LOG(INFO) << "bb..." << batch->bounding_boxes_.cpu_data()[10] << "," << batch->bounding_boxes_.cpu_data()[11] << "," << batch->bounding_boxes_.cpu_data()[12] << "," << batch->bounding_boxes_.cpu_data()[13] << "," << batch->bounding_boxes_.cpu_data()[14];
   top[2]->ReshapeLike(batch->bounding_boxes_);
    LOG(INFO) << batch->bounding_boxes_.count();
    caffe_copy(batch->bounding_boxes_.count(), batch->bounding_boxes_.gpu_data(),
        top[2]->mutable_gpu_data());
  }
  if(batch->append_.count()){
    int idx = batch->bounding_boxes_.count() ? 3 : 2;
    top[idx]->ReshapeLike(batch->append_);
    caffe_copy(batch->append_.count(), batch->append_.gpu_data(), 
         top[idx]->mutable_gpu_data()); 
  }
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  this->prefetch_free_.push(batch);
}
// This function is called on prefetch thread
template <typename Dtype>
void MultiLabelImageDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  MultiLabelImageDataParameter image_data_param = this->layer_param_.multilabel_image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const bool is_color = image_data_param.is_color();
  const int max_labels = image_data_param.max_labels();
  const int references = image_data_param.references();
  const int appended = image_data_param.appended();
  const int expected_labels = max_labels*references;
  const int n_bounding_boxes = image_data_param.n_bounding_boxes();
   
  string bounding_box_folder = image_data_param.bounding_box_folder();
  string root_folder = image_data_param.root_folder();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first + ".jpg",
      new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();
  Dtype* prefetch_append;
  Dtype* prefetch_bounding_box;
  if( appended > 0) prefetch_append = batch->append_.mutable_cpu_data();
  if( n_bounding_boxes >= 1) prefetch_bounding_box = batch->bounding_boxes_.mutable_cpu_data();
  
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    //LOG(INFO) <<  root_folder + lines_[lines_id_].first+".jpg";
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first+".jpg",
        new_height, new_width, is_color);
    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    bool mirrored = this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    trans_time += timer.MicroSeconds();
   
    int copy_base = item_id * expected_labels;
    for( int i = 0; i < expected_labels; i++){
        prefetch_label[copy_base + i] = lines_[lines_id_].second.first[i];
    }
    if( appended > 0){
      int append_base = item_id * appended;
      for( int i = 0; i < appended; i++){  
        prefetch_append[append_base + i] = lines_[lines_id_].second.second[i];
      }   
    }
   
    if( n_bounding_boxes >= 1 ) {
      std::ifstream infile((bounding_box_folder + lines_[lines_id_].first + ".bb").c_str());      
      std::string line, filename;
      int line_num = 0;
      //skip the first line
      std::getline(infile,line);
      while (std::getline(infile, line)) {
    //    LOG(INFO) << "reading";
        if( line_num >= n_bounding_boxes) break;
        std::istringstream iss(line);
        std::string s;   
        int minx, miny, maxx, maxy;
        std::getline(iss,filename , ',');
        std::getline(iss,s , ',');
        minx = atoi(s.c_str());
        std::getline(iss,s , ',');     
        miny = atoi(s.c_str());
        std::getline(iss,s , ','); 
        maxx = atoi(s.c_str());
        std::getline(iss,s , ',');
        maxy = atoi(s.c_str());
  
   //     LOG(INFO) << "filename: " << filename << " " <<  new_width << " " << new_height;
        int bb_base = n_bounding_boxes*item_id*5 + line_num*5;
//        iss >> minx >> miny >> maxx >> maxy; 
//        LOG(INFO) << bb_base << "," << item_id << "," << minx << "," << miny << "," << maxx << "," << maxy;
        if( mirrored ) {
          int _maxx = maxx;
          int _minx = minx;
          maxx = new_width - _minx - 1;
          minx = new_width - _maxx - 1;
        }
//        LOG(INFO) << bb_base << "," << item_id << "," << minx << "," << miny << "," << maxx << "," << maxy;
        prefetch_bounding_box[bb_base] = item_id;
        prefetch_bounding_box[bb_base+1] = minx;
        prefetch_bounding_box[bb_base+2] = miny;
        prefetch_bounding_box[bb_base+3] = maxx;
        prefetch_bounding_box[bb_base+4] = maxy;
        line_num++;
//        LOG(INFO) << "done";
      }
    } 
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.multilabel_image_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(MultiLabelImageDataLayer);
REGISTER_LAYER_CLASS(MultiLabelImageData);

}  // namespace caffe
#endif  // USE_OPENCV
