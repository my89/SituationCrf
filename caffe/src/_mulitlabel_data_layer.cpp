#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
MultiLabelImageDataLayer<Dtype>::~MultiLabelImageDataLayer<Dtype>() {
  this->JoinPrefetchThread();
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

  //while (infile >> filename >> labels) {
  // lines_.push_back(std::make_pair(filename, label));
  // }

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
  if (this->layer_param_.multilabel_image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.multilabel_image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    new_height, new_width, is_color);
  const int channels = cv_img.channels();
  const int height = cv_img.rows;
  const int width = cv_img.cols;
  // image
  const int crop_size = this->layer_param_.transform_param().crop_size();
  const int batch_size = this->layer_param_.multilabel_image_data_param().batch_size();
  if (crop_size > 0) {
    top[0]->Reshape(batch_size, channels, crop_size, crop_size);
    this->prefetch_data_.Reshape(batch_size, channels, crop_size, crop_size);
    this->transformed_data_.Reshape(1, channels, crop_size, crop_size);
  } else {
    top[0]->Reshape(batch_size, channels, height, width);
    this->prefetch_data_.Reshape(batch_size, channels, height, width);
    this->transformed_data_.Reshape(1, channels, height, width);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
//  vector<int> label_shape(expected_labels, batch_size);
  top[1]->Reshape(batch_size, 1, 1, expected_labels);
  this->prefetch_label_.Reshape(batch_size, 1, 1, expected_labels);

  if(n_bounding_boxes > 1){
    top[2]->Reshape(batch_size*n_bounding_boxes , 1 , 1, 5);
    this->prefetch_bounding_box_.Reshape(batch_size*n_bounding_boxes, 1, 1, 5);
    LOG(INFO) << "bounding_boux count in resahpe" << this->prefetch_bounding_boxes_.count();
  }

  if(appended > 0){   
    int i = n_bounding_boxes > 1 ? 3 : 2;
    top[i]->Reshape(batch_size, 1, 1, appended);
    this->prefetch_append_.Reshape(batch_size, 1, 1, appended);
    LOG(INFO) << "append count in reshape " << this->prefetch_append_.count();
  }
 
}

template <typename Dtype>
void MultiLabelImageDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}
template <typename Dtype>
void MultiLabelImageDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
 this->JoinPrefetchThread();
  DLOG(INFO) << "Thread joined";
  // Reshape to loaded data.
  top[0]->Reshape(this->prefetch_data_.num(), this->prefetch_data_.channels(),
      this->prefetch_data_.height(), this->prefetch_data_.width());
  // Copy the data
  caffe_copy(this->prefetch_data_.count(), this->prefetch_data_.cpu_data(),
             top[0]->mutable_cpu_data());
  DLOG(INFO) << "Prefetch copied";
  if (this->output_labels_) {
    caffe_copy(this->prefetch_label_.count(), this->prefetch_label_.cpu_data(),
               top[1]->mutable_cpu_data());
  }

  if( this->prefetch_bounding_box_.count() > 0){
    caffe_copy(this->prefetch_bounding_box_.count(), this->prefetch_bounding_box_.cpu_data(), 
               top[2]->mutable_cpu_data());
  }

  if( this->prefetch_append_.count() > 0){
     int i = this->prefetch_bounding_box_.count() > 0 ? 3 : 2;
     caffe_copy(this->prefetch_append_.count(), this->prefetch_append_.cpu_data(),
               top[i]->mutable_cpu_data());
  }
  // Start a new prefetch thread
  DLOG(INFO) << "CreatePrefetchThread";
  this->CreatePrefetchThread();
}


// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void MultiLabelImageDataLayer<Dtype>::InternalThreadEntry() {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(this->prefetch_data_.count());
  CHECK(this->transformed_data_.count());
  MultiLabelImageDataParameter image_data_param = this->layer_param_.multilabel_image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const int crop_size = this->layer_param_.transform_param().crop_size();
  const bool is_color = image_data_param.is_color();
  const int max_labels = image_data_param.max_labels();
  const int references = image_data_param.references();
  const int appended = image_data_param.appended();
  const int expected_labels = max_labels*references;
  string root_folder = image_data_param.root_folder();
  const int n_bounding_boxes = image_data_param.n_bounding_boxes();
  
  string bounding_box_folder = image_data_param.bounding_box_folder();

  // Reshape on single input batches for inputs of varying dimension.
  if (batch_size == 1 && crop_size == 0 && new_height == 0 && new_width == 0) {
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
        0, 0, is_color);
    this->prefetch_data_.Reshape(1, cv_img.channels(),
        cv_img.rows, cv_img.cols);
    this->transformed_data_.Reshape(1, cv_img.channels(),
        cv_img.rows, cv_img.cols);
  }
  Dtype* prefetch_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* prefetch_label = this->prefetch_label_.mutable_cpu_data();
  Dtype* prefetch_append;
  Dtype* prefetch_bounding_box;
  if( appended > 0) prefetch_append = this->prefetch_append_.mutable_cpu_data();
  if( n_bounding_boxes > 1) prefetch_bounding_box = this->prefetch_bounding_box_.mutable_cpu_data();
  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first+".jpg",
        new_height, new_width, is_color);
    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = this->prefetch_data_.offset(item_id);
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
   
 
    if( n_bounding_boxes > 1 ) {
      std::ifstream infile(bounding_box_folder + lines_[lines_id_].first + ".bb");
      std::string line, filename;
      int line_num = 0;
      //skip the first line
      std::getline(infile,line);
      while (std::getline(infile, line)) {
        std::istringstream iss(line);
        iss >> filename;
        int bb_base = n_bounding_boxes*item_id*5 + line_num*5;
        int minx, miny, maxx, maxy;
        iss >> minx >> miny >> maxx >> maxy; 
        if( mirrored ) {
          maxx = new_width - minx - 1;
          minx = new_width - maxx - 1;
        }
        prefetch_bounding_box[bb_base] = line_num;
        prefetch_bounding_box[bb_base+1] = minx;
        prefetch_bounding_box[bb_base+2] = miny;
        prefetch_bounding_box[bb_base+3] = maxx;
        prefetch_bounding_box[bb_base+4] = maxy;
        line_num++;
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

template <typename Dtype>
void MultiLabelImageDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  this->JoinPrefetchThread();
  // Reshape to loaded data.
  top[0]->Reshape(this->prefetch_data_.num(), this->prefetch_data_.channels(),
      this->prefetch_data_.height(), this->prefetch_data_.width());
  // Copy the data
  caffe_copy(this->prefetch_data_.count(), this->prefetch_data_.cpu_data(),
      top[0]->mutable_gpu_data());
  if (this->output_labels_) {
    caffe_copy(this->prefetch_label_.count(), this->prefetch_label_.cpu_data(),
        top[1]->mutable_gpu_data());
  }  
  if( this->prefetch_append_.count() > 0){
     caffe_copy(this->prefetch_append_.count(), this->prefetch_append_.cpu_data(),
               top[2]->mutable_gpu_data());
  }
  // Start a new prefetch thread
  this->CreatePrefetchThread();
}

INSTANTIATE_CLASS(MultiLabelImageDataLayer);
REGISTER_LAYER_CLASS(MultiLabelImageData);
}  // namespace caffe
