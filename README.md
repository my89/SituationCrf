# SituationCrf

This is an initial version of the CRF release.

To download supporting files:

  ./install.sh

To build this version of caffe:

  follow instructions for setting up dependencies: http://caffe.berkeleyvision.org/installation.html
  
  cd caffe;
  mkdir build;
  cd build;
  cmake ..;
  make -j8;
  cd ../..;

To get dev results:

 caffe/build/tools/caffe test --model of500_crf_1024/network.prototxt --weights of500_crf_1024/crf.caffemodel.h5 --iterations 504 -gpu 0
  
To view per verb summaries of current performance, browse html files in results/results_5/ (top-5) or results/results_25/ (top-25)

Note: I've merged this caffe fork with the version of caffe as of 9/18/16. It has weak cudnn v5 support (~1.8 speed up), and I will merge again when full support is added.
