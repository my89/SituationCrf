curl https://s3.amazonaws.com/my89-frame-annotation/public/output_224.tar > of500/output_224.tar
curl https://s3.amazonaws.com/my89-frame-annotation/public/of500_images_resized.tar > of500/resized.tar
curl https://s3.amazonaws.com/my89-frame-annotation/public/OpenFrame500.tab > of500/OpenFrame500.tab
curl https://s3.amazonaws.com/my89-frame-annotation/public/crf.caffemodel.h5 > of500_crf_1024/crf.caffemodel.h5 

cd of500
tar -xvf output_224.tar 
tar -xvf resized.tar
mv of500_images_resized resized_256
cd ..

