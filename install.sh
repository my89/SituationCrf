curl https://s3.amazonaws.com/my89-frame-annotation/public/output_224.tar > of500/output_224.tar
curl https://s3.amazonaws.com/my89-frame-annotation/public/of500_images_resized.tar > of500/resized.tar
curl https://s3.amazonaws.com/my89-frame-annotation/public/OpenFrame500.tab > of500/OpenFrame500.tab

cd of500
tar -xvf output_224.tar 
tar -xvf resized.tar
mv resized resized_256
cd ..
