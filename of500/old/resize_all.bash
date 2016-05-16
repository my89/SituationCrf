echo $1
for v in `cat splits/$1`; do
  mkdir resized/$v
  for name in `ls images/$v`; do
      convert -resize 256x256\! images/$v/$name resized/$v/$name
  done
done

