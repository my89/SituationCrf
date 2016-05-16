d=$1

for i in `ls images/$d`; do
  for j in `ls images/$d`; do  
    if [ $i != $j ]; then
      echo $i " " $j " "
      compare -metric PHASH images/$d/$i images/$d/$j NULL:
    fi
  done
done
