for i in `ls images`; do
  echo $i
  cd images/$i
  ls | sort -R | split --lines 50
  cat xaa  >> ../../test.txt
  cat xab  >> ../../dev.txt
  rm xaa
  rm xab
  cat xa* >> ../../train.txt
  rm xa*
  cd ../..
done
