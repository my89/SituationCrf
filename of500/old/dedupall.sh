for i in `cat $1`; do
  echo $i
  python dedup.py $i;
done 
