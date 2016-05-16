
j=0
for i in `ls images/`; do 
   bash clean.bash $i;
   echo "$j: $i"
   j=$[$j+1]
done
