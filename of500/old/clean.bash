d=$1
cd images/$d;
mkdir remove
mkdir copy
for i in `ls`; do
  if [ $i != "remove" -a $i != "copy" ]; then
    x=`identify $i | grep -v GIF | grep -v GRAY | cut -f1 -d ' '| sed 's/\[.*\]//'`
    if [ -z "$x" ]; then 
       mv $i remove/;
    else 
       echo "coverting $i"
       y=`convert $i $i.jpg 2>&1`
       if [ -z "$y" ]; then
         mv $i copy/
         mv $i.jpg $i
       else
         echo "problem with $i"
         mv $i remove/
         mv $i.jpg remove/
       fi
    fi     
  fi;
done
rm -rf remove
rm -rf copy
