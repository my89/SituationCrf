import sys

framefile = open(sys.argv[1])

imgs = set()

for line in sys.stdin.readlines():
  imgs.add(line.strip())
  

for line in framefile.readlines() :
	tabs = line.split("\t")       
	if tabs[0].strip() not in imgs: continue
	verb = tabs[0].split("_")[0]
	arg_values = tabs[1].split(")(")
        row = [("verb",verb)]        
	for arg_value in arg_values:
		arg_value_stuff = arg_value.split(";")
		#print arg_value_stuff
		arg = arg_value_stuff[0].replace("(","")
		value = arg_value_stuff[1]
                if len(value.strip()) == 0: value = "null"
                row.append((arg,value))
		print(tabs[0]+"\t"+verb+"\t"+arg+"\t"+value) 
	#row = sorted(row, key = lambda x: x[0])
	#print( str(tabs[0]) + "\t" + str(row))



