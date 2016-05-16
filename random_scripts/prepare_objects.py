import sys

in_framefile = sys.argv[1]
in_trainfile = sys.argv[2]
in_testfile = sys.argv[3]
out_structurefile_base = sys.argv[4]
out_datafile_base = sys.argv[5]
path_prepend = sys.argv[6]

datain = open(in_framefile)
trainin = open(in_trainfile)
testin = open(in_testfile)

verb_id = {}
verb_datapoints = {}
verb_arg_value = {}
test_files = set()
train_files = set()

for line in trainin.readlines():
	train_files.add(line.strip())

for line in testin.readlines():
	test_files.add(line.strip())

value_id = 0
all_values = {}
alldata = {}
for line in datain:
	tabs = line.split("\t")       
	arg_values = tabs[1].split(")(")		        
	if tabs[0] not in alldata: alldata[tabs[0]] = set()
	values = alldata[tabs[0]]	
	for arg_value in arg_values:
		arg_value_stuff = arg_value.split(";")
		#print arg_value_stuff
		arg = arg_value_stuff[0].replace("(","")
		value = arg_value_stuff[1]
		if len(value.strip()) == 0: continue #you don't get credit for predicting empty!
		if value not in all_values: 
			all_values[value] = value_id
			value_id+=1		
		values.add(value)

out_structure = open(out_structurefile_base+"_id_syn.tab", "w")
for (k,v) in all_values.items():
	out_structure.write(str(v) + "\t" + k + "\n")

maxarg = -1
for (img, syns) in alldata.items():
	if len(syns) > maxarg : maxarg = len(syns)

out_train = open(out_datafile_base + "_" + "train" + ".tab", "w")
out_test = open(out_datafile_base + "_" + "test" + ".tab", "w")

for (img,syns) in alldata.items():
	#v = data[1]
	outstr = path_prepend + img
	for s in syns: outstr += "\t" + str(all_values[s])
	for i in range(len(syns), maxarg): outstr += "\t-1"

	if img in train_files:
		#enumerate. the begining will be the real reference, the end will be the target label
		for s in syns: out_train.write(outstr + "\t" + str(all_values[s]) + "\n")
	elif img in test_files:
		out_test.write(outstr + "\t-1\n")

out_train.close()
out_test.close()	

