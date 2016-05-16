import sys

in_framefile = sys.argv[1]
in_verbfile = sys.argv[2]
in_syn_id = sys.argv[3]
in_trainfile = sys.argv[4]
in_devfile = sys.argv[5]
in_testfile = sys.argv[6]
out_structurefile_base = sys.argv[7]
out_datafile_base = sys.argv[8]
path_prepend = sys.argv[9]

datain = open(in_framefile)
verbin = open(in_verbfile)
trainin = open(in_trainfile)
devin = open(in_devfile)
testin = open(in_testfile)
synin = open(in_syn_id)

syn_id  = {}
verb_id = {}
verb_datapoints = {}
verb_arg_value = {}
test_files = set()
dev_files = set()
train_files = set()

for line in synin.readlines():
	tabs = line.split("\t")
	print tabs
	syn_id[tabs[1].strip()] = tabs[0].strip()

i = 0
for line in verbin.readlines():
	verb_datapoints[line.strip()] = []
	verb_id[line.strip()] = i
	i += 1

for line in trainin.readlines():
	train_files.add(line.strip())

for line in devin.readlines():
	dev_files.add(line.strip())

for line in testin.readlines():
	test_files.add(line.strip())

value_id = 0
all_values = {}
for line in datain:
	tabs = line.split("\t")       
	for (k,v) in verb_datapoints.items():
		if not tabs[0].startswith(k): continue
		arg_values = tabs[1].split(")(")		
		#print line
		#print arg_values
		#print(len(arg_values))
		if k not in verb_arg_value: verb_arg_value[k] = {}
		_arg_value = verb_arg_value[k]
		data = {}
		for arg_value in arg_values:
			arg_value_stuff = arg_value.split(";")
			#print arg_value_stuff
			arg = arg_value_stuff[0].replace("(","")
			value = arg_value_stuff[1]
			if arg not in _arg_value: _arg_value[arg] = []
			if value not in _arg_value[arg]: _arg_value[arg].append(value)
			if value not in all_values: 
				all_values[value] = value_id
				value_id+=1		
			data[arg] = value
		v.append((tabs[0], data))

out_framedef = open(out_structurefile_base+"_frameargs.tab", "w")

verb_arg_id = {}
verb_id_arg = {}
verb_arg_value_id = {}
maxk = 0
for (verb,arg_value) in verb_arg_value.items():
	outstr =  verb + "\t" +  str(verb_id[verb]) + "\t" + str(len(arg_value));
#	print(arg_value)
	verb_arg_value_id[verb] = {}
	verb_id_arg[verb] = {}
	verb_arg_id[verb] = {}
        k = 0
	for (arg,values) in arg_value.items():
		verb_arg_value_id[verb][arg] = {}
		verb_id_arg[verb][k] = arg
		verb_arg_id[verb][arg] = k
		i = 0
		for value in values: 
			verb_arg_value_id[verb][arg][value] = i
			i+=1
		outstr += "\t" + arg + "\t" + str(k) + "\t" + str(len(values))
		k+=1
	if k > maxk: maxk = k
	out_framedef.write(outstr+"\n")

out_framedef.close()

out_argdef = open(out_structurefile_base + "_argvalues.tab", "w")
for (verb, _arg_value_id) in verb_arg_value_id.items():
	for (arg, _value_id) in _arg_value_id.items():
		outstr = verb +"\t" + arg + "\t" +  str(verb_id[verb]) + "\t" + str(verb_arg_id[verb][arg]) + "\t" + str(len(_value_id))
		for (value, _id) in _value_id.items():
			if len(value) == 0: value = "-1"
			else: value = syn_id[value]
			outstr += "\t" + str(value) + "\t" + str(_id)
		out_argdef.write(outstr+"\n")

out_argdef.close()

#merge data points for the same image
verb_merged = {}
for (verb, all_img) in verb_datapoints.items():
	if verb not in verb_merged: verb_merged[verb] = {}
	for data in all_img:
		if data[0] not in verb_merged[verb]: verb_merged[verb][data[0]] = []
		verb_merged[verb][data[0]].append(data[1])

out_train = open(out_datafile_base + "_" + "train" + ".tab", "w")
out_test = open(out_datafile_base + "_" + "test" + ".tab", "w")
out_dev = open(out_datafile_base + "_" + "dev" + ".tab", "w")

for (verb, all_img) in verb_merged.items():
	for (img,data_list) in all_img.items():
		#v = data[1]
		outstr = img 
		for data in data_list:
			outstr += "\t" + str(verb_id[verb])
			for i in range(maxk):
				if i not in verb_id_arg[verb]:
					outstr += "\t" + str(-1)
					continue
                                print verb
				arg = verb_id_arg[verb][i]	
				value = data[arg] 
				vid = verb_arg_value_id[verb][arg][value]
				outstr+="\t" + str(vid)
		if img in train_files: out_train.write(path_prepend + outstr + "\n")
		elif img in dev_files: out_dev.write(path_prepend + outstr + "\n")
		elif img in test_files: out_test.write(path_prepend + outstr + "\n")
		#else: print("img from unknown split!")	

out_train.close()
out_dev.close()
out_test.close()	


