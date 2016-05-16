import sys
in_basenetwork_file = sys.argv[1]
in_rep_layer_name = sys.argv[2]
#in_img_blob_name = sys.argv[3]
in_label_blob_name = sys.argv[3]
in_structure_def = sys.argv[4]
in_arg_rep_size = sys.argv[5]
#iin_frame_rep_size = sys.argv[7]
out_network = sys.argv[6]

base_network = open(in_basenetwork_file).read()
relu_template = open("relu.template").read()
inner_product_template = open("inner_product.template").read()
inner_product_template2 = open("inner_product_init2.template").read()
split_template = open("split.template").read()
loss_template = open("loss.template").read()
dropout_template = open("dropout.template").read()
softmax_template = open("softmax.template").read()
attention_template = open("attention.template").read()
slice_template = open("splice.template").read()


struct = open(in_structure_def)

verb_def = {}
id_verb = {}
verb_id_arg = {}
total_args = 0;
for line in struct.readlines():
	tabs = line.split("\t")
	verb = tabs[0]
	verb_id = int(tabs[1])
	verb_args = int(tabs[2])
	args = {}
	verb_id_arg[verb_id] = {}
        for i in range(3, len(tabs),3):
		arg_name = tabs[i]	
		arg_index = tabs[i+1]
		arg_length = tabs[i+2]
		args[arg_index] = arg_length
		verb_id_arg[verb_id][arg_index] = arg_name
                total_args+=1
	verb_def[verb_id] = args
	id_verb[verb_id] = verb

#determine the top for the split
split_bottom = '  bottom: "' + in_rep_layer_name + '"'
split_index = '';
split_top = ''
k = 1;
for (verbid, args) in verb_def.items():
	for (argid, arglength) in args.items():
		basename = id_verb[verbid] + '-' + verb_id_arg[verbid][argid]
		split_top += '\n  top : "rep-' + basename+'"'
		if k < total_args: 
			split_index += '    slice_point: ' + str(k*256) + "\n";
			k+=1
 
output = base_network #+ #"\n" + split_template.replace("${TOP}", split_top).replace("${BOTTOM}", split_bottom)

#make the attention layer. We will do it all together instead of splitting it up

attention = inner_product_template.replace("${NAME}", '  name: "s-attention"');
attention = attention.replace("${TOP}", '  top: "s-attention"')
attention = attention.replace("${BOTTOM}", split_bottom)
attention = attention.replace("${OUTDIM}" ,str(49*total_args));
attend = attention_template.replace("${NAME}", '  name:"attended"');
attend = attend.replace("${BOTTOM1}", '  bottom: "pool5"');
attend = attend.replace("${BOTTOM2}", '  bottom: "s-attention"');   
attend = attend.replace("${TOP}", '  top: "attended"');
attend = attend.replace("${ARGS}", str(total_args));
attend = attend.replace("${MAP}", str(49))
#we will introduce a slicing layer here. 
sl = slice_template.replace("${BOTTOM}", '  bottom:"attended"');
sl = sl.replace("${TOP}" , split_top);
sl = sl.replace("${SLICE}", split_index); 

output = output + "\n" + attention + "\n" + attend + "\n" + sl + "\n"        
#create three layers inner_product to arg embedding, relu, inner_product to arglength
for (verbid, args) in verb_def.items():
	for (argid, arglength) in args.items():
		basename = id_verb[verbid] + '-' + verb_id_arg[verbid][argid]
#  	        attention = inner_product_template.replace("${NAME}", '  name: "s-attention-' + basename + '"')
#		attention = attention.replace("${TOP}", '  top: "s-attention-'+basename+'"')
#		attention = attention.replace("${BOTTOM}", '  bottom: "rep-'+basename+'"')
#		attention = attention.replace("${OUTDIM}" ,"49")
                
#               softmax = softmax_template.replace("${NAME}", '  name:"softmax-'+basename+'"');
# 		softmax = softmax.replace("${TOP}", '  top: "softmax-'+basename+'"')
#               softmax = softmax.replace("${BOTTOM}", '  bottom: "s-attention-'+basename+'"')
                
#                attend = attention_template.replace("${NAME}", '  name:"attend-rep-'+basename+'"');
#                attend = attend.replace("${BOTTOM1}", '  bottom: "pool5"');
#                attend = attend.replace("${BOTTOM2}", '  bottom: "softmax-'+basename+'"');   
 #               attend = attend.replace("${TOP}", '  top: "attend-rep-'+basename+'"');
		dropout = dropout_template.replace("${NAME}", '  name: "dropout-'+basename+'"')
		dropout = dropout.replace("${TOP}", '  top: "dropout-'+basename+'"')
		dropout = dropout.replace("${BOTTOM}", '  bottom: "rep-'+basename+'"')               

#                embed = inner_product_template.replace("${NAME}", '  name: "embed-' + basename + '"')
#		embed = embed.replace("${TOP}", '  top: "embed-'+basename+'"')
#		embed = embed.replace("${BOTTOM}", '  bottom: "rep-'+basename+'"')
#		embed = embed.replace("${OUTDIM}" ,in_arg_rep_size)
		
 #               relu = relu_template.replace("${NAME}", '  name: "relu-'+basename+ '"')
#		relu = relu.replace("${TOP}", '  top:"relu-'+basename+ '"')
#		relu = relu.replace("${BOTTOM}", '  bottom: "embed-'+basename+'"')

		value = inner_product_template2.replace("${NAME}", '  name: "value-' + basename + '"')
		value = value.replace("${TOP}", '  top: "value-' + basename + '"')
		value = value.replace ("${BOTTOM}", '  bottom: "dropout-'+ basename+ '"')
		value = value.replace("${OUTDIM}", arglength)
		output +=  dropout+"\n" + value

#add a verb embedding 
#embed = inner_product_template.replace("${NAME}", '  name: "embed-frame"')
#embed = embed.replace("${TOP}", '  top: "embed-frame"')
#embed = embed.replace("${BOTTOM}", '  bottom: "rep-frame"')
#embed = embed.replace("${OUTDIM}", in_frame_rep_size)
#relu = relu_template.replace("${NAME}", '  name: "relu-frame"')
#relu = relu.replace("${TOP}", '  top: "relu-frame"')
#relu = relu.replace("${BOTTOM}", '  bottom: "embed-frame"')
#dropout = dropout_template.replace("${NAME}", '  name: "dropout-frame"')
#dropout = dropout.replace("${TOP}", '  top: "dropout-frame"')
#dropout = dropout.replace("${BOTTOM}", '  bottom: "relu-frame"')
value = inner_product_template2.replace("${NAME}", '  name: "value-frame"')
value = value.replace("${TOP}", '  top: "value-frame"')
value = value.replace("${BOTTOM}", split_bottom)
value = value.replace("${OUTDIM}", str(len(verb_def)))

#output += "\n" + embed + "\n" + relu   + "\n" + value 
output +=  "\n" + value 
#connect everything to the loss layer using the correct indexing
#args will be connected first, in verb, arg order
loss_bottom = ""
for verbid in range(0, len(verb_def)):
	_id_arg = verb_id_arg[verbid]
	for argid in range(0, len(_id_arg)):
		argid = str(argid)
		bottom_name = "value-"+id_verb[verbid] + '-' + _id_arg[argid]
		loss_bottom += '\n  bottom: "'+ bottom_name + '"'
#the verb is the last bottom
loss_bottom += '\n  bottom: "value-frame"'
loss_bottom += '\n  bottom: "' + in_label_blob_name + '"'

loss = loss_template.replace("${BOTTOM}", loss_bottom)
output += "\n" + loss

out = open(out_network,  "w")
out.write( output + "\n")
out.close()
