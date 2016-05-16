import sys

framefile = open(sys.argv[1])

imgs = set()

for line in sys.stdin.readlines():
  imgs.add(line.strip())

img_found = {}

print '{"of500":['
n = 0
outstr = ""
for line in framefile.readlines() :
        tabs = line.split("\t")       
	if tabs[0].strip() not in imgs: continue
        if tabs[0].strip() not in img_found: img_found[tabs[0]] = 0
        else: img_found[tabs[0]] += 1 
	verb = tabs[0].split("_")[0]
	arg_values = tabs[1].split(")(")
        outstr = '{"image":"' + tabs[0].strip()+ '", "verb":"' + verb + '",'
        args = ""
	for arg_value in arg_values:
		arg_value_stuff = arg_value.split(";")
		#print arg_value_stuff
		arg = arg_value_stuff[0].replace("(","")
		value = arg_value_stuff[1]
                if len(value.strip()) == 0: value = "null"
                if len(args) > 0: args+=","
                args+= '"' + arg+'":"' + value + '"';
		#print(tabs[0]+"\t" + str(img_found[tabs[0].strip()]) + "\t" +verb+"\t"+arg+"\t"+value) 
        if n > 0:
                outstr = "," + outstr 
        print outstr + '"frame":{'+args+'}}';
        n+=1
print "]}"
