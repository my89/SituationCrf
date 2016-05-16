import sys

labels = open(sys.argv[1])
images = open(sys.argv[2])
prefix = sys.argv[3]

label_id = {}
i = 0
for line in labels.readlines():
	label_id[line.strip()] = i
	i+=1

for line in images:
	for (key, value) in label_id.items():
		if line.startswith(key):
			print prefix+"/resized/"+line.strip() + "\t" + str(value)
			break
