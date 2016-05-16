import sys

mid = 0
name_id = {}
for line in sys.stdin.readlines():
	tabs = line.split("\t")
	i = 0
	output = ""
	for tab in tabs:
		tab = tab.strip().lower()
		if i > 0: output+="\t"
		output+=tab
		if i > 0 and i%3 == 0:
			if tab not in name_id: 
				name_id[tab] = mid
				mid+=1
			output+= "\t"+str(name_id[tab]) 
		i+=1
	print output
for (k,v) in name_id.items():
	print k + "\t" + str(v)
#print name_id
#print mid
