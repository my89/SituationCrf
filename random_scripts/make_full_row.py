import sys

in_train_file_name = sys.argv[1]
in_train_file = open(in_train_file_name)
in_dev_file = open(sys.argv[2])
#in_test_file = open(sys.argv[3])
in_maxlabels = int(sys.argv[3])
in_maxn = int(sys.argv[4])
out_structure_file = open(sys.argv[5],"w")
out_train_file = open(sys.argv[6],"w")
out_dev_file = open(sys.argv[7],"w")
#out_test_file = open(sys.argv[9],"w")

def diff(list1, list2):
  #if verbs disagree, it doesn't matter 
  if list1[0] != list2[0] : return 0
  v = 0
  for i in range(1, len(list1)):
    if list1[i] == list2[i]: v+=1
  return v

#ref_count = {}
ref_array = {} #string_array
verb_ref_count = {}
for line in in_train_file:
  tabs = line.split("\t")
  img = tabs[0]
  for i in range(0,3):
    sv = ""
    av = []
    verb = ""
    for j in range(0,in_maxlabels):
      v = tabs[in_maxlabels*i + j + 1]      
      if len(verb) == 0: 
        verb = v.strip() 
      sv += "_" + v.strip()
      av.append(int(v))
    if verb not in verb_ref_count: verb_ref_count[verb] = {}
    if sv not in verb_ref_count[verb]:
      verb_ref_count[verb][sv] = 1
      ref_array[sv] = av
    else: verb_ref_count[verb][sv] += 1


accepted_ref = {}
i = 0
for (verb,ref_count) in verb_ref_count.items():
  #get the most frequent references
  x = sorted(ref_count.items(), key=lambda x: -x[1])
  j = 0
  for k in x:
    if j >= in_maxn: break
    accepted_ref[k[0]] = i
    sout = ""
    for _v in k[0].split("_"):
      sout += "\t" + _v
    out_structure_file.write(sout.strip() + "\n")
    i+=1
    j+=1

out_structure_file.close()

in_train_file = open(in_train_file_name)

def map_file(in_fd, out_fd):
  c = 0
  for line in in_fd:
    c+=1
    if c % 1000 == 0 : print c
    tabs = line.split("\t")
    img = tabs[0]
    svs = []
    for i in range(0,3):
      sv = ""
      av = []
      for j in range(0,in_maxlabels):
        v = tabs[in_maxlabels*i + j + 1]
        sv += "_" + v
        av.append(int(v))
      if sv not in ref_array: ref_array[sv] = av
      svs.append(sv)
    outindex = -1
    for i in range(0,3):
      if svs[i] in accepted_ref:
        outindex = accepted_ref[svs[i]]
        ms = "!"+svs[i]
        break
    mv = -1
    if outindex == -1:
      mv = -1
      mi = -1
      ms = ""
      for sv in svs:
        for _r in accepted_ref:
          tv = diff(ref_array[_r], ref_array[sv]) 
          if tv > mv: 
            mv = tv
            mi = accepted_ref[_r]
      outindex = mi
    #print(line.strip() + "\t" + ms + "\t" + str(outindex)) 
    out_fd.write(line.strip() + "\t" + str(outindex) + "\n")# + str(mv) + "\n") 
  out_fd.close()
   
map_file(in_train_file, out_train_file)
map_file(in_dev_file, out_dev_file)
#map_file(in_test_file, out_test_file)
