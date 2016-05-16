import sys
from nltk.corpus import wordnet

in_verb_file = open(sys.argv[1])
in_syn_file = open(sys.argv[2])
in_name_file = open(sys.argv[3])
in_arg_map = open(sys.argv[4])
in_ref_file = open(sys.argv[5])
in_sys_file = open(sys.argv[6])
in_freq_file = open(sys.argv[7])
topk = int(sys.argv[8])
out_directory = sys.argv[9]

freq = {}

for line in in_freq_file.readlines():
  tabs = line.strip().split(" ")
  count = tabs[0]
  verb = tabs[1].lower()
  role = tabs[2].lower()
  value = tabs[3].lower()
  freq[verb+"_"+role+"_"+value] = int(count)  


fid_frame = {}
i = 0;
verb_total = {}    
verb_score = {}
for line in in_verb_file.readlines():
  fid_frame[i] = line.strip()
  verb_total[line.strip()] = 0
  verb_score[line.strip()] = 0
  i+=1
  
fid_aid_argname = {}
fid_aid_vmap = {}
for line in in_arg_map.readlines():
  tabs = line.split("\t")
  vid = tabs[2]
  aid = tabs[3]
  index = vid + "." + aid
  vid_synid = {}
  argname = tabs[1]
  for k in range(5,len(tabs),2):
    synid = int(tabs[k])
    vid = int(tabs[k+1])
    vid_synid[vid] = synid
  fid_aid_vmap[index] = vid_synid
  fid_aid_argname[index] = argname;

synid_syn = {}
for line in in_syn_file.readlines():
  tabs = line.split("\t")
  synid_syn[int(tabs[0])] = tabs[1].strip()

syn_name = {}
for line in in_name_file.readlines():
  tabs = line.split("\t")
  syn_name[tabs[0]] = tabs[1].strip()

synid_syn[-1] = "null"
syn_name["null"] = "null"

_predictions = []
predictions = []
_k = 0
for line in in_sys_file.readlines():
  _v = []
  _s = []
  tabs = line[:-1].split("\t")
  fid = tabs[1]
  _s.append(fid_frame[int(fid)])
  for i in range(2, 8):
    index = fid +"." + str(i-2)   
    if int(tabs[i]) == -1: break
    if index not in fid_aid_vmap: continue
    syn = synid_syn[fid_aid_vmap[index][int(tabs[i])]]
    value = syn_name[syn]
    _s.append(fid_aid_argname[index])
    _v.append([syn,value])
  addme = {"s":_s, "v":[_v], "score": tabs[8]}
  _predictions.append(addme)
  _k+=1
  if _k == int(topk):
    predictions.append(_predictions)
    _predictions = [] 
    _k = 0
#  print(addme)
print len(predictions)
print len(predictions[0])
references = []
for line in in_ref_file.readlines():
  _v = []
  _s = []
  tabs = line[:-1].split("\t")
  fid = int(tabs[1])
  frame = fid_frame[fid]
  img = tabs[0]
  _s.append(fid_frame[int(fid)])
  for i in range(2, 8):
    index = str(fid) +"." + str(i-2)   
    if int(tabs[i]) == -1: break
    _s.append(fid_aid_argname[index])
  for r in range(0,3):
    __v = []
    for i in range(1+7*r+1, 1+7*(r+1)):
      if int(tabs[i]) == -1: continue
      index = str(fid) +"." + str( (i-2) % 7)
      syn = synid_syn[fid_aid_vmap[index][int(tabs[i])]]
      value = syn_name[syn]
      __v.append((syn,value))
    _v.append(__v)
  #addme = {"i":"https://s3.amazonaws.com/my89-frame-annotation/images/"+img, "s":_s, "v":_v}
  addme = {"i":"https://s3.amazonaws.com/my89-frame-annotation/public/images_256/"+img, "s":_s, "v":_v}
  references.append(addme)
print len(references)

col_correct = {}
col_count = {}
verb_correct = {}
verb_count = {}

def tohtml(img, structure, values, colors, counts, score=None):
  cols = len(structure)-1
  rv = "<table border=2 style='background-color:"+colors[0]+"; width:500px; margin:0 auto'>\n"
  if len(img) > 0:
    rv += "<tr align='center'><td colspan=" + str(cols) + '><img width="300" height="300" src="' + img + '"/></tr>\n'
  #print structure
  #print values
  #print colors
  verb = structure[0]
  if score == None: rv += "<tr align='center'><td colspan ="+str(cols) + ">" + verb + "</td></tr>"
  else: 
    rv += "<tr align='center'><td colspan ="+str(cols) + ">" + verb + "</td></tr>"
    rv += "<tr align='center'><td colspan ="+str(cols) + ">" + score + "</td></tr>"
  rv += "<tr align='center'>"
  for i in structure[1:]:
    rv += "<td><b>" + i.lower() + "</b></td>"
  rv += "</tr>\n"
  for _v in values:
    rv += "<tr align='center'>"
    k = 1
   # print colors
   # print _v
    for x in _v:
      rv += "<td style='background-color:"+colors[k]+"'>" + x[1] + "</td>"
      k+=1
    rv += "</tr>\n"
#  print counts
  if counts != 0:
    min_min = 9999;
    rv += "<tr>"
    for i in range(0,len(structure)-1):
       role = structure[i+1]
       min_value = 0;

       all_null = 1 
       for _v in values:
         if _v[i][0] != "null": all_null = 0
 
       for _v in values:
         key = verb+"_"+role.lower()+"_"+_v[i][0]
#         print _v[i][0]
         if all_null == 0 and _v[i][0] == "null" : continue
         if key in counts: c = counts[key]
         else: c = 0
         if c > min_value: min_value = c
       if min_value not in col_correct: 
         col_correct[min_value] = 0
         col_count[min_value] = 0
       if colors[i+1] == "LightGreen": col_correct[min_value]+=1
       col_count[min_value]+=1
       if min_min > min_value: min_min = min_value
       rv+="<td style='text-align:center'>"+str(min_value)+"</td>"
    if min_min not in verb_correct: 
      verb_correct[min_min] = 0
      verb_count[min_min] = 0
    if colors[0] == "LightGreen": verb_correct[min_min]+=1
    verb_count[min_min] += 1
  rv += "</table>\n"
  return rv


rand ={}
for i in range(0, len(references)):
  _refs = references[i]
  found = -1
  index = 0

  #compute the color array 
  colors = [] 
  for _pred in predictions[i]:
    if _pred["s"][0] == _refs["s"][0]:
      colors.append("LightGreen");
      length = len(_pred["s"])-1
      #if length not in rand: rand[length] = []
      #r = tohtml(_refs["i"] , _refs["s"], _refs["v"],"LightGreen",freq)
      #p = tohtml( "", _pred["s"] , _pred["v"], "LightGreen",0) 
#      rand[length].append((r,p))
      verb_score[_refs["s"][0]]+=1
      found = index
   #   print ("length=" + str(length))
      for k in range(0, length):
        found = 0
        for _v in _refs["v"]:
          if _v[k][0] == _pred["v"][0][k][0]: found = 1  
        if found : colors.append("LightGreen")
        else: colors.append("LightCoral")
    index +=1
 
  #print ("ref verb = " + str(_refs["s"][0])) 
  nothing = []
  for k in range(0,7): nothing.append("LightCoral") 
  if len(colors) == 0: colors = nothing 
  verb_total[_refs["s"][0]] += 1

  r = tohtml(_refs["i"] , _refs["s"], _refs["v"],colors,freq)
  verb = _refs["s"][0]
  index = 0
  outfile = out_directory + "/" + verb + ".html"	
  outfile = open(outfile, "a")
  outfile.write(r)
  for _pred in predictions[i]:
  #_pred = predictions[i]
    if _pred["s"][0] == _refs["s"][0]: _color = colors
    else: _color = nothing 
    p = tohtml( "", _pred["s"] , _pred["v"], _color,0, _pred["score"]) 
    outfile.write("</br>" +p)
    index +=1
  outfile.write("<hr><hr>\n");
  outfile.close()

outfile = open(out_directory + "/verb.txt", "w");
for (k,v) in verb_total.items():
  outfile.write( k + "\t" + str(verb_score[k]/float(v)) + "\n") 
outfile.close()

outfile = open("verb_hist.txt", "w")
for (k,v) in verb_correct.items():
  outfile.write(str(k) + "\t" + str(float(verb_correct[k])/float(verb_count[k])) + "\n")
outfile.close()
outfile = open("role_hist.txt", "w")
for (k,v) in col_correct.items():
  outfile.write(str(k) + "\t" + str(float(col_correct[k])/float(col_count[k])) + "\n")
outfile.close()
 
#import random
#outfile = open(out_directory + "/50random.html", "w");
#for (k,v) in rand.items():
#  random.shuffle(v)
#  for i in range(0,10):
#    outfile.write(v[i][0]);
#    outfile.write(v[i][1])
#outfile.close()
