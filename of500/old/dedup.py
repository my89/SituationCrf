from PIL import Image
import imagehash
import sys
import os
import shutil

img_directory = sys.argv[1]

img = []
hashv = []
removed = set()
for line in os.listdir("images/" + img_directory):
  img.append(line.strip())
  print("computing hash: " + line.strip())
  hashv.append(imagehash.phash(Image.open("images/"+img_directory + "/" +line.strip())))
  
os.mkdir("images/"+img_directory + "/dup")
for i in range(0,len(img)):
  for j in range(i+1,len(img)):
    if i in removed or j in removed: continue
    if hashv[i] - hashv[j] < 10:
      print(img[i] + " dup " + img[j] + " " + str(hashv[i]-hashv[j]) + " " + str(hashv[i]) + " " + str(hashv[j])) 
      os.rename("images/"+img_directory + "/" + img[j] , "images/"+img_directory + "/dup/" + img[j])
      removed.add(j)
shutil.rmtree("images/"+img_directory+"/dup", ignore_errors=True)

print("total dups = " + str(len(removed))) 
