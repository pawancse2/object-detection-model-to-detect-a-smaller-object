# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 00:08:14 2019

@author: Pawan
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 23:06:04 2019

@author: Pawan
"""

import math
import sys
import os
import xml.etree.ElementTree as ET
import csv


ANNOTATIONS_FILE = "C:\\Users\\Pawan\\Documents\\ML\\annotations_test_modified2.csv"
CLASSES_FILE = "C:\\Users\\Pawan\\Documents\\ML\\classes_test_modified2.csv"
DATASET_DIR="C:\\Users\\Pawan\\Documents\\ML\\TEST_XML_FILE"

annotations = []
classes = set([])


for xml_file in [f for f in os.listdir(DATASET_DIR) if f.endswith(".xml")]:
  tree = ET.parse(os.path.join(DATASET_DIR, xml_file))
  root = tree.getroot()

  file_name = None
  
  
  for elem in root:
    if elem.tag == 'filename':      
      fileSplit=elem.text.split("/")
      ModifiedFileName="C:\\Users\\Pawan\\Downloads\\dataset_test_rgb\\rgb\\test"+"\\"+fileSplit[9]
      file_name = ModifiedFileName      

    if elem.tag == 'object':
      obj_name = None
      coords = []
      for subelem in elem:
        if subelem.tag == 'name':
          obj_name = subelem.text
        if subelem.tag == 'bndbox':
          for subsubelem in subelem:
            if((str(subsubelem)).find('min')>0):    
              coords.append(math.floor(float(subsubelem.text))) 
            else:
             coords.append(math.ceil(float(subsubelem.text)))
      if  coords[0]>=coords[2] or coords[1] >= coords[3] :
              continue 
      if   coords[2]<=coords[0] or coords[3]<=coords[1]  :
              continue
      item = [file_name] + coords + [obj_name]
      annotations.append(item)
      classes.add(obj_name)

with open(ANNOTATIONS_FILE, 'w') as f:
  writer = csv.writer(f)
  writer.writerows(annotations)

with open(CLASSES_FILE, 'w') as f:
  for i, line in enumerate(classes):
      f.write('{},{}\n'.format(line,i))