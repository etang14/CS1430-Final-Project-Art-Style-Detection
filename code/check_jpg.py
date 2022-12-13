from struct import unpack
import os
from os.path import isfile, join

# one-time code used for checking all jpeg image files for corruption

marker_mapping = {
    0xffd8: "Start of Image",
    0xffe0: "Application Default Header",
    0xffdb: "Quantization Table",
    0xffc0: "Start of Frame",
    0xffc4: "Define Huffman Table",
    0xffda: "Start of Scan",
    0xffd9: "End of Image"
}


class JPEG:
    def __init__(self, image_file):
        with open(image_file, 'rb') as f:
            self.img_data = f.read()
    
    def decode(self):
        data = self.img_data
        while(True):
            marker, = unpack(">H", data[0:2])
            # print(marker_mapping.get(marker))
            if marker == 0xffd8:
                data = data[2:]
            elif marker == 0xffd9:
                return
            elif marker == 0xffda:
                data = data[-2:]
            else:
                lenchunk, = unpack(">H", data[2:4])
                data = data[2+lenchunk:]            
            if len(data)==0:
                break        


bads = []
styles = ['abstract-expressionism', 'cubism', 'impressionism', 'renaissance', 'romanticism']
for style in styles:
    images = [f for f in os.listdir("data/images/" + style) if isfile(join("data/images/" + style + "/", f))]
    for img in images:
        image = join("data/images/" + style + "/", img)
        image = JPEG(image) 
        try:
            image.decode()   
        except:
            bads.append(img)


for name in bads:
    print(name)
    os.rename(join("data/images/" + style + "/", name), "data/" + name)