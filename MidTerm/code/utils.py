import matplotlib.pyplot as plt
import string
import cv2
import os

class Plot:
    def __init__(self,rows,columns):
        self.i=0
        self.j=0
        self.rows=rows
        self.columns=columns
        self.fig, self.plts =  plt.subplots(rows,columns,figsize=(13,13),squeeze=False)
        alphabet_string = string.ascii_lowercase
        self.alphabet_list = list(alphabet_string)

    def set(self,image,title,cmap='gray'):
        self.plts[self.i][self.j].imshow(image,cmap=cmap)
        self.plts[self.i][self.j].axis('off')
        self.plts[self.i][self.j].title.set_text(self.alphabet_list[self.i*self.columns+self.j] + ") " + title)
        if(self.j<self.columns-1):
            self.j+=1
        elif(self.i<self.rows):
            self.i+=1
            self.j=0
        # print("Setting plot at x: "self.i, " y: ",self.j)

    def plot(self,image,title):
        self.plts[self.i][self.j].plot(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        # self.plts[self.i][self.j].axis('off')
        self.plts[self.i][self.j].title.set_text(title)
        if(self.j<self.columns-1):
            self.j+=1
        elif(self.i<self.rows):
            self.i+=1
            self.j=0

    def show(self):
        # self.plts.tight_layout()
        self.fig.show()

    def save(self,savepath,filename):
        if(not (os.path.isdir(savepath))):
            os.makedirs(savepath)
            
        self.fig.savefig(savepath+filename, bbox_inches='tight')
        print("Plots saved on ", savepath+filename)

def createFolder(path):
    if not os.path.exists(path):
        os.makedirs(path)
