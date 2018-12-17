import  pandas as  pd
import  numpy  as  np
import matplotlib.pyplot as plt

url = open("binaryData.csv","r")
df = pd.read_csv(url,names=['URLlong','characters','suspWord','sql','xss','crlf','kolmogorov','kullback','class'])

url.close()
