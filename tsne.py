import  pandas as  pd
import  numpy  as  np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

url = open("binaryData.csv","r")
df = pd.read_csv(url,names=['URLlong','characters','suspWord','sql','xss','crlf','kolmogorov','kullback','class'])
X = df.iloc[:,0:8].values
y = df.iloc[:,8].values

X_std = StandardScaler().fit_transform(X)

tsne = TSNE(n_components=3, random_state=0)
x_tsne = tsne.fit_transform(X_std)
anomalousX3,anomalousY3,anomalousZ3,normalX3,normalY3,normalZ3 = [],[],[],[],[],[]
count = 0
for tipo in y:
    if str(tipo) == '1':
        anomalousX3.append(x_tsne[count][0])
        anomalousY3.append(x_tsne[count][1])
        anomalousZ3.append(x_tsne[count][2])
    elif str(tipo) == '0':
        normalX3.append(x_tsne[count][0])
        normalY3.append(x_tsne[count][1])
        normalZ3.append(x_tsne[count][2])
    count += 1


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(anomalousX3,anomalousY3,anomalousZ3, c='b',marker ='o',label='Anomalous')
ax.scatter(normalX3,normalY3,normalZ3, c='r',marker ='o',label='Normal')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.legend(loc='lower left')
plt.tight_layout()
plt.title('t-SNE 3D')


tsne = TSNE(n_components=2, random_state=0)
x_tsne = tsne.fit_transform(X_std)
count = 0
anomalousX,anomalousY,normalX,normalY = [],[],[],[]
for tipo in y:
    if str(tipo) == '1':
        anomalousX.append(x_tsne[count][0])
        anomalousY.append(x_tsne[count][1])
    elif str(tipo) == '0':
        normalX.append(x_tsne[count][0])
        normalY.append(x_tsne[count][1])
    count += 1


plt.figure()
plt.scatter(x=anomalousX,y=anomalousY,c='blue',marker='o',label='Anomalous')
plt.scatter(x=normalX,y=normalY,c='red',marker='o',label='Normal')
plt.ylabel('Y')
plt.xlabel('X')
plt.legend(loc='lower left')
plt.tight_layout()
plt.title('t-SNE')
plt.grid()

markers=('o', 'o')
color_map = {0:'red', 1:'blue'}
plt.figure()
for idx, cl in enumerate(np.unique(y)):
    plt.scatter(x=x_tsne[y==cl,0], y=x_tsne[y==cl,1], c=color_map[idx], marker=markers[idx], label=cl)
plt.xlabel('X in t-SNE')
plt.ylabel('Y in t-SNE')
plt.legend(loc='upper left')
plt.title('t-SNE visualization of test data')

plt.show()

url.close()
