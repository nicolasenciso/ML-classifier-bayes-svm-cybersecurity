import  pandas as  pd
import  numpy  as  np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def normalizedData():
    data = open("binaryData.txt","r")
    newData = open("normalizedData.txt","w")
    for line in data:
        splittedLine = line.split(',')
        summatory = 0
        count = 0
        size = len(splittedLine)-1
        for column in splittedLine:
            if count != size:
                summatory = summatory + float(column)
            count += 1
        count = 0
        newLine = ""
        #proof = 0
        for number in splittedLine:
            if count != size:
                newLine = newLine + str(float(number)/(float(summatory))) + ","
                #proof = proof + float(number)/(float(summatory))
            else:
                newLine = newLine + str(number)
            count += 1
        newData.writelines(newLine)
        #print(proof)

    data.close()
    newData.close()

def toBinaryData():
    data = open("rawData.txt","r")
    newData = open("binaryData.txt","w")
    for line in data:
        string = ""
        newLine = line.split(',')
        limit = len(newLine)-1
        if str(newLine[limit]) == "anomalous\n":
            del newLine[limit]
            for col in newLine:
                string = string + str(col) + ","
            string = string + "1"
            newData.writelines(string+"\n")
        elif str(newLine[limit]) == "normal\n":
            del newLine[limit]
            for col in newLine:
                string = string + str(col) + ","
            string = string + "0"
            newData.writelines(string+"\n")
    data.close()
    newData.close()

def bayesClassifierGaussian(X_entreno, y_entreno, X_testeo, y_testeo, name):
    model = GaussianNB()
    model.fit(X_entreno,y_entreno)
    predicted_labels = model.predict(X_testeo)
    listPredicted = predicted_labels.tolist()
    listGivenTest = y_testeo.tolist()
    accuracy = accuracy_score(y_testeo, predicted_labels)
    """print("****** PREDICTED MODEL ********")
    print(predicted_labels)
    print("********* TEST MODEL ************")
    print(y_testeo)"""
    print("\n*** ACCURACY GAUSSIAN BAYES **************")
    print(accuracy)
    print("\n ***** RESULTS BAYES GAUSSIAN "+str(name)+"********\n")
    print(confusion_matrix(y_testeo,predicted_labels))  
    print(classification_report(y_testeo,predicted_labels))
    #y_pred_prob = model.predict_proba(X_testeo)[:,0] PARA COMPROBAR RESULTADOS
    #print(y_pred_prob)
    y_pred_prob = model.predict_proba(X_testeo)[:,1]
    fpr, tpr, thresholds = roc_curve(y_testeo, y_pred_prob)
    auc = roc_auc_score(y_testeo, y_pred_prob)
    print("****** AUC BAYES GAUSSIAN "+str(name)+" *********")
    print(auc)
    return (fpr,tpr)

def bayesClassifierMultinomial(X_entreno, y_entreno, X_testeo, y_testeo,name):
    model = MultinomialNB()
    model.fit(X_entreno,y_entreno)
    predicted_labels = model.predict(X_testeo)
    accuracy = accuracy_score(y_testeo, predicted_labels)
    listPredicted = predicted_labels.tolist()
    listGivenTest = y_testeo.tolist()
    print("\n*** ACCURACY MULTINOMIAL BAYES **************")
    print(accuracy)
    print("\n ***** RESULTS BAYES MULTINOMIAL "+str(name)+" ********\n")
    print(confusion_matrix(y_testeo,predicted_labels))  
    print(classification_report(y_testeo,predicted_labels))
    y_pred_prob = model.predict_proba(X_testeo)[:,1]
    fpr, tpr, thresholds = roc_curve(y_testeo, y_pred_prob)
    auc = roc_auc_score(y_testeo, y_pred_prob)
    print("****** AUC BAYES MULTINOMIAL "+str(name)+" *********")
    print(auc)
    return (fpr,tpr)


def SVMlineal(X_entreno,y_entreno,X_testeo, y_testeo,name):
    svclassifier = SVC(kernel='linear', probability=True)
    svclassifier.fit(X_entreno,y_entreno)
    y_pred = svclassifier.predict(X_testeo)
    print("\n ***** RESULTS SVM LINEAR "+str(name)+"********\n")
    print(confusion_matrix(y_testeo,y_pred))  
    print(classification_report(y_testeo,y_pred)) 
    print("\n*** ACCURACY **************")
    accuracy = accuracy_score(y_testeo, y_pred)
    print(accuracy)
    y_pred_prob = svclassifier.predict_proba(X_testeo)[:,1]
    fpr, tpr, thresholds = roc_curve(y_testeo, y_pred_prob)
    auc = roc_auc_score(y_testeo, y_pred_prob)
    print("****** AUC SVM LINEAR "+str(name)+" *********")
    print(auc)
    return (fpr,tpr)

    
    
def SVMpolynomial(X_entreno,y_entreno,X_testeo, y_testeo, name):
    svclassifier = SVC(kernel='poly', probability=True)
    svclassifier.fit(X_entreno,y_entreno)
    y_pred = svclassifier.predict(X_testeo)
    print("\n ***** RESULTS SVM POLYNOMIAL "+str(name)+"********\n")
    print(confusion_matrix(y_testeo,y_pred))  
    print(classification_report(y_testeo,y_pred))
    print("\n*** ACCURACY **************")
    accuracy = accuracy_score(y_testeo, y_pred)
    print(accuracy)
    y_pred_prob = svclassifier.predict_proba(X_testeo)[:,1]
    fpr, tpr, thresholds = roc_curve(y_testeo, y_pred_prob)
    auc = roc_auc_score(y_testeo, y_pred_prob)
    print("****** AUC SVM POLYNOMIAL "+str(name)+" *********")
    print(auc)
    return (fpr,tpr)
    
    
    """degree = 8
    svclassifier = SVC(kernel='poly', degree=degree, probability=True)
    svclassifier.fit(X_entreno,y_entreno)
    y_pred = svclassifier.predict(X_testeo)
    print("\n ***** RESULTS SVM POLYNOMIAL degree="+str(degree)+"***"+str(name)+"*****\n")
    print(confusion_matrix(y_testeo,y_pred))  
    print(classification_report(y_testeo,y_pred))
    print("\n*** ACCURACY **************")
    accuracy = accuracy_score(y_testeo, y_pred)
    print(accuracy)
    y_pred_prob = svclassifier.predict_proba(X_testeo)[:,1]
    fpr, tpr, thresholds = roc_curve(y_testeo, y_pred_prob)
    auc = roc_auc_score(y_testeo, y_pred_prob)
    print("****** AUC SVM POLYNOMIAL "+str(name)+" *********")
    print(auc)
    return (fpr,tpr)
    """
    
def SVMgaussian(X_entreno,y_entreno,X_testeo, y_testeo, name):
    c = 1.11
    gamma = 0.09
    svclassifier = SVC(kernel='rbf', C=c, gamma=gamma, probability=True)
    svclassifier.fit(X_entreno,y_entreno)
    y_pred = svclassifier.predict(X_testeo)
    print("\n ***** RESULTS SVM GAUSSIAN C="+str(c)+" gamma="+str(gamma)+"*****"+str(name)+"***\n")
    print(confusion_matrix(y_testeo,y_pred))  
    print(classification_report(y_testeo,y_pred))
    print("\n*** ACCURACY **************")
    accuracy = accuracy_score(y_testeo, y_pred)
    print(accuracy)
    y_pred_prob = svclassifier.predict_proba(X_testeo)[:,1]
    fpr, tpr, thresholds = roc_curve(y_testeo, y_pred_prob)
    auc = roc_auc_score(y_testeo, y_pred_prob)
    print("****** AUC SVM GAUSSIAN "+str(name)+" *********")
    print(auc)
    return (fpr,tpr)
    
    
def SVMsigmoid(X_entreno,y_entreno,X_testeo, y_testeo,name):
    svclassifier = SVC(kernel='sigmoid', probability=True)
    svclassifier.fit(X_entreno,y_entreno)
    y_pred = svclassifier.predict(X_testeo)
    print(y_pred)
    print(y_testeo)
    print("\n ***** RESULTS SVM SIGMOID "+str(name)+"********\n")
    print(confusion_matrix(y_testeo,y_pred)) 
    print(classification_report(y_testeo,y_pred))
    print("\n*** ACCURACY **************")
    accuracy = accuracy_score(y_testeo, y_pred)
    print(accuracy)
    y_pred_prob = svclassifier.predict_proba(X_testeo)[:,1]
    fpr, tpr, thresholds = roc_curve(y_testeo, y_pred_prob)
    auc = roc_auc_score(y_testeo, y_pred_prob)
    print("****** AUC SVM SIGMOID "+str(name)+" *********")
    print(auc)
    return (fpr,tpr)
    
    
def logisticReg(X_entreno,y_entreno,X_testeo, y_testeo,name):
    logreg = LogisticRegression()
    logreg.fit(X_entreno, y_entreno)
    y_pred_class = logreg.predict(X_testeo)
    print("\n ***** RESULTS LOGISTIC REGRESSION "+str(name)+"********\n")
    print(confusion_matrix(y_testeo,y_pred_class)) 
    print(classification_report(y_testeo,y_pred_class))
    print("\n*** ACCURACY **************")
    accuracy = accuracy_score(y_testeo, y_pred_class)
    print(accuracy)
    y_pred_prob = logreg.predict_proba(X_testeo)[:,1]
    fpr, tpr, thresholds = roc_curve(y_testeo, y_pred_prob)
    auc = roc_auc_score(y_testeo, y_pred_prob)
    print("****** AUC LOGISCTIC REGRESSION "+str(name)+" *********")
    print(auc)
    return (fpr,tpr)

def randomForest(X_entreno,y_entreno,X_testeo, y_testeo,name):
    classifier=RandomForestClassifier(n_estimators=100)
    classifier=classifier.fit(X_entreno,y_entreno)
    predictions=classifier.predict(X_testeo)
    print("\n ***** RESULTS RANDOM FOREST "+str(name)+"********\n")
    print(confusion_matrix(y_testeo,predictions)) 
    print(classification_report(y_testeo,predictions))
    print("\n*** ACCURACY **************")
    accuracy = accuracy_score(y_testeo, predictions)
    print(accuracy)
    y_pred_prob = classifier.predict_proba(X_testeo)[:,1]
    fpr, tpr, thresholds = roc_curve(y_testeo, y_pred_prob)
    auc = roc_auc_score(y_testeo, y_pred_prob)
    print("****** AUC RANDOM FOREST "+str(name)+" *********")
    print(auc)
    return (fpr,tpr)

def plottingROC(fpr):
    fig, ax = plt.subplots()
    ax.plot(fpr[0][0], fpr[0][1], 'crimson', label='NB-Multinomial')
    """ax.plot(fpr[1][0], fpr[1][1], 'black', label='NB-Gaussian')
    ax.plot(fpr[2][0], fpr[2][1], 'darkgreen', label='SVM lineal')
    ax.plot(fpr[3][0], fpr[3][1], 'blue', label='SVM poly')
    ax.plot(fpr[4][0], fpr[4][1], 'brown', label='SVM Gaussian')
    ax.plot(fpr[5][0], fpr[5][1], 'darkgreen', label='SVM sigmoid')
    ax.plot(fpr[6][0], fpr[6][1], 'magenta', label='LogReg')
    ax.plot(fpr[7][0], fpr[7][1], 'red', label='RandForest')"""
    leg = ax.legend()
    ax.legend(loc='lower right', frameon=True)
    plt.title('ROC curve ML classifier')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.grid(True)
    plt.show()

#original data rawData
url = open("binaryData.csv","r")
df = pd.read_csv(url,names=['URLlong','characters','suspWord','sql','xss','crlf','kolmogorov','kullback','class'])
url.close()
y = df.iloc[:,8].values #dependent variable as y
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3)

X_entreno = X_train.iloc[:,0:8].values
y_entreno = y_train

X_testeo = X_test.iloc[:,0:8].values
y_testeo = y_test

#from raw data
fprs = []
fptxt = open('txtResults/svmpoly.txt','w')
#sal = (bayesClassifierMultinomial(X_entreno,y_entreno,X_testeo,y_testeo,"original data"))
#fptxt.writelines(str(sal[0].tolist())+'#'+str(sal[1].tolist())+'\n')
#sal = (bayesClassifierGaussian(X_entreno,y_entreno,X_testeo,y_testeo,"original data"))
#fptxt.writelines(str(sal[0].tolist())+'#'+str(sal[1].tolist())+'\n')
#sal = (SVMlineal(X_entreno,y_entreno,X_testeo,y_testeo,"original data"))
#fptxt.writelines(str(sal[0].tolist())+'#'+str(sal[1].tolist())+'\n')
sal = (SVMpolynomial(X_entreno,y_entreno,X_testeo,y_testeo,"original data"))
fptxt.writelines(str(sal[0].tolist())+'#'+str(sal[1].tolist())+'\n')
#sal = (SVMgaussian(X_entreno,y_entreno,X_testeo,y_testeo,"original data"))
#fptxt.writelines(str(sal[0].tolist())+'#'+str(sal[1].tolist())+'\n')
#sal = (SVMsigmoid(X_entreno,y_entreno,X_testeo,y_testeo,"original data"))
#fptxt.writelines(str(sal[0].tolist())+'#'+str(sal[1].tolist())+'\n')
#sal = (logisticReg(X_entreno,y_entreno,X_testeo,y_testeo,"original data"))
#fptxt.writelines(str(sal[0].tolist())+'#'+str(sal[1].tolist())+'\n')
#sal = (randomForest(X_entreno,y_entreno,X_testeo,y_testeo,"original data"))
#fptxt.writelines(str(sal[0].tolist())+'#'+str(sal[1].tolist())+'\n')
fptxt.close()
fptxt = open('txtResults/svmpoly.txt','r')
for line in fptxt:
    newLine = line.split('#') #to get points from txt and put it in fprs array
    fpr = newLine[0]
    tpr = newLine[1]
    fpr = fpr.split('[')
    tpr = tpr.split('[')
    fpr = str(fpr[1])
    tpr = str(tpr[1])
    fpr = fpr.split(']')
    tpr = tpr.split(']')
    fpr = str(fpr[0])
    tpr = str(tpr[0])
    fpr = fpr.split(',')
    tpr = tpr.split(',')
    newTpr = []
    newFpr = []
    for point in tpr:
        newTpr.append(float(point))
    for point in fpr:
        newFpr.append(float(point))
    fprs.append((newFpr,newTpr))
fptxt.close()
plottingROC(fprs)


#normalizedData()
url = open("normalizedData.csv","r")
df = pd.read_csv(url,names=['URLlong','characters','suspWord','sql','xss','crlf','kolmogorov','kullback','class'])
url.close()
y = df.iloc[:,8].values #dependent variable as y
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3)

X_entreno = X_train.iloc[:,0:8].values
y_entreno = y_train

X_testeo = X_test.iloc[:,0:8].values
y_testeo = y_test

#from normalized data  Xi / sum(Xn)
fprs = []
"""frps.append(bayesClassifierMultinomial(X_entreno,y_entreno,X_testeo,y_testeo,"Normalized data"))
fprs.append(bayesClassifierGaussian(X_entreno,y_entreno,X_testeo,y_testeo,"Normalized data"))
fprs.append(SVMlineal(X_entreno,y_entreno,X_testeo,y_testeo,"Normalized data"))
fprs.append(SVMpolynomial(X_entreno,y_entreno,X_testeo,y_testeo,"Normalized data"))
fprs.append(SVMgaussian(X_entreno,y_entreno,X_testeo,y_testeo,"Normalized data"))
fprs.append(SVMsigmoid(X_entreno,y_entreno,X_testeo,y_testeo,"Normalized data"))
fprs.append(logisticReg(X_entreno,y_entreno,X_testeo,y_testeo,"Normalized data"))
fprs.append(randomForest(X_entreno,y_entreno,X_testeo,y_testeo,"Normalized data"))"""

#plottingROC(fprs)
