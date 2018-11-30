import  pandas as  pd
import  numpy  as  np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC

def normalizedData():
    data = open("rawData.txt","r")
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

def bayesClassifier(X_entreno, y_entreno, X_testeo, y_testeo):
    model = GaussianNB()
    model.fit(X_entreno,y_entreno)
    predicted_labels = model.predict(X_testeo)
    """print("****** PREDICTED MODEL ********")
    print(predicted_labels)
    print("********* TEST MODEL ************")
    print(y_testeo)"""
    print("\n")
    listPredicted = predicted_labels.tolist()
    listGivenTest = y_testeo.tolist()
    count, countMatched = 0,0
    #print("--GIVEN --- PREDICTED")
    for label in listPredicted:
        if str(listGivenTest[count]) == str(label):
            countMatched += 1
        #print(str(listGivenTest[count])+" --- "+str(label))
        #print("------------------------")
        count += 1
    print("RESULTS: ")
    print("--------------------------------------")
    print(count)
    print("Matched: "+str(countMatched)+" / "+str(count))
    print("Ratio: "+str(float(countMatched)/float(count)))
    accuracy = accuracy_score(y_testeo, predicted_labels)
    print("\n*** ACCURACY GAUSSIAN BAYES **************")
    print(accuracy)
    model = MultinomialNB()
    model.fit(X_entreno,y_entreno)
    predicted_labels = model.predict(X_testeo)
    accuracy = accuracy_score(y_testeo, predicted_labels)
    listPredicted = predicted_labels.tolist()
    listGivenTest = y_testeo.tolist()
    count, countMatched = 0,0
    #print("--GIVEN --- PREDICTED")
    for label in listPredicted:
        if str(listGivenTest[count]) == str(label):
            countMatched += 1
        #print(str(listGivenTest[count])+" --- "+str(label))
        #print("------------------------")
        count += 1
    print("RESULTS: ")
    print("--------------------------------------")
    print(count)
    print("Matched: "+str(countMatched)+" / "+str(count))
    print("Ratio: "+str(float(countMatched)/float(count)))
    print("\n*** ACCURACY MULTINOMIAL BAYES **************")
    print(accuracy)

def SVMclassifier(X_entreno,y_entreno,X_testeo, y_testeo):
    """c = 1
    degree = 3
    svclassifier = SVC(kernel='linear',C=c)
    svclassifier.fit(X_entreno,y_entreno)
    y_pred = svclassifier.predict(X_testeo)
    print("\n ***** RESULTS SVM LINEAR ********"+"C="+str(c)+" degree="+str(degree)+"\n")
    print(confusion_matrix(y_testeo,y_pred))  
    print(classification_report(y_testeo,y_pred)) """

    """svclassifier = SVC(kernel='poly', degree=8)
    svclassifier.fit(X_entreno,y_entreno)
    y_pred = svclassifier.predict(X_testeo)
    print("\n ***** RESULTS SVM POLYNOMIAL ********\n")
    print(confusion_matrix(y_testeo,y_pred))  
    print(classification_report(y_testeo,y_pred))"""

    c = 1.11
    gamma = 1
    svclassifier = SVC(kernel='rbf')
    svclassifier.fit(X_entreno,y_entreno)
    y_pred = svclassifier.predict(X_testeo)
    print("\n ***** RESULTS SVM GAUSSIAN C="+str(c)+" gamma="+str(gamma)+"********\n")
    #print("\n ***** RESULTS SVM GAUSSIAN C="+str(c)+"********\n")
    #print("\n ***** RESULTS SVM GAUSSIAN ********\n")
    print(confusion_matrix(y_testeo,y_pred))  
    print(classification_report(y_testeo,y_pred))

    """svclassifier = SVC(kernel='sigmoid')
    svclassifier.fit(X_entreno,y_entreno)
    y_pred = svclassifier.predict(X_testeo)
    print("\n ***** RESULTS SVM SIGMOID ********\n")
    print(confusion_matrix(y_testeo,y_pred))  
    print(classification_report(y_testeo,y_pred))"""


#original data rawData
url = open("rawData.csv","r")
df = pd.read_csv(url,names=['URLlong','characters','suspWord','sql','xss','crlf','kolmogorov','kullback','class'])
url.close()
y = df.iloc[:,8].values #dependent variable as y
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3)

X_entreno = X_train.iloc[:,0:8].values
y_entreno = y_train

X_testeo = X_test.iloc[:,0:8].values
y_testeo = y_test

#from raw data
print("\n=================== FROM ORIGINAL DATA WITH 8 FEATURES ==========\n")
#bayesClassifier(X_entreno,y_entreno,X_testeo,y_testeo)
#SVMclassifier(X_entreno,y_entreno,X_testeo,y_testeo)

normalizedData()
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
print("\n=================== FROM NORMALIZED DATA WITH 8 FEATURES ==========\n")
bayesClassifier(X_entreno,y_entreno,X_testeo,y_testeo)
SVMclassifier(X_entreno,y_entreno,X_testeo,y_testeo)