# import needed libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_score
from sklearn.metrics import precision_score
############################################
df = pd.read_csv('iris_csv.csv') # load the Iris-dataset
# print(df.info()) # show some basic informations about dataset
############################################
# seperate target column as y to other columns
x = df.drop('class',axis=1,inplace=False)
y= df['class']
############################################
x_train, x_test, y_train, y_test = tts(x,y,test_size=0.3,random_state=2) # create random train and test variables from x and y
############################################
ns = rfc(bootstrap=True,random_state=20) # call Knn.- we have 3 class so n_neighbors is 3
ns.fit(x_train,y_train) # use x_train data to learn algorithm
pred = ns.predict(x_test) # predict the x_test algorithm
pred_results = pd.DataFrame({'expected_y':y_test,'predicted_y':pred}) # save the y_test and predicts in a csv file
########################################
# how to run scores
y_true = y_test # create new variable for y_test to use in scores
y_pred = pred # create new variable for predicted values to use in scores

acc_sc =  (accuracy_score(y_true,y_pred)).astype(str) # call and save accuarcy of real test targets and predictions score in a variable
conf_mat = (confusion_matrix(y_true,y_pred)) # call and save confusion matrix of real test targets and predictions in a variable
jac_sc = jaccard_score(y_true,y_pred, labels = ['Iris-setosa','Iris-versicolor','Iris-virginica'] , average='macro').astype(str) # call and save Iou of real test targets and predictions score in a variable
prec_sc = (precision_score(y_true,y_pred,labels = ['Iris-setosa','Iris-versicolor','Iris-virginica'] , average='macro')).astype(str) # call and save persicion of real test targets and predictions score in a variable

#print scores
print('accuracy score is : ' , acc_sc)
print('confusion matrix is : ', conf_mat )
print('IoU_score is : ', jac_sc)
print('precision score is : ' , prec_sc)

f = open('rfc_accuracy score.txt','w') # to open or create a txt file
f.write(acc_sc) # write a variable in the opened file
f.close() # to stop writing on the opened file

np.savetxt('rfc_confusion matrix.txt',conf_mat, fmt = '%10.5f') # save confusion matrix 

f = open('rfc_IoU score.txt','w')
f.write(jac_sc)
f.close()

f = open('rfc_precision score.txt','w')
f.write(prec_sc)
f.close()

pred_results.to_csv('rfc_results.csv') # create a csv file for real y test data and prediction of it