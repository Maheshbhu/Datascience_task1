from re import X
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as ans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
print("hello world")

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"


column_names=["sepal_length","sepal_width","patal_length","petal_width","class"]
iris_data=pd.read_csv(url,names=column_names)
iris_data.head(50) #give 50 rows 
iris_data.describe() #it gives mean and standard daviation
ans.pairplot(iris_data,hue="class") #ploat the graph
#plt.show()

x=iris_data.drop("class",axis=1) #drop the column class
y=iris_data["class"] # asign class in y 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

knn=KNeighborsClassifier(n_neighbors=3)# here we want 3 neighbours 

knn.fit(x_train,y_train)

y_pred=knn.predict(x_test)
#print("Acuracy:",accuracy_score(y_test,y_pred))
#for calculating accuracy

#for classification report
#print(classification_report(y_test,y_pred))

x_test.head(2)
