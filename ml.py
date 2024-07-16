import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

selectKBest_names = ['Source IP',
                     'Destination IP',
                     'Timestamp',
                     'Flow Duration',
                     'Fwd Packet Length Std',
                     'Bwd Packet Length Max',
                     'Bwd Packet Length Min',
                     'Bwd Packet Length Mean',
                     'Bwd Packet Length Std',
                     'Bwd IAT Total',
                     'Fwd PSH Flags',
                     'Packet Length Std',
                     'Packet Length Variance',
                     'RST Flag Count',
                     'URG Flag Count',
                     'CWE Flag Count',
                     'Down/Up Ratio',
                     'Avg Bwd Segment Size',
                     'Init_Win_bytes_forward',
                     'Init_Win_bytes_backward',
                     'Active Mean',
                     'Active Min',
                     'Idle Mean',
                     'Idle Min',
                     'Inbound',
                     'Label']

author_names = ['Destination IP',
                'Flow Duration',
                'Source IP',
                'Total Length of Bwd Packets',
                'Bwd IAT Mean',
                'Fwd IAT Mean',
                'Flow IAT Mean',
                'Destination Port',
                'Bwd Packet Length Mean',
                'Source Port',
                'Average Packet Size',
                'Total Backward Packets',
                'Subflow Bwd Packets',
                'Fwd Packet Length Mean',
                'Packet Length Mean',
                'Total Fwd Packets',
                'Subflow Fwd Packets',
                'Total Length of Fwd Packets',
                'Down/Up Ratio',
                'Protocol',
                'Label']

my_names1 = ['Protocol',
             'Flow Duration',
             'Total Fwd Packets',
             'Total Backward Packets',
             'Total Length of Fwd Packets',
             'Total Length of Bwd Packets',
             'Fwd Packet Length Max',
             'Fwd Packet Length Min',
             'Fwd Packet Length Mean',
             'Fwd Packet Length Std',
             'Bwd Packet Length Max',
             'Bwd Packet Length Min',
             'Bwd Packet Length Mean',
             'Bwd Packet Length Std',
             'Flow Bytes/s',
             'Flow Packets/s',
             'Flow IAT Mean',
             'Flow IAT Std',
             'Flow IAT Max',
             'Flow IAT Min',
             'Fwd IAT Total',
             'Fwd IAT Mean',
             'Fwd IAT Std',
             'Fwd IAT Max',
             'Fwd IAT Min',
             'Bwd IAT Total',
             'Bwd IAT Mean',
             'Bwd IAT Std',
             'Bwd IAT Max',
             'Bwd IAT Min',
             'Fwd PSH Flags',
             'Bwd PSH Flags',
             'Fwd URG Flags',
             'Bwd URG Flags',
             'Fwd Header Length',
             'Bwd Header Length',
             'Fwd Packets/s',
             'Bwd Packets/s',
             'Min Packet Length',
             'Max Packet Length',
             'Packet Length Mean',
             'Packet Length Std',
             'Packet Length Variance',
             'FIN Flag Count',
             'SYN Flag Count',
             'RST Flag Count',
             'PSH Flag Count',
             'ACK Flag Count',
             'URG Flag Count',
             'CWE Flag Count',
             'ECE Flag Count',
             'Down/Up Ratio',
             'Average Packet Size',
             'Avg Fwd Segment Size',
             'Avg Bwd Segment Size',
             'Fwd Avg Bytes/Bulk',
             'Fwd Avg Packets/Bulk',
             'Fwd Avg Bulk Rate',
             'Bwd Avg Bytes/Bulk',
             'Bwd Avg Packets/Bulk',
             'Bwd Avg Bulk Rate',
             'Subflow Fwd Packets',
             'Subflow Fwd Bytes',
             'Subflow Bwd Packets',
             'Subflow Bwd Bytes',
             'Init_Win_bytes_forward',
             'Init_Win_bytes_backward',
             'act_data_pkt_fwd',
             'min_seg_size_forward',
             'Active Mean',
             'Active Std',
             'Active Max',
             'Active Min',
             'Idle Mean',
             'Idle Std',
             'Idle Max',
             'Idle Min',
             'Label']


def cal_accuracy(Y_test, Y_pred):
    print("Confusion Matrix : ")
    print(confusion_matrix(Y_test, Y_pred))
    print("Accuracy : ")
    print(accuracy_score(Y_test, Y_pred) * 100)
    print("Report : ")
    print(classification_report(Y_test, Y_pred))


data = pd.read_csv('balanced_dataset.csv')
names = my_names1  # switch the names to test
data = data[names]
X = data.iloc[:, 0:-1]
Y = data.iloc[:, -1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

print("******** Decision Tree ********")
dt = DecisionTreeClassifier(criterion='entropy', random_state=0)
dt.fit(X_train, Y_train)
Y_pred_dt = dt.predict(X_test)
cal_accuracy(Y_test, Y_pred_dt)

print("******** Naive Bayes ********")
nb = GaussianNB()
nb.fit(X_train, Y_train)
Y_pred_nb = nb.predict(X_test)
cal_accuracy(Y_test, Y_pred_nb)

print("******** Logistic Regression ********")
lg = LogisticRegression(solver='lbfgs', random_state=0)
lg.fit(X_train, Y_train)
Y_pred_lg = lg.predict(X_test)
cal_accuracy(Y_test, Y_pred_lg)

print("******** Support Vector Machine ********")
svm = SVC(kernel='poly', random_state=0)
svm.fit(X_train, Y_train)
Y_pred_svm = svm.predict(X_test)
cal_accuracy(Y_test, Y_pred_svm)

print("******** K Nearest Neighbor ********")
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn.fit(X_train, Y_train)
Y_pred_knn = knn.predict(X_test)
cal_accuracy(Y_test, Y_pred_knn)

print("******** Random Forest ********")
rf = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
rf.fit(X_train, Y_train)
Y_pred_rf = rf.predict(X_test)
cal_accuracy(Y_test, Y_pred_rf)
