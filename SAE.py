import scipy
from sklearn.metrics import accuracy_score
# data = sio.loadmat('/Volumes/TOSHIBA EXT/TIZSL/ZSL/AwA/awa_demo_data.mat')
import numpy as np 
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

path = './Animals_with_Attributes2/'

classname = pd.read_csv(path+'classes.txt',header=None,sep = '\t')
dic_class2name = {classname.index[i]:classname.loc[i][1] for i in range(classname.shape[0])}
dic_name2class = {classname.loc[i][1]:classname.index[i] for i in range(classname.shape[0])}
    
def make_test_attributetable():
    attribut_bmatrix = pd.read_csv(path+'predicate-matrix-continuous.txt',header=None,sep = '\s+')
    test_classes = pd.read_csv(path+'testclasses.txt',header=None)
    test_classes_flag = []
    for item in test_classes.iloc[:,0].values.tolist():
        test_classes_flag.append(dic_name2class[item])
    return attribut_bmatrix.iloc[test_classes_flag,:]

def make_train_attributetable():
    attribut_bmatrix = pd.read_csv(path+'predicate-matrix-continuous.txt',header=None,sep = '\s+')
    train_classes = pd.read_csv(path+'trainclasses.txt',header=None)
    train_classes_flag = []
    for item in train_classes.iloc[:,0].values.tolist():
        train_classes_flag.append(dic_name2class[item])
    return attribut_bmatrix.iloc[train_classes_flag,:]


trainfeatures = np.load(path+'resnet101_trainfeatures.npy')
testfeatures = np.load(path+'resnet101_testfeatures.npy')

train_attributelabel = np.load(path+'AWA2_train_continuous_01_attributelabel.npy')
train_attributetable = make_train_attributetable()

test_attributetable = make_test_attributetable()
testlabel = np.load(path+'AWA2_testlabel.npy')     

   

def normalize(fea):
    nSmp,mFea = fea.shape
    feaNorm = np.sqrt(np.sum(np.square(fea),1))
    fea = fea/np.mat(feaNorm).T
    return fea

    
# trainfeatures = normalize(trainfeatures.T).T
# testfeatures = normalize(testfeatures.T).T

lam = 500000

S = np.mat(train_attributelabel).T
X = np.mat(trainfeatures).T

A = S*S.T
B = lam*X*X.T
C = (1+lam)*S*X.T
W = scipy.linalg.solve_sylvester(A,B,C)

W = normalize(W)

test_pre_attribute =  testfeatures.dot(W.T)
test_attributetable = normalize(test_attributetable.T).T


dist = 1-cosine_similarity(test_pre_attribute,test_attributetable)


label_lis = []
for i in range(dist.shape[0]):
    loc = dist[i].argmin()
    hehe  = test_attributetable.index
    label_lis.append(test_attributetable.index[loc])


print(accuracy_score(list(testlabel),label_lis))


















