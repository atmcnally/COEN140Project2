# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
from collections import Counter

# potentially do cross validation here, start without
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html is useful for this

traindata = pd.read_csv(
    filepath_or_buffer='train.dat', 
    header=None, 
    sep='\t')

traincategory = traindata.iloc[:, 0]
traindata = traindata.iloc[:, 1]

testdata = pd.read_csv(
    filepath_or_buffer='test.dat', 
    header=None, 
    sep='\t')


# %%
# to do:
# try different methods of feature extraction
# implement diminsionality reduction


# %%
# create list of c-mers for the row
# this grabs three letters at a time
# cmer refers to a count of characters
def cmer(row, c=3):
  # Given a row and parameter c, return the vector of c-mers associated with the row

  if len(row) < c:
    return [row]
  cmers = []
  for i in range(len(row)-c+1):
    cmers.append(row[i:(i+c)])
  return cmers


# %%
from scipy.sparse import csr_matrix

# not fully confident this still works as intended

def build_matrix(data, num):
    r""" Build sparse matrix from a list of documents, 
    each of which is a list of word/terms in the document.  
    """
    mat = [cmer(row, num) for row in data]

    nrows = len(mat)
    idx = {}
    tid = 0
    nnz = 0
    for d in mat:
        wordlist = [x[0] for x in d]
        nnz += len(set(wordlist))
        d = wordlist
        for w in d: #can change here to differen cmer/wmer
            if w not in idx:
                idx[w] = tid
                tid += 1
    ncols = len(idx)
        
    # set up memory
    ind = np.zeros(nnz, dtype=np.int)
    val = np.zeros(nnz, dtype=np.double)
    ptr = np.zeros(nrows+1, dtype=np.int)
    i = 0  # document ID / row counter
    n = 0  # non-zero counter
    # transfer values
    for d in mat:
        listofwords = [x[0] for x in d]
        cnt = Counter(listofwords) #same as above with cmer/wemer
        keys = list(k for k,_ in cnt.most_common())
        l = len(keys)

        for j, k in enumerate(keys):
            ind[j + n] = idx[k]
            val[j + n] = cnt[k]

        ptr[i+1] = ptr[i] + l
        n += l
        i += 1

    mat = csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.double)
    mat.sort_indices()
    return mat


# %%
combinedData = np.append(traindata, testdata)
# this is the sparse matrix of frequencies
# change parameter to build_matrix to change length of cmers
cmerMatrix = build_matrix(combinedData, 3)[0:1566, :]
testMatrix = build_matrix(combinedData, 3)[1566:, :]
# not sure if this needs to be normalized


# %%
#Using the Locally Linear Embedding to reduce the number of dimensions
from sklearn.manifold import LocallyLinearEmbedding

#Reducing the number of dimensions by half (from 23 to 12)
embedding = LocallyLinearEmbedding(n_components=12)
cmer_transformed = embedding.fit_transform(cmerMatrix.toarray())
test_transformed = embedding.fit_transform(testMatrix.toarray())


# %%
# Random Forest Classifier .87
# AdaBoostClassifier .66
from sklearn.ensemble import ExtraTreesClassifier
classifier = ExtraTreesClassifier(n_estimators=1000, random_state=0)
classifier.fit(cmerMatrix, traincategory)

trainPred = classifier.predict(cmerMatrix)
testPred = classifier.predict(testMatrix)


# %%
from sklearn.metrics import matthews_corrcoef
# seems to become 1 after a few runs?
print(matthews_corrcoef(traincategory, trainPred))


# %%
test_predictions_file = open('output.dat', 'w+')
pd.Series(testPred).to_csv("output.dat", index=False, header=None)


# %%
# should use dimensionality reduction -- must show that we tried to use it


