import streamlit as st
from sklearn import datasets
import numpy as np

st.title("Project X")

st.write(""" 
# Explore Classifier
*Try Parameters*
""")

dataset_name= st.sidebar.selectbox("Select Dataset", ("Iris","Wine","Breast","Diabetes"))

classifier_name= st.sidebar.selectbox("Select Classifier", ("KNN","SVM","Random Forest"))

def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast":
        data = datasets.load_breast_cancer()
    elif dataset_name == "Diabetes":
        data =datasets.load_diabetes()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X,y

X,y = get_dataset(dataset_name)
st.write("Shape of Dataset:" ,X.shape)
st.write("Number of Classes:", len(np.unique(y)))

def add_parameter(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("max_depth", 2,15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["n_estimators"] = n_estimators
        params["max_depth"] = max_depth
    return params

add_parameter(classifier_name)











