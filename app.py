import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler



st.title("Explore Classifier")

dataset_name= st.sidebar.selectbox("Select Dataset", ("Iris","Wine","Breast Cancer"))

classifier_name= st.sidebar.selectbox("Select Classifier", ("KNN","SVM","Random Forest","Naive Bayes","Logistic","Decision Tree"))

def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X,y , pd.DataFrame(X, columns=data.feature_names)

X,y ,df = get_dataset(dataset_name)
st.write("Shape of Dataset:" ,X.shape)
st.write("Number of Classes:", len(np.unique(y)))

toggle_button = st.sidebar.button('Show Dataset Head')

if 'show_dataset' not in st.session_state:
    st.session_state.show_dataset = False

if toggle_button:
    st.session_state.show_dataset = not st.session_state.show_dataset

if st.session_state.show_dataset:
    st.write('First 6 rows of the dataset:')
    st.dataframe(df.head(6))

def add_parameter(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    elif clf_name == "Naive Bayes":
        alpha = st.sidebar.slider("alpha", 0.0, 1.0, 0.5)
        fit_prior = st.sidebar.checkbox("fit_prior", value=True)
        params["alpha"] = alpha
        params["fit_prior"] = fit_prior
    
        params["alpha"] = alpha
        params["fit_prior"] = fit_prior
    elif clf_name == "Logistic":
        C = st.sidebar.slider("C",0.01,1.0)
        penalty = st.sidebar.selectbox("Penalty", ["l1", "l2", "elasticnet"])
        solver = "liblinear" if penalty in ["l1", "l2"] else "saga"
        l1_ratio = st.sidebar.slider("l1_ratio", 0.0, 1.0, key="l1_ratio") if penalty == "elasticnet" else None
        params["C"] = C
        params["penalty"] = penalty
        params["solver"] = solver
        params["l1_ratio"] = l1_ratio
    elif clf_name == "Decision Tree":
        criterion = st.sidebar.selectbox("Criterion",["gini","entropy","log_loss"])
        params["criterion"] = criterion
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["n_estimators"] = n_estimators
        params["max_depth"] = max_depth
    scale_option = st.sidebar.selectbox("Feature Scaling", ["None", "StandardScaler", "MinMaxScaler"])
    params["scale_option"] = scale_option
    return params

params = add_parameter(classifier_name)

if params["scale_option"] == "StandardScaler":
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
elif params["scale_option"] == "MinMaxScaler":
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

def get_classifier(clf_name,params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"])
    elif clf_name == "Naive Bayes":
        clf = MultinomialNB(alpha=params["alpha"], fit_prior=params["fit_prior"])
    elif clf_name == "Logistic":
        clf = LogisticRegression(C=params["C"],
                                 penalty=params["penalty"],
                                 solver=params["solver"],
                                 l1_ratio=params["l1_ratio"])
    elif clf_name == "Decision Tree":
        clf = DecisionTreeClassifier(criterion=params.get("criterion", "gini"))
    else:
        clf = RandomForestClassifier(n_estimators=params["n_estimators"],
                                     max_depth=params["max_depth"],random_state=1234)
    return clf

clf = get_classifier(classifier_name,params)

# Classification

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

if params["scale_option"] != "None":
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
st.write(f"classifier={classifier_name}")
st.write(f"Accuracy: {acc:.4f}")

f1 = f1_score(y_test, y_pred, average='weighted') 
st.write(f"F1 Score: {f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
st.pyplot(plt.gcf())

#Plotting
pca =PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

fig, ax = plt.subplots(figsize=(6, 4))
scatter = ax.scatter(x1, x2, c=y, alpha=0.8, cmap="viridis")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
fig.colorbar(scatter)

st.pyplot(fig)
