import streamlit as st, pandas as pd, numpy as np, plotly.express as px
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
st.title("Medical Data Prediction Platform"); st.warning("Synthetic placeholder features only. Not for diagnosis, screening, or treatment.")
X,y=make_classification(n_samples=3000,n_features=8,n_informative=5,weights=[.7,.3],random_state=9); cols=[f"feature_{i}" for i in range(8)]
a,b,c,d=train_test_split(X,y,test_size=.25,random_state=42,stratify=y); m=RandomForestClassifier(n_estimators=200,class_weight="balanced",random_state=42).fit(a,c); p=m.predict_proba(b)[:,1]
x1,x2=st.columns(2); x1.metric("Synthetic records",len(X)); x2.metric("ROC-AUC",f"{roc_auc_score(d,p):.3f}")
imp=pd.DataFrame({"feature":cols,"importance":m.feature_importances_}).sort_values("importance"); st.plotly_chart(px.bar(imp,x="importance",y="feature",orientation="h"),use_container_width=True)
vals={k:st.slider(k,-3.,3.,0.) for k in cols}; st.metric("Illustrative model output",f"{m.predict_proba(pd.DataFrame([vals]))[0,1]:.1%}")
