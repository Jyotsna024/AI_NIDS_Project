import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="AI NIDS", layout="centered")

def generate_traffic_data():
    traffic={
        "packet_size":np.random.randint(50,1500,700),
        "duration":np.random.uniform(0.01,5.0,700),
        "protocol_type":np.random.randint(1,4,700),
        "traffic_rate":np.random.randint(10,1000,700),
        "label":np.random.randint(0,2,700)
    }
    return pd.DataFrame(traffic)

data=generate_traffic_data()

X=data[["packet_size","duration","protocol_type","traffic_rate"]]
y=data["label"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)

model=RandomForestClassifier(n_estimators=120,random_state=42)
model.fit(X_train,y_train)

st.title("AI-Based Network Intrusion Detection System")
st.write("Simulate network traffic and detect intrusions using Machine Learning")

packet_size=st.slider("Packet Size (Bytes)",50,1500)
duration=st.slider("Connection Duration (Seconds)",0.01,5.0)
protocol=st.selectbox("Protocol Type",[1,2,3])
rate=st.slider("Traffic Rate (Packets/sec)",10,1000)

if st.button("Analyze Traffic"):
    prediction=model.predict([[packet_size,duration,protocol,rate]])
    if prediction[0]==1:
        st.error("Suspicious Activity Detected")
    else:
        st.success("Network Traffic is Normal")
