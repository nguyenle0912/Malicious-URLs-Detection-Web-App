import pickle
import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# load the model from disk
loaded_model = pickle.load(open('detect_mal_url_model.sav', 'rb'))
# Using Tokenizer
vectorizer = pickle.load(open('vectorizer.pk', 'rb'))


# defining the function which will make the prediction using the data which the user inputs 
def prediction(user_url):   
    vectorized_user_url = vectorizer.transform([user_url])
    prediction = loaded_model.predict(vectorized_user_url)
    return prediction[0]

# this is the main function in which we define our webpage  
def main():
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:white;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Malicious URL Detector</h1> 
    </div> 
    """
    benign_html = """<span style="color:white;padding:8px;background-color:#04AA6D;">The URL is benign!</span>"""
    malicious_html = """<span style="color:white;padding:8px;background-color:#f44336;">The URL is malicious!</span>"""

    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True)
    st.markdown("***")
    user_url = st.text_input("Enter the suspicious url here: ")
    if user_url:
        if prediction(user_url) == "benign":
            st.markdown(benign_html, unsafe_allow_html=True)
        elif prediction(user_url) == "malicious":
            st.markdown(malicious_html, unsafe_allow_html=True)
    
if __name__ == '__main__':
    main()
