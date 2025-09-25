import streamlit as st

def cached(func):
    return st.cache_data(func)
