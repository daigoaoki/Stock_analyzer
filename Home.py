import streamlit as st

st.set_page_config(page_title = "Stock analyze", layout="wide")
# st.markdown('''
# #<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-EVSTQN3/azprG1Anm3QDgpJLIm9Nao0Yz1ztcQTwFspd3yD65VohhpuuCOmLASjC" crossorigin="anonymous">
# #''', unsafe_allow_html=True)


# with open('style2.css') as f:
#     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

hide_st_style = """
            <style>
            footer {visibility: visible;}
            footer:after {content:'Copyright @ AI trader X'; display:block; position: relative;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


#st.sidebar.image("figure/MODECLOGO.png", width=100)
st.sidebar.subheader("Analyzer V 0.0.1")


st.header("Stock analyzer V 0.0.1")
st.markdown("Hello there!")
st.markdown("This web application is a software which is designed for using stock analyzing")

