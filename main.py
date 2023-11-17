import streamlit as st
import helper as lch
import textwrap

st.title('Youtube Assistant')

with st.sidebar:
    with st.form(key='query_input_form'):
        video_url = st.sidebar.text_area(
            label='Video share link..',
            max_chars=50,
            key='url'
        )
        query = st.sidebar.text_area(
           label='Ask me about the video',
           max_chars=50,
           key='query'
       )
        submit_btn = st.form_submit_button(label='Get Answers!')
       
if query and video_url :
    db = lch.create_vector_db_from_youtube_url(video_url= video_url)

    answer = lch.get_response_for_query(query= query , db= db )
    st.subheader('Answer')
    st.text(textwrap.fill(answer,width=80))
  
        
        
        