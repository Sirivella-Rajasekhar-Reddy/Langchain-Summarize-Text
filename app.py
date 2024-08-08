import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import WebBaseLoader

st.set_page_config(page_title="Langchain : Summarize text from Website", page_icon="📚")
st.title(" 📚 Summarize text from Website")

st.subheader("Summarize Text")

web_url=st.text_input("Enter Web Url")

with st.sidebar:
    groq_api_key=st.text_input("Enter Groq API key", type="password")

llm=ChatGroq(groq_api_key=groq_api_key, model="Gemma2-9b-It")

text_template="""
Provide a summary of the following product data in 500 words:
Content:{text}

"""
prompt=PromptTemplate(input_variables=['text'], template=text_template)

if st.button("Summarize"):
    try:
        if not groq_api_key.strip() or not web_url.strip():
            st.error("Please provide the information to get started")
        elif not validators.url(web_url):
            st.error("Please enter a valid Url. It can be website url")
        else:
            loader=WebBaseLoader(web_path=[web_url])
            docs=loader.load()
            chain=load_summarize_chain(llm=llm, chain_type="map_reduce", combine_prompt=prompt, verbose=True)
            output_summary=chain.run(docs)
            st.success(output_summary)
    except Exception as e:
        st.exception(f'exception : {e}')
