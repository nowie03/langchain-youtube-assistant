from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain.document_loaders import YoutubeLoader

load_dotenv()

embeddings = OpenAIEmbeddings()

def create_vector_db_from_youtube_url(video_url:str):
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000 , chunk_overlap = 100)
    docs = text_splitter.split_documents(transcript)
    
    db = Chroma.from_documents(docs,embeddings)
    return db

def get_response_for_query (query,db,k=4):
    docs = db.similarity_search(query,k = k)
    docs_page_content = " ".join([d.page_content for d in docs])
    
    llm = OpenAI(model = "text-davinci-003")
    prompt = PromptTemplate(
        input_variables=['question','docs'],
        template="""
        you are a helpful youtube assistant that can help in answering questions about videos
        from their transcript 
        
        answer the following question : {question}
        by searching the following transcript : {docs}
        
        only use the factual information from the transcript to answer the question 
        if you feel like you dont have enough information to answer the question 
        say ' i dont know ' 
        
        your answers should be detailed
        """
    )
    
    chain = LLMChain(llm =  llm , prompt = prompt )
    
    resoponse = chain.run(question = query  , docs = docs_page_content)
    resoponse = resoponse.replace("\n"," ")
    
    return resoponse
    
    

if __name__ == "__main__":
    video_url = 'https://youtu.be/FvCt7GxvRDA?si=xUm9gYMwIjHCTYfp'
    db = create_vector_db_from_youtube_url(video_url=video_url)
    query = 'what is the main content of the video'
 
    print(get_response_for_query(query= query , db = db ))


