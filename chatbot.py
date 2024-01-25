from langchain_community.llms import OpenAI
from src.pdfloader import load_pdf
from src.prompt import pdfreaderprompt
from langchain.chains import RetrievalQA
import os
os.environ["OPENAI_API_KEY"] = "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"


def CreateChatbot():
    #openai_api_key = input("Input your open AI Key >>")
    vectorDB = load_pdf("Realwave_Tasks_20231031.pdf", openai_api_key=os.environ.get('OPENAI_API_KEY'))
    llm = OpenAI(temperature=0, openai_api_key=os.environ.get('OPENAI_API_KEY'))
    prompt = pdfreaderprompt()
    chain_type_kwargs = {
        "prompt": prompt
    }
    chatbot = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vectorDB.as_retriever(),
        chain_type_kwargs=chain_type_kwargs
    )
    return chatbot

if __name__ == '__main__':
    chatbot = CreateChatbot()
    #while (True):
    query = "provide the important content in json format.\n Make sure the output that user want is clean and usable for business purposes.\n Do not add unnecessary content if there is no answer for the question.\n Also make sure the font remains the same in the output."
    #input("Ask Anything >> ")
    response = chatbot.run(query)
    print (response)