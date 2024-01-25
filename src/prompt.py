from langchain.prompts import PromptTemplate

def pdfreaderprompt():
    prompt_template = """You are an assistant that only speaks JSON. Do not write normal text
 
{context}

Make sure you reply question only from the context.

{question}
"""
    return PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
