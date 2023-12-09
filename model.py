from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
import chainlit as cl
from constants import CHROMA_SETTINGS
import chromadb
from langchain.prompts import PromptTemplate
from langchain.llms import LlamaCpp


persist_directory = CHROMA_SETTINGS.persist_directory

zypher_prompt_template = """
<|system|>
Answer the question based on the context below. Keep the answer medium and concise. Respond "Unsure about answer" if not sure about the answer.
</s>
<|user|>
Context: {context}
Question: {question}
Answer:
</s>
<|assistant|>
"""
stableLM_zephyr_template = """
<|user|>
Answer the question based on the context below. Keep the answer short and concise. Respond "Unsure about answer" if not sure about the answer.
Context: {context}
Question: {question}
Answer:
<|endoftext|>
<|assistant|>
"""

yi_prompt_template = """
<|im_start|>system
Answer the question based on the context below. Keep the answer medium and concise. Respond "Unsure about answer" if not sure about the answer.<|im_end|>
<|im_start|>user
Context: {context}
Question: {question}
Answer:
<|im_end|>
<|im_start|>assistant
"""

mistral_prompt_template = """<s>[INST]Answer the question based on the context below. Keep the answer short and concise. Respond "Unsure about answer" if not sure about the answer.

Context: {context}
Question: {question}
Answer:
[/INST]
"""

qa_prompt_template = (
    "Context information is below.\n"
    "---------------------\n"
    "{context}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Query: {question}\n"
    "Answer: "
)

qa_prompt_template_zypher = (
    "<|system|>"
    "</s>"
    "<|user|>"
    "Context information is below.\n"
    "---------------------\n"
    "{context}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Query: {question}\n"
    "Answer: "
    "</s>"
    "<|assistant|>"
)

qa_template = """<s>[INST] You are a helpful assistant.
Use the following context to Answer the question below briefly:

{context}

{question} [/INST]
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    #prompt = PromptTemplate(template=qa_prompt_template,
    #                        input_variables=['context', 'question'])
    prompt = PromptTemplate.from_template(stableLM_zephyr_template)
    
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 3},search_type="mmr"),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = LlamaCpp(
    model_path="./models/stablelm-zephyr-3b.Q8_0.gguf",
    repetition_penalty=1.4,
    temperature=0.01,
    max_tokens=7500,
    n_ctx=8192,
    top_k=5,
    verbose=True,
    #n_gpu_layers = 1,
    n_batch = 1024,
    #f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
  )
    return llm

#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name='./models/BAAI_bge-large-en-v1.5',
                                       model_kwargs={'device': 'cpu'})
    chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS , path=persist_directory)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS, client=chroma_client)

    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa


#chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Bot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        for source in sources:
            answer += f"\nSources:" + str(source.metadata['source'])
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer).send()

