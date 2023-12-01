import os
import boto3
from langchain.prompts import PromptTemplate 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.llms.bedrock import Bedrock
from langchain.embeddings import BedrockEmbeddings
from langchain.chat_models import BedrockChat
import chainlit as cl
from chainlit.input_widget import Select, Slider
from prompt_template import get_template
from langchain.vectorstores import Chroma

AWS_REGION = os.environ["AWS_REGION"]
#AWS_PROFILE = os.environ["AWS_PROFILE"]
CHROMA_DB_PATH = "vectordb/chromadb/demo.db"

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

@cl.author_rename
def rename(orig_author: str):
    mapping = {
        "ConversationChain": bedrock_model_id
    }
    return mapping.get(orig_author, orig_author)

@cl.on_chat_start
async def main():

    #boto3.setup_default_session(profile_name=AWS_PROFILE)

    bedrock = boto3.client("bedrock", region_name=AWS_REGION)
    bedrock_runtime = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    
    response = bedrock.list_foundation_models(byOutputModality="TEXT")
    
    model_ids = []
    for item in response["modelSummaries"]:
        model_ids.append(item['modelId'])
    
    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="Amazon Bedrock - Model",
                values=model_ids,
                initial_index=model_ids.index("anthropic.claude-v2"),
            ),
            Slider(
                id="Temperature",
                label="Temperature",
                initial=0.3,
                min=0,
                max=1,
                step=0.1,
            ),
            Slider(
                id="MAX_TOKEN_SIZE",
                label="Max Token Size",
                initial=1024,
                min=256,
                max=4096,
                step=256,
            ),
        ]
    ).send()

    await setup_agent(settings)

    # Create embeddings
    embedding_model_id : str = "amazon.titan-embed-text-v1"

    embeddings = BedrockEmbeddings(
        client = bedrock_runtime,
        model_id = embedding_model_id
    )

    #
    vectordb = Chroma(embedding_function=embeddings) #, persist_directory=CHROMA_DB_PATH)

    ###### Load Initial File #######

    files = None

    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a text file to begin!",
            accept=["text/plain"], #"application/pdf"
            max_size_mb=10,
            timeout=90,
        ).send()

    file = files[0]

    msg = cl.Message(content=f"Processing `{file.name}`...", disable_human_feedback=True)
    await msg.send()

    # Decode the file
    text = file.content.decode('utf-8')

    # Split the text into chunks
    texts = text_splitter.split_text(text)

    # Create a metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    #vectordb = Chroma(embedding_function=embeddings, persist_directory=CHROMA_DB_PATH)
    await cl.make_async(vectordb.add_texts) (
        texts, metadatas=metadatas
    )
    #vectordb = await cl.make_async(Chroma.from_texts) (
    #    texts, embeddings, 
    #    metadatas=metadatas
    #)
    #retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    retriever = vectordb.as_retriever(search_type="similarity")

    #similar_docs = retriever.get_relevant_documents("What is the document about?", kwargs={ "k": 2 })

    #print(similar_docs)

    # Set Retriever to the user session
    cl.user_session.set("vectordb", vectordb)
    cl.user_session.set("retriever", retriever)

    await setup_agent(settings)

    msg = cl.Message(content=f"Processing `{file.name}` complete.", disable_human_feedback=True)
    await msg.send()

bedrock_model_id = None
@cl.on_settings_update
async def setup_agent(settings):

    # Get ConversationChain from the user session
    retriever = cl.user_session.get("retriever") 

    #global bedrock_model_id
    bedrock_model_id = settings["Model"]
    
    # Instantiate the chain for user session
    llm = Bedrock(
        region_name=AWS_REGION,
        model_id=bedrock_model_id,
        model_kwargs={"temperature": settings["Temperature"]}
    )

    provider = bedrock_model_id.split(".")[0]
    
    human_prefix="Human"
    ai_prefix="AI"

    MAX_TOKEN_SIZE = int(settings["MAX_TOKEN_SIZE"])
    
    # Model specific adjustments
    if provider == "anthropic":
        llm.model_kwargs["max_tokens_to_sample"] = MAX_TOKEN_SIZE
        human_prefix="H"
        ai_prefix="A"
    elif provider == "ai21":
        llm.model_kwargs["maxTokens"] = MAX_TOKEN_SIZE
    elif provider == "cohere":
        llm.model_kwargs["max_tokens"] = MAX_TOKEN_SIZE    
    elif provider == "amazon":
        llm.model_kwargs["maxTokenCount"] = MAX_TOKEN_SIZE
    else:
        print(f"Unsupported Provider: {provider}")

    prompt = PromptTemplate(
        template=get_template(provider),
        input_variables=["history", "input"], verbose=True
    )

    if retriever is not None:
        message_history = ChatMessageHistory()

        conversation = ConversationalRetrievalChain.from_llm(
            llm = llm, 
            chain_type = "stuff", 
            retriever = retriever, 
            return_source_documents = False,
            memory = ConversationBufferMemory(
                human_prefix=human_prefix,
                ai_prefix=ai_prefix,
                memory_key="chat_history",
                #output_key="response",
                chat_memory=message_history,
                return_messages=True, verbose=True
            ),
            verbose=True)
        print("Created Chain with Retriever")
    else:
        conversation = ConversationChain(
            prompt=prompt, 
            llm=llm, 
            memory=ConversationBufferMemory(
                human_prefix=human_prefix,
                ai_prefix=ai_prefix
            ),
            verbose=True
        )
        print("Created Chain with no Retriever")
    
    # Set ConversationChain to the user session
    cl.user_session.set("llm_chain", conversation)

@cl.on_message
async def main(message: cl.Message):

    # Check File Attachments
    print(f"Attachments: {message.elements}")

    for element in message.elements:
        msg = cl.Message(
            content=f"Processing `{element.name}`...", disable_human_feedback=True
        )
        await msg.send()

    # Get ConversationChain from the user session
    conversation = cl.user_session.get("llm_chain") 

    res = await conversation.acall(
        message.content, 
        callbacks=[cl.AsyncLangchainCallbackHandler()]
    )
    print(f"***** {res}")
    if "answer" in res:
        await cl.Message(content=res["answer"]).send() 
    elif "response" in res:
        await cl.Message(content=res["response"]).send()

@cl.on_chat_end
def end():
    print("bye", cl.user_session.get("id"))