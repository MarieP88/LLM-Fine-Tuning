from flask import Flask, render_template, request, jsonify, session, send_file
import pickle
import os
import json
import time
import logging
import io
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage
from gtts import gTTS

# these are the  environment variables
load_dotenv(dotenv_path="/home/ubuntu/eu_ai_act_orcawise/without_socket/.env")
api_key = os.getenv("OPENAI_API_KEY")
app = Flask(__name__)
app.secret_key = os.urandom(24)
vectorstore = None

# Configure logging
logger = logging.getLogger('app_logger')
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logging.getLogger('werkzeug').setLevel(logging.WARNING)

previous_response = None

def load_vectorstore():
    global vectorstore
    vector_file_path = 'vector_store.pickle'
    with open(vector_file_path, 'rb') as f:
        vectorstore = pickle.load(f)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def load_conversation_chain():
    global vectorstore
    if vectorstore is None:
        load_vectorstore()
    return get_conversation_chain(vectorstore)


def fallback_similarity_check(q2, r1):
    """
    Fallback approach in case the GPT model fails. This could be any rule-based
    or simpler method to determine similarity between q2 and r1.
    """
    # Example fallback: Basic word overlap percentage between q2 and r1
    q2_words = set(q2.split())
    r1_words = set(r1.split())

    # Calculate the ratio of overlapping words
    overlap_ratio = len(q2_words.intersection(r1_words)) / max(len(q2_words), len(r1_words))

    # Log the fallback output
    logger.debug(f"Fallback similarity ratio (overlap): {overlap_ratio}")

    # If the overlap ratio is above a certain threshold, return True
    if overlap_ratio >= 0.5:  # Assuming 50% overlap as a threshold
        return True
    else:
        return False

def analyse_using_chatgpt(q2, r1):
    llm = ChatOpenAI()
    user_question = f"Provided below details \n{q2} and \n\n{r1}. Please provide the Probability from range (0 to 1) for similarity between {q2} and {r1}. Provide the answer in probability value only."
    logger.debug(f"Current Question : {q2}")
    prob_output = llm([HumanMessage(content=user_question)]).content
    logger.debug(f"Probability output from GPT: {prob_output}")
    try:
        if float(prob_output) >= .7:
            return True
        else:
            return False
    except Exception as e:
        print(f"show the error is {e}")
        logger.error(f"Error in GPT processing: {e}")
        return fallback_similarity_check(q2, r1)

def analyze_continuation(current_question):
    global previous_response
    if previous_response is None:
        return False, ""
    else:
        boolean_output = analyse_using_chatgpt(current_question, previous_response)
        if boolean_output:
            return True, previous_response
        else:
            return False, ""

def handle_userinput(user_question):
    global previous_response
    start_time = time.time()
    analyse_continuation_or_not = analyze_continuation(current_question=user_question)
    end_time = time.time()
    duration = end_time - start_time
    logger.debug(f"Function took {duration:.2f} seconds to complete.")
    logger.debug(f"question asked by user:{user_question}")
    logger.debug(f"Continuation analysis result: {analyse_continuation_or_not}")

    if analyse_continuation_or_not[0] is False:
        llm = ChatOpenAI()
        initial_response = llm([HumanMessage(content=user_question)]).content

        conversation_chain = load_conversation_chain()
        retrieval_response = conversation_chain({'question': user_question})
        combined_input = f"{initial_response}\n\n{retrieval_response['answer']}\n\nBased on above prompt please generate response. Use bulletins, headings, subheading wherever needed. Response should be in such a way that answer is elaborative and meaning full. Present any stats or reference if needed."
        final_response = llm([HumanMessage(content=combined_input)]).content

        if 'chat_history' not in session:
            session['chat_history'] = []
        session['chat_history'].append(AIMessage(content=final_response))
        previous_response = final_response
        return {
            'chat_history': session['chat_history'],
            'user_history': retrieval_response['chat_history']
        }
    else:
        logger.info("Handling continuation of the previous response.")
        user_question_copy = user_question
        user_question = f'Based on current question:{user_question} and previous response: {analyse_continuation_or_not[1]}, please generate new response.'

        llm = ChatOpenAI()
        initial_response = llm([HumanMessage(content=user_question)]).content
        conversation_chain = load_conversation_chain()
        retrieval_response = conversation_chain({'question': user_question_copy})
        final_response = llm([HumanMessage(content=initial_response)]).content

        if 'chat_history' not in session:
            session['chat_history'] = []
        session['chat_history'].append(AIMessage(content=final_response))
        previous_response = final_response

        return {
            'chat_history': session['chat_history'],
            'user_history': retrieval_response['chat_history']
        }

def message_to_dict(message):
    if isinstance(message, HumanMessage):
        return {'role': 'user', 'content': message.content}
    elif isinstance(message, AIMessage):
        return {'role': 'bot', 'content': message.content}
    elif isinstance(message, str):
        return {'role': 'bot', 'content': message}
    else:
        raise ValueError(f"Unknown message type: {type(message).__name__}")

def convert_messages_to_dict(messages):
    chat_history_dict = [message_to_dict(message) for message in messages]
    try:
        return json.dumps(chat_history_dict)
    except TypeError as e:
        pass

@app.route('/handle_input', methods=['POST'])
def handle_input():
    data = request.get_json()
    question = data.get('param')
    chat = handle_userinput(question)
    json_chat_history = convert_messages_to_dict(chat['chat_history'])
    json_user_history = convert_messages_to_dict(chat['user_history'])
    return jsonify({
        'bot_history': json_chat_history,
        'user_history': json_user_history
    })

@app.route('/process-audio', methods=['POST'])
def process_text():
    data = request.get_json()
    text = data.get('text', '')
    tts = gTTS(text, lang='en')
    audio_io = io.BytesIO()
    tts.write_to_fp(audio_io)
    audio_io.seek(0)
    print(f"Received text: {text}")
    return send_file(audio_io, mimetype='audio/mp3')

@app.route('/')
def index():
    global previous_response
    previous_response = None  # Reset previous response on reload
    if 'chat_history' not in session:
        session['chat_history'] = []
    return render_template('index.html')

if __name__ == '__main__':
    #print("Loading the credentials.......")
    #load_dotenv(dotenv_path="/home/ubuntu/eu_ai_act_orcawise/without_socket/.env")
    #print("Credentials are loaded...")
    #print(os.getenv('OPENAI_API_KEY'))
    load_vectorstore()
    app.run(debug=True, port=8000)

