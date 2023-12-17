############qa_driven_chatbot.py##############

import streamlit as st
from streamlit.components.v1 import html
import pandas as pd
import requests
import time
import numpy as np

st.title("QA pair-Driven ChatBot")


############ qa pair processing 

uploaded_file = st.file_uploader("Choose a QA pair file", type = 'xlsx')

if uploaded_file is not None:
    df1 = pd.read_excel(uploaded_file)
    df1 = df1.rename(columns={"Question": "question", "Q": "Question"})
    df1 = df1.rename(columns={"Answer": "answer", "A": "answer"})
    df1 = df1[[
        "question", 
        "answer", 
        ]].drop_duplicates()

    st.write(f'loaded {len(df1)} QA pairs from {uploaded_file.name}')

    st.dataframe(df1[["question", "answer"]])

    start = time.time()

    ## embedding by batches. batch size 100. large batch embedding is not supported. 
    question_embedding = []
    answer_embedding = []

    records = df1.to_dict("records")
    num_qa_for_embedding = len(records)
    batch_num = int(np.ceil(num_qa_for_embedding/100))

    for batch_id  in range(batch_num+1):
        try:
            batch = records[batch_id*100:(batch_id+1)*100]
            questions = [r["question"] for r in batch]
            answers = [r["answer"] for r in batch]
            question_embedding += requests.post(
                'http://37.224.68.132:27333/text_embedding/all_MiniLM_L6_v2',
                json = {"texts":questions}).json()["embedding_vectors"]
            answer_embedding += requests.post(
                'http://37.224.68.132:27333/text_embedding/all_MiniLM_L6_v2',
                json = {"texts":answers}).json()["embedding_vectors"]
        except:
            pass

    df1["question_embedding"] = question_embedding
    df1["answer_embedding"] = answer_embedding

    end = time.time()

    st.write(f'embedding of QA texts completed. runing time: {end - start:0.2f} s')

    st.dataframe(df1[["question", "answer", "question_embedding", "answer_embedding"]])

    st.session_state["qa_pairs"] = df1[[
        "question", 
        "answer", 
        "question_embedding", 
        "answer_embedding",
        ]].to_dict("records")


############ chatbot

system_prompt = f'Complete the following chat. Only respond to the last instruction. Your response should be short and abstract, less than 64 words. Do not try to continue the conversation by generating more instructions. Stop generating more responses when the current generated response is longer than 64 tokens.'

def chat_reponse(user_input):

    prompt_start = time.time()

    ## embedding of user input
    user_input_embedding = requests.post(
        'http://37.224.68.132:27333/text_embedding/all_MiniLM_L6_v2',
        json = {"texts":[user_input]}).json()["embedding_vectors"][0]
    user_input_embedding = np.array(user_input_embedding)

    #print(f'embedding of question is done: {str(user_input_embedding)}')

    ## find the most similar question
    qa_pairs_current = st.session_state["qa_pairs"].copy()
    for qa in qa_pairs_current:
        question_score = np.dot(
            user_input_embedding,
            np.array(qa["question_embedding"])
            )
        qa["question_score"] = question_score
    qa_pairs_current = sorted(qa_pairs_current, key=lambda x: x['question_score'], reverse = False)

    #print(f'similarity calculation is done')

    ## if the top 1 question has a score > 0.9 then return it
    most_similar_question = qa_pairs_current[-1]
    if most_similar_question["question_score"] >= st.session_state["semantic_threshold"]:
        answer = most_similar_question["answer"]        
        st.write(f'I find a matched qustion from the QA pair for your question: {most_similar_question["question"]} Matching socre: {most_similar_question["question_score"]:0.2f} runing time: {time.time()-prompt_start:0.2f} s')
        return answer

    st.write(f'I cannot find a matched qustion from the QA pair. Switch to generative model.')

    ## find the most similar answer
    try:
        for qa in qa_pairs_current:
            answer_score = np.dot(
                user_input_embedding,
                np.array(qa["answer_embedding"])
                )
            qa["answer_score"] = answer_score
            qa["overall_score"] = np.max([qa["question_score"], qa["answer_score"]])
        qa_pairs_current = [e for e in qa_pairs_current if e["overall_score"] >= 0.6]
        qa_pairs_current = sorted(qa_pairs_current, key=lambda x: x['overall_score'], reverse = False)
    except:
        pass

    ## prompt engineering
    similar_qa = qa_pairs_current[-4:]
    len_message = 8-len(similar_qa)
    last_messages = st.session_state["messages"][-len_message:]

    prompts = []

    for qa in similar_qa:
        prompts.append(f"[INST] {qa['question'].strip()} [/INST]")
        prompts.append(f"{qa['answer']}")

    for ms in last_messages:
        inst_str_start = "[INST] " if ms["role"] == "user" else ""
        inst_str_end = " [/INST]" if ms["role"] == "user" else ""        
        prompts.append(f"{inst_str_start}{ms['content'].strip()}{inst_str_end}")

    prompts = '\n'.join(prompts)
    prompt = f"<s> <<SYS>> {system_prompt.strip()} <</SYS>>\n{prompts}"
    st.write(f'prompt generated. running time: {time.time() - prompt_start:0.2f} s')

    st.write(f'{prompt}')


    ## call the llm
    # 'http://37.224.68.132:27037/tonomus_llm/llama2_generate'
    # 'http://37.224.68.132:24267/tonomus_llm/mistral_7b_instruct_generate'
    # 'http://37.224.68.132:27427/llama_2_7b_chat_gptq/generate'
    # 1.3 s
    prompt_start = time.time()
    response = requests.post(
        'http://37.224.68.132:27037/tonomus_llm/llama2_generate',
        json = {"prompt": prompt}
        ).json()["response"]
    response = response.split('[INST')[0].strip()
    st.write(f'LLM response generated. running time: {time.time() - prompt_start:0.2f} s')
    return response


def on_btn_click():
    del st.session_state["messages"][:]


st.session_state.setdefault('messages', [])
st.session_state.setdefault('semantic_threshold', 0.9)

values = st.slider(
    'Set semantic score threshold',
    0.0, 1.0, value = 0.9)
st.session_state["semantic_threshold"] = values
st.write('Semantic matching score threshold:', values)

st.button("Clear message", on_click=on_btn_click)

## display the historical chat
for message in st.session_state["messages"][-10:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# accept user input
if user_input := st.chat_input():
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state["messages"].append({
        "role":"user",
        "content":user_input,
        })

    try:
        response = chat_reponse(user_input)
    except Exception as e:
        response = f"Opps. Error: {e}"

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state["messages"].append({
        "role":"assistant",
        "content":response,
        })

############qa_driven_chatbot.py##############