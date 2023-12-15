import streamlit as st
from pdf_reader import *

# Creating Session State Variable
if 'API_Key' not in st.session_state:
    st.session_state['API_Key'] = ''
if 'Pinecone_API_Key' not in st.session_state:
    st.session_state['Pinecone_API_Key'] =''
if 'summary' not in st.session_state:
    st.session_state.summary = ''
if 'history' not in st.session_state:
    st.session_state.history = {}
if 'chat' not in st.session_state:
    st.session_state.chat = ''
if 'counter' not in st.session_state:
    st.session_state.counter = 1

st.title('PDF Chat Bot') 

#********SIDE BAR Funtionality started*******

# Sidebar to capture the API keys
st.session_state['API_Key'] = st.sidebar.text_input("What's your OPENAI API key?",type="password")
# File uploader widget
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

load_button = st.sidebar.button("UPLOAD", key="load_button")

#If the bove button is clicked, pushing the data to Pinecone...
if load_button:
    #Proceed only if API keys are provided
    if st.session_state['API_Key'] != '' and uploaded_file is not None:
        file = save_pdf(uploaded_file)
        file = "uploaded.pdf"
        
        st.session_state.summary = load_db_sum(file, st.session_state['API_Key'])
        st.session_state.chat = load_db(file, st.session_state['API_Key'])
        st.session_state.history = {}

    elif st.session_state['API_Key'] == '':
        st.sidebar.error("Please enter your OpenAI API key.")
    elif uploaded_file is None:
        st.sidebar.error("Please attach a PDF file.")

#********SIDE BAR Funtionality ended*****
        

if st.session_state['API_Key'] != '' and uploaded_file is not None:
    file = "uploaded.pdf"
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown("#### **Summary**")
    st.markdown('<hr style="margin: -10px 0; border-top: 1px solid black;">', unsafe_allow_html=True)
    st.write(st.session_state.summary)

    # create a variable for the chat
    conversation = {}

    #Captures User Inputs
    user_input = st.text_input('Ask about the PDF',key="prompt")  # The box for the text prompt
    # document_count = st.slider('No.Of links to return 🔗 - (0 LOW || 5 HIGH)', 0, 5, 2,step=1)

    submit = st.button("SUBMIT") 

    if submit:
        #Proceed only if API keys are provided
        if st.session_state.summary == '':
            st.error("Please upload the PDF file.")
        
        # user_input = request.form['user_input']
        else:
            result = st.session_state.chat({"question": user_input})
            answer_text = str(result['answer'])
            question_text = str(result['question'])
            user = "User"
            chatbot = "Chat Bot"
            conversation.update({user: question_text, chatbot: answer_text})

            user_hist = f"[{st.session_state.counter}] {user}"
            chat_hist = f"[{st.session_state.counter}] {chatbot}"
            st.session_state.history.update({user_hist : question_text})
            st.session_state.history.update({chat_hist : answer_text})
            st.session_state.counter += 1


            st.markdown('<br>', unsafe_allow_html=True)
            st.markdown("#### **Conversation**")
            st.markdown('<hr style="margin: -10px 0; border-top: 1px solid black;">', unsafe_allow_html=True)

            table_data = list(conversation.items())

            # Display the table with keys bolded using HTML
            html_table = """
                <style>
                    table, tr {border:hidden;}
                    table, td {border:hidden;}
                </style>
                <table><tr><th><strong></strong></th><th></th></tr>
                """
            for key, value in table_data:
                html_table += f"<tr><td style='width: 90px;'><strong>{key}:</strong></td><td>{value}</td></tr>"
            html_table += "</table>"
            st.markdown(html_table, unsafe_allow_html=True)


            st.markdown('<br>', unsafe_allow_html=True)
            st.markdown("#### **Chat History**")
            st.markdown('<hr style="margin: -10px 0; border-top: 1px solid black;">', unsafe_allow_html=True)
            
            table_data2 = list(st.session_state.history.items())

            # Display the table with keys bolded using HTML
            html_table = """
                <style>
                    table, tr {border:hidden;}
                    table, td {border:hidden;}
                </style>
                <table><tr><th><strong></strong></th><th></th></tr>
                """
            for key, value in table_data2:
                key = key[4:]
                html_table += f"<tr><td style='width: 90px;'><strong>{key}:</strong></td><td>{value}</td></tr>"
            html_table += "</table>"
            st.markdown(html_table, unsafe_allow_html=True)

elif st.session_state['API_Key'] == '':
    st.error("Please enter your OpenAI API key.")
elif uploaded_file is None:
    st.session_state.summary = ''
    st.error("Please upload the PDF file.")


   
