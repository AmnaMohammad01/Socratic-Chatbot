import os
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# ✅ Use st.secrets for API key (Secure for GitHub hosting)
openai_api_key = st.secrets["OPENAI_API_KEY"]
if not openai_api_key:
    st.error("❌ OpenAI API key not found! Set it in Streamlit secrets.")
    st.stop()

# Initialize OpenAI GPT model (Using GPT-3.5 for cost efficiency)
try:
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.8,  # Higher temperature for creative questioning
        openai_api_key=openai_api_key
    )
    st.write("✅ OpenAI GPT-3.5 model initialized.")
except Exception as e:
    st.error(f"Error initializing OpenAI: {e}")
    st.stop()

# Load and process documents
try:
    loader = TextLoader("knowledge_base.txt")  # Ensure this file exists
    raw_documents = loader.load()

    # Split documents into smaller chunks for better processing
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100, length_function=len)
    documents = text_splitter.split_documents(raw_documents)

    # Generate embeddings using OpenAI
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)

    # Create a ChromaDB vector store
    vector_store = Chroma.from_documents(documents, embeddings)

    # Set up the RAG pipeline using LangChain
    rag_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",
                                            retriever=vector_store.as_retriever())

    st.write("✅ Vector store & RAG pipeline initialized.")
except FileNotFoundError:
    st.error("❌ knowledge_base.txt not found in docs/ directory.")
    st.stop()
except Exception as e:
    st.error(f"❌ An error occurred during document processing: {e}")
    st.stop()

# ✅ Optimized Socratic Teaching Prompt
def get_system_message(context=""):
    """Socratic AI Tutor - Enforces a strict Socratic approach."""
    
    base_message = (
        "You are a professor of Parallel & Distributed Computing, and your teaching style is based entirely on the Socratic method. "
        "Instead of directly providing answers, always respond with thought-provoking questions that guide the student to discover the answer themselves. "
        "Your goal is to develop the student’s critical thinking skills."
        "\n\nInstructions:\n"
        "1. NEVER provide direct answers. Instead, respond with well-structured guiding questions.\n"
        "2. If the user asks about a concept, ask them how they think it works first.\n"
        "3. If they struggle, provide a small hint, then ask another question to lead them closer to the answer.\n"
        "4. If the student attempts to answer but is incorrect, encourage them to think deeper and refine their reasoning.\n"
        "5. If the student cannot answer after two attempts, briefly summarize the key idea and ask a related follow-up question.\n"
        "6. If using document information, introduce it as: 'In [Passage X], we see: _quoted text_. What do you think this suggests?'\n"
    )

    if context:
        return f"{base_message}\nRelevant Passages:\n{context}"
    else:
        return f"{base_message}\nNo relevant documents found. Use general knowledge."

# ✅ Optimized Context Formatting (Limit to 2 Passages)
def format_context(search_results):
    """Format search results into a compact context string."""
    if not search_results:
        return ""
    
    context = "Relevant passages:\n\n"
    for idx, result in enumerate(search_results[:2], 1):  # Only include top 2 passages
        context += f"**[Passage {idx}]**:\n_{result.page_content}_\n\n"  # FIXED LINE
    return context

# Streamlit UI
st.title("Socratic RAG Chatbot")
st.write("Ask me anything about Parallel & Distributed Computing!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask me anything!"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        # Retrieve relevant passages from vector store
        search_results = vector_store.similarity_search(prompt, k=2)  # Get top 2 matches
        context = format_context(search_results)

        # Get system message
        system_message = get_system_message(context)

        # Reformulate the user's question into a Socratic question format
        socratic_question = f"Let's think about this step by step. {prompt} What do you already know about this topic?"
        full_prompt = system_message + "\n\nStudent's Question: " + socratic_question

        # Run the RAG pipeline
        response = rag_chain.run(full_prompt)

        # Display assistant response in chat message container
        st.chat_message("assistant").markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    except Exception as e:
        st.error(f"❌ An error occurred during response generation: {e}")
