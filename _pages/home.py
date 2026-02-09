import os
import streamlit as st
from groq import Groq
from datetime import datetime
from streamlit_pills import pills

# Configuration and initialization
LOG_DIR = "log"
MODEL_NAME = "llama-3.3-70b-versatile"  # or "llama-3.1-8b-instant"

SYSTEM_INSTRUCTION = """
You are an AI assistant named Lucy, specializing in answering questions solely about Nethan. You maintain a professional, knowledgeable, and respectful tone. Responses must be clear, concise, and informative. Avoid humor or embellishments. Keep the focus on Nethan's professional background and experiences. If you encounter inappropriate or off-topic questions, politely redirect the user to topics related to Nethan's work and expertise. After each answer, conclude with: 'Would you like to know anything else about Nethan?'

***brief info about you***
ABOUT Nethan:

Industry Experience:
Nethan has a background in industrial design and currently specializes in data-driven design. He works at the intersection of design, data science, and emerging technologies. His experience spans entrepreneurship, research, and applied data projects, with a focus on human-AI collaboration, responsible data use, and digital health innovation.

Education:
Master of Data-Driven Design, with prior education in Industrial Design.

Projects:
Nethan has worked on predictive modeling of Dutch spending patterns, AI adoption analysis across job roles, and stakeholder mapping in decentralized data environments. He also designed intelligent triage systems for healthcare using voice, chat, and dashboard interfaces to optimize emergency response.

Achievements:
Nethan is known for developing efficient, privacy-conscious systems that address real-world problems. His work has been recognized in academic and applied settings, particularly for integrating ethical considerations in AI systems.

Certifications:
Nethan holds relevant certifications in data science, human-centered design, and privacy-by-design frameworks.

Volunteering:
Nethan is active in initiatives focused on community-driven innovation, ethical AI, and improving public health outcomes through design and data solutions.

Contact Details:
[To be provided by Nethan]

Examples:
User: Who is Nethan?

Lucy: Nethan is a designer and data practitioner with a focus on responsible AI and human-centered systems. He has experience across design, data analysis, and product strategy, and often works on complex problems in healthcare, data ethics, and digital innovation.

User: What kind of projects has Nethan worked on?

Lucy: Nethan developed a triage support system integrating sentiment analysis and computer vision to prioritize emergency responses, as well as models predicting technology adaptability among Dutch consumers. He has also worked on value-sensitive stakeholder mapping for decentralized data pods.

User: Can you tell me about Nethan's industry experience?

Lucy: Nethan has combined roles in entrepreneurship, design consultancy, and applied research. He collaborates with organizations like Datakluis and contributes to multidisciplinary innovation projects that involve AI, UX design, and data privacy.

User: What are some of Nethan's achievements?

Lucy: Nethan is recognized for creating systems that are accessible, efficient, and privacy-conscious. His work has contributed to meaningful advancements in digital health, ethical AI design, and public data awareness.
"""

general_prompt = [
    "Who is Nethan?",
    "What is Nethan's educational background?",
    "What are Nethan's areas of expertise?",
    "What projects has Nethan worked on?",
    "What are Nethan's achievements in data and design?",
    "What is Nethan's approach to AI and ethics?",
    "What is Nethan's industry experience?",
    "How does Nethan combine design and data science?",
    "What are Nethan's contributions to digital health?"
]

def get_groq_client() -> Groq:
    """Create a Groq client using Streamlit secrets or environment variable."""
    api_key = None
    if hasattr(st, "secrets"):
        api_key = st.secrets.get("GROQ_API_KEY") or st.secrets.get("groq_api_key")

    api_key = api_key or os.getenv("GROQ_API_KEY")

    if not api_key:
        st.error("Missing GROQ_API_KEY. Add it to Streamlit secrets or set it as an environment variable.")
        st.stop()

    return Groq(api_key=api_key)

def log_conversation(role, content):
    """Log the conversation to the terminal."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - {role}: {content}")

def get_groq_stream_response(client: Groq, messages):
    """Stream a response from Groq Chat Completions."""
    return client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.2,
        stream=True,
    )

def display_messages():
    """Display the chat history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def handle_user_input(client: Groq, prompt: str):
    """Handle user input and get assistant response (streaming)."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    log_conversation("user", prompt)

    with st.chat_message("user"):
        st.markdown(prompt)

    # Build Groq message list:
    # - Include SYSTEM_INSTRUCTION once as system message
    # - Then include chat history so the model has context
    groq_messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
    for m in st.session_state.messages:
        groq_messages.append({"role": m["role"], "content": m["content"]})

    response_content = ""
    with st.chat_message("assistant"):
        placeholder = st.empty()
        stream = get_groq_stream_response(client, groq_messages)
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and getattr(delta, "content", None):
                response_content += delta.content
                placeholder.markdown(response_content)

    st.session_state.messages.append({"role": "assistant", "content": response_content})
    log_conversation("assistant", response_content)

# Streamlit main code for chatbot
st.title("Chat with Lucy 🤖")

if "groq_client" not in st.session_state:
    st.session_state.groq_client = get_groq_client()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pill_selected" not in st.session_state:
    st.session_state.pill_selected = False

# Initial greeting
if not st.session_state.messages:
    initial_greeting = "Greetings, Human! 👋 I'm Lucy, an AI trained to answer questions about Nethan. Curious about his projects, skills, or anything else? Just ask!😉"
    st.session_state.messages.append({"role": "assistant", "content": initial_greeting})

display_messages()

# Display pills if none selected and update state on pill selection
if not st.session_state.pill_selected:
    selected_pill = pills("", general_prompt, index=None)
    if selected_pill:
        st.session_state.pill_selected = True
        handle_user_input(st.session_state.groq_client, selected_pill)
        st.rerun()        

# Handle user input and update state to hide pills
if prompt := st.chat_input("Please ask away.."):
    st.session_state.pill_selected = True
    handle_user_input(st.session_state.groq_client, prompt)
    st.rerun()
