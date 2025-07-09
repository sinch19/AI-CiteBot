import streamlit as st
import base64
import re
import os
import traceback
from dotenv import load_dotenv 

load_dotenv() 

# NOW import main - its top-level code will run AFTER .env is loaded
from main import supervisor, call_gemini_llm, GEMINI_FLASH_MODEL

# ========== API Key Check ==========
# This check still reads the key loaded from .env
google_api_key = os.environ.get('GOOGLE_API_KEY')
show_api_key_warning = False
if not google_api_key:
    show_api_key_warning = True
    # Updated message slightly
    print("WARNING: GOOGLE_API_KEY env var not detected in environment or .env file.")
# ===================================

# ========== PAGE CONFIGURATION ========== #
st.set_page_config(
    page_title="Mavericks AI Research Assistant",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========== STYLING ========== #
# Function to load local CSS file
def load_local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        # print(f"Successfully loaded CSS from {file_name}") # Quieter log
    except FileNotFoundError:
        st.error(f"Styling Error: Cannot find the CSS file '{file_name}'. Make sure it's in the same directory as app.py.")
        print(f"ERROR: CSS file '{file_name}' not found.")
    except Exception as e:
        st.error(f"An error occurred while loading CSS: {e}")
        print(f"ERROR loading CSS: {e}")
        print(traceback.format_exc())

# Load the external CSS file
load_local_css("style.css")

# ========== STREAMLIT LAYOUT ========== #
st.markdown('<h1 class="title">AI-CiteBot</h1>', unsafe_allow_html=True)

if show_api_key_warning:
    st.warning(
        """*WARNING: Google AI API Key Not Found!*
        Create a .env file in the project directory with GOOGLE_API_KEY=YOUR_KEY or set the environment variable.""",
        icon="⚠"
    )

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'topic' not in st.session_state:
    st.session_state.topic = ""
if 'follow_up_question' not in st.session_state:
    st.session_state.follow_up_question = ""
if 'follow_up_answer' not in st.session_state:
    st.session_state.follow_up_answer = ""


# --- Input Form ---
with st.form(key="search_form"):
    query = st.text_input(
        "🔍 Enter Research Topic:",
        key="query_input",
        value=st.session_state.topic,
        placeholder="e.g., quantum entanglement applications"
    )
    submitted = st.form_submit_button(" Analyze Topic ")

    if submitted:
        current_query = st.session_state.query_input
        if current_query:
            st.session_state.topic = current_query
            st.session_state.results = None # Reset results
            st.session_state.follow_up_question = "" # Reset follow-up
            st.session_state.follow_up_answer = "" # Reset follow-up

            # API key presence is checked implicitly now when call_gemini_llm runs
            # We can maybe remove this specific check here if confident .env/env var is the only way
            # Let's keep the UI warning based on initial check though
            if show_api_key_warning:
                 st.error("Cannot run analysis: GOOGLE_API_KEY not found in environment or .env file.")
            else:
                with st.spinner(f"Analyzing '{current_query}'... Please wait."):
                    try:
                        print(f"--- Running supervisor for '{current_query}' ---")
                        results = supervisor(current_query)
                        st.session_state.results = results
                        print(f"--- Supervisor finished for '{current_query}' ---")
                    except Exception as e:
                        print(f"--- Supervisor ERROR for '{current_query}' ---")
                        print(traceback.format_exc())
                        st.error(f"An unexpected error occurred during analysis: {e}")
                        st.session_state.results = {"status": "error", "message": f"Analysis Error: {e}"}
        else:
            st.warning("Please enter a research topic in the input field.")

# ========== DISPLAY RESULTS ========== #
# (Rest of the file remains the same as v2.16 - PDF Download Removed)
if st.session_state.results:
    results = st.session_state.results
    st.markdown('<div class="results-container">', unsafe_allow_html=True) # Main container

    if results.get("status", "error").startswith("success"):
        # --- Display Core Result ---
        st.markdown("###  Most Relevant Paper Found ")
        st.markdown(f"<p><strong>Title:</strong> {results.get('title', 'N/A')}</p>", unsafe_allow_html=True)
        main_link = results.get("link", "#")
        link_html = f'<p><strong>Source:</strong> <a href="{main_link}" target="_blank">View Source Page</a></p>'
        st.markdown(link_html, unsafe_allow_html=True)

        # --- Summary Box ---
        summary_text = results.get('llm_summary', 'Not available.')
        st.markdown('<div class="summary-box">', unsafe_allow_html=True)
        st.markdown(f"<strong> Paper Summary:</strong>", unsafe_allow_html=True)
        st.markdown(f"<p>{summary_text}</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # --- Follow-up Q&A Section ---
        st.markdown("---")
        follow_up_header = f"#### You can ask a follow-up question on '{st.session_state.topic}':"
        st.markdown(follow_up_header)

        follow_up_q = st.text_area(
            label="Enter queries:", key="follow_up_input",
            value=st.session_state.follow_up_question, height=100
        )

        ask_follow_up = st.button("💬 Ask Maira", key="follow_up_button")

        if ask_follow_up:
            st.session_state.follow_up_question = follow_up_q
            if not follow_up_q:
                st.warning("Please enter a follow-up question.")
            # Check summary status before potentially costly LLM call
            elif summary_text == 'Not available.' or summary_text.startswith("LLM Summary failed") or summary_text.startswith("LLM Error"):
                 st.error("Cannot ask follow-up: Summary is not available or failed to generate.")
            # Check for API key warning - call_gemini_llm will handle actual missing key
            elif show_api_key_warning:
                 st.error("Cannot ask follow-up: GOOGLE_API_KEY not found in environment or .env file.")
            else:
                follow_up_prompt = (
                    f"Consider the following summary text and the user's question. Provide a helpful and informative answer, using the summary as primary context but also leveraging your general knowledge if relevant.\n\n"
                    f"SUMMARY TEXT:\n\"\"\"\n{summary_text}\n\"\"\"\n\n"
                    f"USER'S QUESTION:\n\"\"\"\n{st.session_state.follow_up_question}\n\"\"\"\n\n"
                    f"ANSWER:"
                )
                with st.spinner("Thinking..."):
                    # Error handling now primarily within call_gemini_llm
                    answer = call_gemini_llm(follow_up_prompt, model_name=GEMINI_FLASH_MODEL, max_output_tokens=500)
                    st.session_state.follow_up_answer = answer
                    # Print LLM error if it occurred, handled by the wrapper now
                    if answer.startswith("LLM Error:"):
                         print(f"Follow-up LLM call failed: {answer}")


        if st.session_state.follow_up_answer:
             st.markdown("##### Answer:")
             if st.session_state.follow_up_answer.startswith("LLM Error:") or st.session_state.follow_up_answer.startswith("Error"):
                 st.error(st.session_state.follow_up_answer)
             else:
                 st.info(st.session_state.follow_up_answer)
        st.markdown("---")

        # --- Selection Reason Box ---
        st.markdown('<div class="reason-box">', unsafe_allow_html=True)
        st.markdown(f"<strong>🤖 Paper Selection:</strong>", unsafe_allow_html=True)
        st.markdown(f"<p>{results.get('ranking_reason', 'Not available.')}</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # --- Considered Links Box ---
        considered_links = results.get("considered_links", [])
        if considered_links:
            st.markdown('<div class="shortlist-box">', unsafe_allow_html=True)
            st.markdown("<strong>📊 Links Considered for Ranking:</strong>", unsafe_allow_html=True)
            st.markdown("<ol style='padding-left: 25px; margin-top: 10px;'>", unsafe_allow_html=True)
            for item in considered_links:
                title = item.get('title', 'Link')
                link = item.get('link', '#')
                st.markdown(f'<li><a href="{link}" target="_blank">{title}</a></li>', unsafe_allow_html=True)
            st.markdown("</ol>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # --- Related Topics Box ---
        related_topics = results.get("related_topics", [])
        if related_topics:
            st.markdown('<div class="topics-box">', unsafe_allow_html=True)
            st.markdown(f"<strong> Suggested Related Topics:</strong>", unsafe_allow_html=True)
            st.markdown("<ul>", unsafe_allow_html=True)
            for topic_item in related_topics:
                st.markdown(f"<li>{topic_item}</li>", unsafe_allow_html=True)
            st.markdown("</ul>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    elif results.get("status") == "error":
        st.error(f"⚠ Analysis Failed: {results.get('message', 'An unknown error occurred.')}")

    else: # Handle unexpected status
        st.warning(f"Received an unexpected status: {results.get('status')}")
        print(f"Unexpected results status: {results}")

    st.markdown('</div>', unsafe_allow_html=True) # Close results-container

# --- Footer ---
st.markdown('<p class="footer">Mavericks AI Research Assistant - Final Round</p>', unsafe_allow_html=True)