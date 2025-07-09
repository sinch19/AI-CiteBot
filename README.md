# AI-CiteBot
AI CiteBot

AI CiteBot is an AI-driven research assistant that retrieves, ranks, and summarizes research papers based on a user’s query. It combines the power of web scraping and large language models (LLMs) to offer relevant results with minimal effort from the user. The interface is built using Streamlit and served via a Flask wrapper for better integration and launch flexibility.

---

Setup Instructions

1. Clone the Repository

2. Install Dependencies

Make sure you have Python installed (see prerequisites below), then run:

AI CiteBot

AI CiteBot is an AI-driven research assistant that retrieves, ranks, and summarizes research papers based on a user’s query. It combines the power of web scraping and large language models (LLMs) to offer relevant results with minimal effort from the user. The interface is built using Streamlit and served via a Flask wrapper for better integration and launch flexibility.

---

Setup Instructions (Local Environment)

1. Download Project Files

Download all the project files and place them in a single directory on your system.

2. Set Up a Virtual Environment (Optional but Recommended)

Activate the virtual environment:

- On Windows:
venv\Scripts\activate

- On macOS/Linux:
source venv/bin/activate


3. Install Dependencies
Make sure you have Python installed, then run:
pip install -r requirements.txt


4. Add API Key

Create a `.env` file in the same directory as the code and add your API key:
API_KEY=your_api_key_here

> Note: You can also paste your API key directly inside `main.py` for quick testing, but using a `.env` file is recommended for security.

5. Run the App

Start the Streamlit application:
streamlit run main.py
