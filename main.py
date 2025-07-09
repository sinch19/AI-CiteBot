import os
import re
import random
import requests
from bs4 import BeautifulSoup
import traceback
import time
import json

# Attempt to import necessary libraries
try: from googlesearch import search
except ImportError: search = None; print("ERROR: 'googlesearch-python' missing.")

try: from fuzzywuzzy import fuzz
except ImportError: fuzz = None; print("ERROR: 'fuzzywuzzy' missing.")

try: import google.generativeai as genai
except ImportError: genai = None; print("ERROR: 'google-generativeai' missing.")

# ========== CONFIGURATION ========== #
# API Key is now read just-in-time inside call_gemini_llm

# Model Configuration
GEMINI_FLASH_MODEL = "gemini-1.5-flash-latest"
GEMINI_PRO_MODEL = "gemini-1.5-pro-latest"

# Safety Settings for LLM
SAFETY_SETTINGS = [
    {"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in [
        "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"
    ]
]

# Search, Scrape, and Summary Parameters
NUM_GOOGLE_RESULTS = 10
RANKING_SNIPPET_LENGTH = 1500
CONTENT_SCRAPE_LIMIT = 3500
SUMMARY_CONTEXT_LIMIT = 3500
MEMORY_FILE = "memory.json"
NUM_PAPERS_TO_RANK = 7

# ========== GOOGLE GEMINI LLM WRAPPER (Configuration Moved Inside) ========== #
def call_gemini_llm(prompt, model_name=GEMINI_FLASH_MODEL, max_output_tokens=None):
    """Calls the specified Google Gemini model with the given prompt."""
    if not genai:
        return "LLM Error: AI library not available."

    # --- Configure GenAI just before the call ---
    try:
        api_key = os.environ.get('GOOGLE_API_KEY')
        if not api_key:
            print("[LLM ERROR] GOOGLE_API_KEY not found in environment.")
            # Check if .env file exists as a hint
            if not os.path.exists(".env"):
                 print("[LLM HINT] Also, '.env' file not found in project directory.")
            return "LLM Error: API Key not found."
        genai.configure(api_key=api_key)
        # print("[LLM INFO] Google AI Configured for this call.") # Optional log
    except Exception as config_e:
        print(f"[LLM ERROR] Failed to configure Google AI: {config_e}")
        return f"LLM Error: Configuration failed ({type(config_e)._name_})."
    # --- End GenAI Configuration ---

    # print(f"[LLM INFO] Calling Model: {model_name}") # Can be verbose
    try:
        model = genai.GenerativeModel(model_name)
        generation_config = genai.types.GenerationConfig(max_output_tokens=max_output_tokens) if max_output_tokens else None
        response = model.generate_content(
            prompt,
            safety_settings=SAFETY_SETTINGS,
            generation_config=generation_config
        )
        if not response.candidates:
            block_reason = "Unknown"
            try: block_reason = response.prompt_feedback.block_reason.name
            except Exception: pass
            print(f"[LLM WARNING] Response blocked. Reason: {block_reason}")
            return f"LLM Error: Response blocked ({block_reason})."
        try:
            return response.text
        except ValueError:
             if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                 return response.candidates[0].content.parts[0].text
             else:
                 # If block_reason wasn't defined above (e.g., prompt_feedback was None), provide a generic message
                 block_reason_text = f"Reason: {block_reason}" if 'block_reason' in locals() else "Unknown reason."
                 print(f"[LLM ERROR] Text extraction failed. {block_reason_text}")
                 return f"LLM Error: Failed text extraction ({block_reason_text})."


    except Exception as e:
        print(f"[LLM ERROR] API call failed: {type(e)._name_} - {e}")
        # Log specific details for common errors like DefaultCredentialsError
        if "DefaultCredentialsError" in str(type(e)):
             print("[LLM DEBUG] DefaultCredentialsError suggests ADC fallback - check API Key config timing.")
        # print(traceback.format_exc()) # Uncomment for full traceback if needed
        return f"LLM Error: API call failed ({type(e)._name_})."


# ========== MEMORY SYSTEM ========== #
def load_memory():
    if not os.path.exists(MEMORY_FILE):
         print(f"[Memory INFO] Memory file '{MEMORY_FILE}' not found. Starting fresh.")
         return {}
    try:
        with open(MEMORY_FILE, 'r') as f: return json.load(f)
    except Exception as e:
        print(f"[Memory WARNING/ERROR] Problem loading memory file '{MEMORY_FILE}': {e}")
        return {} # Return empty on any error

def save_memory(memory):
    try:
        with open(MEMORY_FILE, 'w') as f: json.dump(memory, f, indent=2)
    except Exception as e:
        print(f"[Memory ERROR] Failed to save memory to '{MEMORY_FILE}': {e}")


# ========== AGENTS ========== #

def generation_agent(topic):
    if search is None: return []
    sites = [
        "arxiv.org", "nature.com", "ncbi.nlm.nih.gov",
        "science.org", "sciencedaily.com", "researchgate.net",
        "theconversation.com", "pubmed.ncbi.nlm.nih.gov", "ieeexplore.ieee.org",
        "dl.acm.org", "jamanetwork.com", "thelancet.com", "cell.com"
    ]
    site_query = " OR ".join([f"site:{s}" for s in sites])
    query = f'"{topic}" ({site_query})'
    print(f"[Search INFO] Querying Google ({NUM_GOOGLE_RESULTS} results)...")
    results = []
    try:
        results = list(search(query, num_results=NUM_GOOGLE_RESULTS, lang='en'))
        print(f"[Search INFO] Found {len(results)} URLs.")
    except Exception as e: print(f"[Search ERROR] Google search failed: {e}")
    return results

def scrape_content(url):
    print(f"[Scrape INFO] Scraping: {url}")
    try:
        if not isinstance(url, str) or not url.startswith(('http://', 'https://')) or \
           "google.com/search?" in url or "scholar.google." in url or url.lower().endswith(".pdf"):
            return None, None # Skip invalid/unwanted URLs quietly
        headers = {"User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"} # Be a friendly bot
        res = requests.get(url, headers=headers, timeout=15)
        res.raise_for_status()
        if 'html' not in res.headers.get('Content-Type', '').lower(): return None, None

        soup = BeautifulSoup(res.text, 'html.parser')
        title = soup.title.string.strip() if soup.title and soup.title.string else url
        main_content = soup.find('article') or soup.find('main') or soup.body
        paragraphs = main_content.find_all('p') if main_content else []
        text = " ".join(p.get_text(" ", strip=True) for p in paragraphs)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        content = text[:CONTENT_SCRAPE_LIMIT]

        if not content or len(content) < 150: return title, None
        return title, content
    except requests.exceptions.RequestException as e: print(f"[Scrape ERROR] Request failed for {url}: {e}"); return url, None
    except Exception as e: print(f"[Scrape ERROR] Failed processing {url}: {e}"); return url, None

def reflection_agent(title, content, query):
    if not all([title, content, query, fuzz]): return False
    try: return fuzz.token_set_ratio(content, query) >= 60
    except Exception as e: print(f"[Reflection ERROR] Fuzzy matching error: {e}"); return False

def ranking_agent(papers_data, topic):
    if not papers_data: return None, "No papers provided for ranking."

    options_text = ""
    print(f"[Ranking INFO] Preparing {len(papers_data)} papers for ranking...")
    for i, paper in enumerate(papers_data):
        content_snippet = paper.get('content', '')[:RANKING_SNIPPET_LENGTH]
        title = paper.get('title', 'Untitled')
        options_text += f"Option {i+1}:\nTitle: {title}\nSnippet: {content_snippet}...\n\n"

    num_options = len(papers_data)
    prompt = (
        f"Topic: '{topic}'\nAnalyze the following {num_options} text snippets. "
        f"Identify the SINGLE BEST option (most relevant, informative, suitable source).\n"
        f"Respond ONLY with the option number (1-{num_options}). No other text.\n\n{options_text}"
    )

    llm_response = call_gemini_llm(prompt, model_name=GEMINI_FLASH_MODEL, max_output_tokens=10)
    print(f"[Ranking DEBUG] LLM Response: '{llm_response}'")

    selected_index = 0
    user_friendly_reason = "Defaulted to first relevant paper found."

    if llm_response.startswith("LLM Error:"):
        user_friendly_reason = f"Ranking failed ({llm_response}); using first paper."
    else:
        match = re.search(r"^\s*(\d+)\s*$", llm_response)
        if match:
            try:
                option_number = int(match.group(1))
                if 1 <= option_number <= num_options:
                    selected_index = option_number - 1
                    user_friendly_reason = f"AI selected option {option_number} as most relevant."
                    print(f"[Ranking INFO] AI selected index {selected_index}.")
                else: user_friendly_reason = f"AI gave invalid option ({option_number}); using first paper."
            except ValueError: user_friendly_reason = f"Could not parse AI digit ('{llm_response}'); using first paper."
        else: user_friendly_reason = f"AI response format unexpected ('{llm_response}'); using first paper."

    return papers_data[selected_index], user_friendly_reason

def summarization_agent(title, content):
    if not content: return "Content was empty, cannot summarize."
    content_for_summary = content[:SUMMARY_CONTEXT_LIMIT]
    prompt = (
        f"Generate a detailed, single paragraph summary (approx 150-200 words) capturing key points, findings, and conclusions of the text below.\n"
        f"Title: {title}\nContent Snippet:\n{content_for_summary}\n\nDetailed Summary Paragraph:"
    )
    summary = call_gemini_llm(prompt, model_name=GEMINI_PRO_MODEL, max_output_tokens=800)
    if summary.startswith("LLM Error:"):
        print(f"[Summarization WARN] LLM failed: {summary}")
        return f"LLM Summary failed ({summary})."
    print(f"[Summarization INFO] Summary generated ({len(summary.split())} words).")
    return summary

def related_topics_agent(original_topic, paper_title, paper_content_or_summary):
    if not paper_content_or_summary: return []
    context_snippet = paper_content_or_summary[:500]
    prompt = (
        f"Based on topic '{original_topic}' and paper '{paper_title}', suggest 3 distinct related research topics/keywords. "
        f"Format ONLY as a comma-separated list (e.g., Topic A, Topic B, Topic C).\nSnippet: '{context_snippet}...'"
    )
    topics_str = call_gemini_llm(prompt, model_name=GEMINI_FLASH_MODEL, max_output_tokens=100)
    if topics_str.startswith("LLM Error:"):
        print(f"[Related Topics WARN] LLM failed: {topics_str}")
        return []
    related_topics = [t.strip() for t in topics_str.split(',') if t.strip()]
    print(f"[Related Topics INFO] Suggested: {related_topics[:3]}")
    return related_topics[:3]

# ========== SUPERVISOR ========== #

def supervisor(topic):
    print(f"\n=== Analyzing Topic: '{topic}' ===")
    start_time = time.time()
    memory = load_memory()

    # 1. Check cache
    if topic in memory and "best_paper_details" in memory[topic]:
        print(f"CACHE HIT for: {topic}")
        cached_data = memory[topic]
        final_cached_data = { # Ensure all keys expected by app.py are present
            "status": "success_cached",
            "link": cached_data.get("best_paper_details", {}).get("link", "#"),
            "title": cached_data.get("best_paper_details", {}).get("title", "N/A"),
            "llm_summary": cached_data.get("llm_summary", "Cached summary not found."),
            "ranking_reason": cached_data.get("ranking_reason", "Cached reason not found."),
            "related_topics": cached_data.get("related_topics", []),
            "considered_links": cached_data.get("considered_links", []),
        }
        duration = round(time.time() - start_time, 2)
        print(f"--- Analysis From Cache Finished ({duration}s) ---\n")
        return final_cached_data

    print(f"CACHE MISS for: {topic}. Starting new analysis...")
    # 2. Generation
    urls = generation_agent(topic)
    if not urls: return {"status": "error", "message": "Failed to find any URLs via Google Search."}

    # 3. Scraping & Reflection
    papers = []
    urls_to_process = urls[:15]
    print(f"Processing up to {len(urls_to_process)} URLs...")
    for url in urls_to_process:
        title, content = scrape_content(url)
        if content and title and reflection_agent(title, content, topic):
            papers.append({"title": title, "content": content, "link": url})
            if len(papers) >= NUM_PAPERS_TO_RANK + 2: break # Optimization

    if not papers: return {"status": "error", "message": "No relevant content found."}
    print(f"Found {len(papers)} relevant documents.")

    # 4. Ranking
    papers_to_rank = papers[:NUM_PAPERS_TO_RANK]
    considered_links_data = [{"title": p.get("title"), "link": p.get("link")} for p in papers_to_rank]
    selected_paper_dict, ranking_reason = ranking_agent(papers_to_rank, topic)
    if selected_paper_dict is None: return {"status": "error", "message": f"Ranking failed: {ranking_reason}"}
    print(f"Selected paper: '{selected_paper_dict.get('title')}'")

    # 5. Summarization
    llm_summary = summarization_agent(selected_paper_dict.get('title'), selected_paper_dict.get('content'))

    # 6. Related Topics
    context_for_related = llm_summary if not llm_summary.startswith("LLM Error:") else selected_paper_dict.get('content','')
    related_topics = related_topics_agent(topic, selected_paper_dict.get('title'), context_for_related)

    # 7. Prepare & Cache Output
    final_data = {
        "status": "success",
        "link": selected_paper_dict.get("link"),
        "title": selected_paper_dict.get("title"),
        "llm_summary": llm_summary,
        "ranking_reason": ranking_reason,
        "related_topics": related_topics,
        "considered_links": considered_links_data,
    }
    memory_entry = { # Data to cache
        "best_paper_details": { "title": final_data["title"], "link": final_data["link"] },
        "llm_summary": final_data["llm_summary"],
        "ranking_reason": final_data["ranking_reason"],
        "related_topics": final_data["related_topics"],
        "considered_links": final_data["considered_links"]
    }
    memory[topic] = memory_entry
    save_memory(memory)
    print("[Cache INFO] Results saved.")

    # 8. Log duration & return
    duration = round(time.time() - start_time, 2)
    print(f"--- Analysis Finished ({duration}s) --- Reason: {ranking_reason}\n")
    return final_data

# ========== MAIN EXECUTION (for testing) ========== #
if __name__ == "__main__":
    test_topic = input("Enter research topic for testing : ")
    if test_topic:
        result = supervisor(test_topic)
        print("\n--- Supervisor Result ---")
        for key, value in result.items():
            if isinstance(value, list): print(f"{key}: (list with {len(value)} items)")
            elif isinstance(value, str) and len(value) > 100: print(f"{key}: {value[:100]}...")
            else: print(f"{key}: {value}")
        print("-------------------------\n")
    else: print("No topic entered. Exiting test.")