import streamlit as st
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
import re
import difflib 
import random
import unicodedata
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
import math

# Utility to clean meta/instructional lines from LLM output
def clean_llm_output(text):
    # Remove lines like (Note: I'll respond accordingly based on the user's query, following the structured process outlined above.)
    lines = text.split('\n')
    filtered = [line for line in lines if not re.search(r'\(note:.*structured process.*\)', line, re.IGNORECASE)]
    return '\n'.join(filtered).strip()

# --- Page Config ---
st.set_page_config(page_title="91Trucks", layout="wide")

# --- Load Environment Variables ---
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN") or st.secrets.get("HUGGINGFACEHUB_API_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

# --- Helper Functions (Copied and adapted from api.py) ---

def clean_price(price):
    if isinstance(price, str):
        price_lower = price.lower().strip()
        if 'coming soon' in price_lower:
            return 'Coming Soon'
        price = re.sub(r'[₹,]', '', price_lower)
        if 'lakh' in price:
            value = float(re.findall(r'(\d+\.?\d*)', price)[0])
            return value
        elif 'crore' in price:
            value = float(re.findall(r'(\d+\.?\d*)', price)[0]) * 100
            return value
    return None

@st.cache_data
def load_data():
    df_main = pd.read_csv('data/finalist_data.csv')
    df_qna = pd.read_csv('data/91Trucks_QnAs.csv')
    df_main["Vehicle Image"] = df_main["Vehicle Image"].fillna("")
    df_main["Price"] = df_main["Vehicle Price"].apply(clean_price)
    return df_main, df_qna

@st.cache_resource
def get_vector_store(_df_main, _df_qna):
    with st.spinner("Initializing knowledge base... this may take a moment."):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = []
        for _, row in _df_main.iterrows():
            price_str = f"{row['Price']} Lakh" if pd.notnull(row['Price']) else "N/A"
            text = (f"Brand: {row['Brand Name']}, Model: {row['Model Name']}, Vehicle Name: {row['Vehicle Name']}, "
                    f"Electric: {row['Electric']}, Price: {price_str}, Fuel Type: {row['Fuel Type']}, "
                    f"Variant: {row['Variant Name']}, Power: {row['Power']}, Image: {row['Vehicle Image']}, "
                    f"End Point: {row.get('End Point', '')}")
            docs.append(text)
        for _, row in _df_qna.iterrows():
            docs.append(f"Question: {row['question']} Answer: {row['answer']}")
        documents = text_splitter.create_documents(docs)
        embeddings = HuggingFaceEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

@st.cache_resource
def get_qa_chain(_vector_store):
    prompt_template = """
    As a commercial vehicle expert, your goal is to provide detailed information and recommendations about commercial vehicles based on the provided context.

    When a user asks a question, follow this structured process:

    1.  **Identify the User's Need:** Analyze the user's query to understand if they are asking about a specific vehicle, comparing vehicles, asking about a brand, or have a general question about suitability for a task (e.g., "best truck for construction").

    2.  **Gather Key Information:** Systematically extract all relevant details from the provided **Context**. The context is your only source of truth. Key information includes:
        *   Vehicle Name and Brand
        *   Price
        *   Key Specifications (Payload, GVW, Engine Power, Fuel Type, Mileage)
        *   Features and description.

    3.  **Synthesize and Structure the Analysis:** Present the gathered information in a clear, well-structured format. Use markdown for headings and lists. For example:

        *   **For a specific vehicle query:**
            ### [Vehicle Name]
            *   **Price:** ...
            *   **Payload:** ...
            *   **GVW:** ...
            *   **Fuel Type:** ...
            *   **Key Features:** ...
            only show those features where there is value present for features having not vavailable , dont show those features.

        *When a user asks about a particular vehicle brand (e.g. tata , mahindra , force, mahindra , euler, eicher ..etc, respond by showing only the top 5 most relevant or popular models from that brand. Do not display the full list, even if more models are available.

Instructions for generation:
Display only the top 5 model names (no more).

You can rank them based on relevance, popularity, or alphabetical order if popularity is unknown.

Keep the tone informative and helpful.

Do not mention that there are more models unless the user specifically asks.

Keep the reply concise (within 80 words).

        *   **For a comparison query ("A vs B"):**
            Present a side-by-side comparison table or list.

        *   **For a brand query:**
            List 3-4 popular models from that brand with their key specifications.

        *   **For "reasons to buy" or recommendation queries:**
            Always mention the specific vehicle name in your response. Focus on the key benefits, pros, and reasons why this vehicle would be a good choice. Highlight unique features, cost-effectiveness, reliability, and suitability for specific use cases.

    4.  **Analyze and Justify (if applicable):** Based *only* on the extracted information, provide a brief analysis.
        *   For example, if the user asks for a vehicle for a specific purpose (e.g., "city logistics"), analyze the specs to justify a recommendation: "With its compact size and good mileage, the [Vehicle Name] is well-suited for city logistics."
        *   Highlight key strengths or weaknesses based on the data. For instance, "This vehicle offers a high payload for its price segment."

    5.  **Provide a Recommendation Summary (if applicable):**
        *   If the user's query implies a need for a recommendation, conclude with a clear summary.
        *   Clearly state what the vehicle is best used for.
        *   Mention any potential trade-offs based on the data (e.g., "It has high power, but the mileage is lower than its competitors.").
        *   If you cannot make a recommendation from the context, state that. Do not make up information.

    **Important:** When recommending or discussing specific vehicles, always include the exact vehicle name in your response so it can be properly identified and displayed with additional information.

    **Context:**
    {context}

    **Question:**
    {question}

    **Answer:**
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama3-70b-8192", temperature=0.2, max_tokens=2048)
    return RetrievalQA.from_chain_type(llm=llm, retriever=_vector_store.as_retriever(), chain_type_kwargs={"prompt": prompt}, return_source_documents=True, input_key="question")

def get_brand_models(brand_name, df, category=None):
    brand_df = df[df['Brand Name'].str.lower() == brand_name.lower()]
    if category:
        found_category = False
        for col in ['Category Name', 'category_name', 'Vehicle Type', 'Category']:
            if col in brand_df.columns:
                # Only filter if the column is not all NaN
                if brand_df[col].notna().any():
                    brand_df = brand_df[brand_df[col].str.lower().str.contains(category, na=False)]
                    found_category = True
        if not found_category:
            st.warning(f"Sorry, I can't filter by category '{category}' because the data is missing category information.")
            return []
    if brand_df.empty:
        return []
    model_names = brand_df['Vehicle Name'].dropna().unique()
    if 'Popularity' in brand_df.columns:
        brand_df = brand_df.sort_values('Popularity', ascending=False)
        model_names = brand_df['Vehicle Name'].dropna().unique()
    else:
        model_names = sorted(model_names)
    top_model_names = model_names[:5]
    models_data = []
    for name in top_model_names:
        vehicle_info = get_vehicle_data(name, df)
        if vehicle_info:
            models_data.append(vehicle_info)
    return models_data

def is_greeting(query):
    GREETINGS = {"hi", "hello", "namaste", "hey", "good morning", "good afternoon", "good evening", "greetings", "good day", "hii", "hiii", "heyy", "hey there", "hello there", "yo", "sup", "hola", "bonjour", "ciao", "salaam", "shalom", "howdy", "hiya", "wassup", "what's up", "goodbye", "bye"}
    q = query.strip().lower()
    # Remove punctuation and extra whitespace
    q = re.sub(r'[^a-zA-Z ]', '', q)
    q = ' '.join(q.split())
    # If all words in the query are greetings, treat as greeting
    words = set(q.split())
    if words and all(word in GREETINGS for word in words):
        return True
    # If the whole query matches a greeting...
    if q in GREETINGS:
        return True
    return False

def extract_display_name(query, names):
    query_clean = query.strip().lower()
    for name in names:
        if query_clean == name.strip().lower(): return name
    return None

def find_qna_answer(query, qna_df, threshold=0.85):
    q = query.strip().lower()
    best_score = 0
    best_answer = None
    for _, row in qna_df.iterrows():
        question = str(row['question']).strip().lower()
        score = difflib.SequenceMatcher(None, q, question).ratio()
        if score > best_score:
            best_score = score
            best_answer = row['answer']
    if best_score >= threshold:
        return best_answer
    return None

def find_vehicle_in_text(text, names, threshold=0.7, exclude=None):
    text_lower = text.lower().strip()
    exclude = exclude or set()
    # 1. Try exact match
    for name in names:
        if name.lower().strip() == text_lower and name not in exclude:
            return name
    # 2. Try full-string substring match
    for name in names:
        if text_lower in name.lower() and name not in exclude:
            return name
    # 3. Fuzzy match, but exclude already matched names
    best_match = None
    best_score = 0
    for name in names:
        if name in exclude:
            continue
        score = difflib.SequenceMatcher(None, text_lower, name.lower()).ratio()
        if score > best_score:
            best_score = score
            best_match = name
    if best_score >= threshold:
        return best_match
    return None

def get_vehicle_data(display_name, df):
    display_name_clean = display_name.lower().strip()
    vehicle_data_df = df[df['Vehicle Name'].str.lower().str.strip().str.contains(display_name_clean, na=False)]
    if not vehicle_data_df.empty:
        prices = vehicle_data_df['Price'].dropna().tolist()
        # Separate numeric prices and 'Coming Soon'
        numeric_prices = [p for p in prices if isinstance(p, (int, float))]
        coming_soon = any(isinstance(p, str) and 'coming soon' in p.lower() for p in prices)
        # Show numeric price if available, otherwise Coming Soon, otherwise Not available
        if numeric_prices:
            min_price, max_price = min(numeric_prices), max(numeric_prices)
            price_str = f"₹{min_price} Lakh" if min_price == max_price else f"₹{min_price} - {max_price} Lakh"
        elif coming_soon:
            price_str = "Coming Soon"
        else:
            price_str = "Not available"
        fuel_types_str = ', '.join(vehicle_data_df['Fuel Type'].dropna().unique()) or "Not available"
        power_str = ', '.join(vehicle_data_df['Power'].dropna().unique()) or "Not available"
        variants_str = ', '.join(vehicle_data_df['Variant Name'].dropna().unique()) or "Not available"

        pros_list = vehicle_data_df['Pros'].dropna().unique().tolist() if 'Pros' in vehicle_data_df.columns else []
        cons_list = vehicle_data_df['Cons'].dropna().unique().tolist() if 'Cons' in vehicle_data_df.columns else []
        pros_str = ' | '.join(str(p) for p in pros_list if str(p) != 'nan')
        cons_str = ' | '.join(str(c) for c in cons_list if str(c) != 'nan')
        
        row = vehicle_data_df.iloc[0].to_dict()
        data = {
            "Vehicle Name": row['Vehicle Name'],
            "Brand Name": row['Brand Name'],
            "Vehicle Image": row['Vehicle Image'],
            "Description": row.get('Vehicle Description', "Not available"),
            "Average Rating": row.get('Average Rating', "Not available"),
            "End Point": row.get('End Point'),
            "Price": price_str,
            "Fuel Type": fuel_types_str,
            "Power": power_str,
            "Variant Name": variants_str,
            "Pros": pros_str if pros_str else "Not available",
            "Cons": cons_str if cons_str else "Not available",
            "Category Name": row.get('Category Name', None),
        }
        return data
    return None

def get_display_name_suggestion(query, names, df):
    query_clean = query.strip().lower()
    suggestions = []
    for name in names:
        if query_clean in name.strip().lower() and query_clean != name.strip().lower():
            vehicle_data_df = df[df['Vehicle Name'].str.lower().str.strip().str.contains(name.strip().lower(), na=False)]
            if not vehicle_data_df.empty:
                prices = vehicle_data_df['Price'].dropna().tolist()
                numeric_prices = [p for p in prices if isinstance(p, (int, float))]
                coming_soon = any(isinstance(p, str) and 'coming soon' in p.lower() for p in prices)
                price_str = "Not available"
                if coming_soon:
                    price_str = "Coming Soon"
                elif numeric_prices:
                    min_price, max_price = min(numeric_prices), max(numeric_prices)
                    price_str = f"₹{min_price} Lakh" if min_price == max_price else f"₹{min_price} - {max_price} Lakh"
                fuel_types = vehicle_data_df['Fuel Type'].dropna().unique().tolist()
                fuel_types_str = ', '.join(fuel_types) if fuel_types else "Not available"
                powers = []
                for p in vehicle_data_df['Power'].dropna().unique():
                    try:
                        powers.append(float(re.findall(r'\d+\.?\d*', str(p))[0]))
                    except Exception:
                        continue
                power_str = ', '.join(vehicle_data_df['Power'].dropna().unique()) if powers else "Not available"
                ratings = vehicle_data_df['Average Rating'].dropna().tolist()
                avg_rating = None
                if ratings:
                    try:
                        avg_rating = sum([float(r) for r in ratings]) / len(ratings)
                    except Exception:
                        avg_rating = None
                rating_str = f"{avg_rating:.1f}" if avg_rating and avg_rating >= 3 else None
                vehicle_image = vehicle_data_df['Vehicle Image'].iloc[0] if 'Vehicle Image' in vehicle_data_df.columns else None
                end_point = vehicle_data_df['End Point'].iloc[0] if 'End Point' in vehicle_data_df.columns else None
                suggestion = {
                    "Model Name": name,
                    "Price": price_str,
                    "Fuel Type": fuel_types_str,
                    "Power": power_str,
                }
                if rating_str:
                    suggestion["Average Rating"] = rating_str
                if vehicle_image:
                    suggestion["Vehicle Image"] = vehicle_image
                if end_point:
                    suggestion["End Point"] = end_point
                suggestions.append(clean_vehicle_row(suggestion))
    return suggestions if suggestions else None

def get_vehicle_comparison_data(query, df, names_set):
    # Split on 'vs' or 'versus', strip and lower-case both sides
    parts = re.split(r'\s+vs\s+|\s+versus\s+', query.lower())
    if len(parts) != 2:
        return None, "Please provide two vehicles to compare using 'vs', like 'Tata Ace vs Mahindra Jeeto'."
    v1_name = find_vehicle_in_text(parts[0].strip(), names_set)
    v2_name = find_vehicle_in_text(parts[1].strip(), names_set, exclude={v1_name} if v1_name else set())
    if not v1_name or not v2_name:
        return None, "I couldn't identify one or both of the vehicles. Please check the names and try again."
    v1_data = get_vehicle_data(v1_name, df)
    v2_data = get_vehicle_data(v2_name, df)
    return [v1_data, v2_data], None

def clean_vehicle_row(row):
    result = {}
    for key, value in row.items():
        val = str(value).strip().lower() if value is not None else ""
        if val not in ["not available", "n/a", "-", ""] and not (isinstance(value, float) and math.isnan(value)):
            result[key] = value
    return result

# --- Helper Functions (add at the top, after imports) ---
def is_pronoun_followup(query):
    pronouns = ["it", "this", "that", "these", "those"]
    helping_verbs = ["is it", "does it", "can it", "will it", "has it", "have it", "was it", "were it"]
    q = query.lower()
    return any(p in q.split() for p in pronouns) or any(hv in q for hv in helping_verbs)

def is_family_query(query, brand_names):
    q = query.lower()
    # If query contains 'truck' or 'trucks' and at least one other word, treat as family/brand query
    if ("truck" in q or "trucks" in q) and len(q.split()) > 1:
        return q.split()[0]  # Return the first word as the likely brand
    for brand in brand_names:
        if brand in q and ("truck" in q or "trucks" in q or "family" in q or "range" in q):
            return brand
    return None

# --- Helper Functions (add after is_family_query) ---
def extract_category(query):
    categories = ["truck", "trucks", "van", "vans", "pickup", "pickups", "auto", "autos", "bus", "buses"]
    for cat in categories:
        if cat in query:
            return cat.rstrip('s')  # normalize to singular
    return None

# --- Helper Functions (add after extract_category) ---
def is_new_intent_query(query):
    keywords = [
        "cheapest", "best", "most expensive", "top", "lowest", "highest", "price under", "price below", "budget", "affordable"
    ]
    q = query.lower()
    return any(kw in q for kw in keywords)

# --- UI Rendering Functions ---

def display_vehicle_card(vehicle_data):
    with st.container(border=True):
        if vehicle_data.get("Vehicle Image"):
            st.image(vehicle_data["Vehicle Image"], width=200)
        st.subheader(vehicle_data.get("Vehicle Name", "Unknown Vehicle"))
        
        details_map = {
            "Price": vehicle_data.get("Price"),
            "Power": vehicle_data.get("Power"),
            "Fuel Type": vehicle_data.get("Fuel Type"),
            "Average Rating": vehicle_data.get("Average Rating"),
            "Variant(s)": vehicle_data.get("Variant Name")
        }
        
        details = ""
        for key, value in details_map.items():
            if value and str(value).strip().lower() not in ["not available", "nan", "n/a", ""]:
                details += f"**{key}:** {value}<br>"
        st.markdown(details, unsafe_allow_html=True)
        
        description = vehicle_data.get("Description")
        if description and str(description).strip().lower() not in ["not available", "nan", "n/a", ""]:
            with st.expander("View Description"):
                st.markdown(description)

        if vehicle_data.get("Pros") and vehicle_data.get("Pros") != "Not available":
            with st.expander("View Pros"):
                for pro in vehicle_data["Pros"].split(' | '):
                    st.markdown(f"• {pro.strip()}")
                
        if vehicle_data.get("Cons") and vehicle_data.get("Cons") != "Not available":
            with st.expander("View Cons"):
                for con in vehicle_data["Cons"].split(' | '):
                    st.markdown(f"• {con.strip()}")

        if vehicle_data.get("End Point"):
            st.link_button("Learn More", vehicle_data["End Point"])

def display_vehicle_with_llm_response(vehicle_data, llm_response):
    """Display vehicle information with LLM response, small image, and See More button. Show fallback if missing."""
    with st.container(border=True):
        col1, col2 = st.columns([1, 3])
        with col1:
            if vehicle_data.get("Vehicle Image"):
                st.image(vehicle_data["Vehicle Image"], width=150)
            else:
                st.write(":camera: No image available.")
        with col2:
            st.subheader(vehicle_data["Vehicle Name"])
            st.markdown(llm_response)
            if vehicle_data.get("End Point"):
                st.link_button("See More", vehicle_data["End Point"])
            else:
                st.write(":link: No additional link available.")

def display_brand_summary(brand_name, models):
    st.subheader(f"Here are some popular models from {brand_name.title()}:")
    
    for model in models:
        with st.container(border=True):
            col1, col2 = st.columns([1, 2])
            with col1:
                if model.get("Vehicle Image"):
                    st.image(model["Vehicle Image"], use_container_width=True)
            with col2:
                st.markdown(f"**{model['Vehicle Name']}**")
                details = ""
                if model.get("Price"):
                    details += f"**Price:** {model['Price']}<br>"
                if model.get("Fuel Type"):
                    details += f"**Fuel Type:** {model['Fuel Type']}<br>"
                st.markdown(details, unsafe_allow_html=True)

                if model.get("End Point"):
                    st.link_button("See more", model["End Point"])

def display_comparison_table(v1_data, v2_data):
    st.subheader("Side-by-Side Comparison")
    
    # Define which keys to show in the table and their display order
    display_keys = [
        "Price", "Power", "Fuel Type", "Average Rating", "Variant Name"
    ]
    
    # Use the full vehicle name for the table headers
    v1_name = v1_data.get("Vehicle Name", "Vehicle 1")
    v2_name = v2_data.get("Vehicle Name", "Vehicle 2")

    table_data = {"Feature": [], v1_name: [], v2_name: []}
    
    for key in display_keys:
        v1_value = v1_data.get(key, "Not available")
        v2_value = v2_data.get(key, "Not available")
        
        # Only add row if at least one vehicle has a non-empty value for this feature
        v1_valid = v1_value and str(v1_value).strip().lower() not in ["not available", "nan", "n/a", ""]
        v2_valid = v2_value and str(v2_value).strip().lower() not in ["not available", "nan", "n/a", ""]

        if v1_valid or v2_valid:
            table_data["Feature"].append(key.replace("_", " ").title())
            table_data[v1_name].append(v1_value if v1_valid else "—")
            table_data[v2_name].append(v2_value if v2_valid else "—")
            
    if table_data["Feature"]:
        df_compare = pd.DataFrame(table_data)
        st.table(df_compare.set_index("Feature"))
    else:
        st.info("No comparable features found between these two vehicles.")

def display_comparison(v1_data, v2_data, conclusion):
    st.subheader("Vehicles for Comparison")
    col1, col2 = st.columns(2)
    with col1:
        display_vehicle_card(v1_data)
    with col2:
        display_vehicle_card(v2_data)
    
    st.markdown("---")
    
    # Display the side-by-side comparison table
    display_comparison_table(v1_data, v2_data)
    
    if conclusion:
        st.markdown("---")
        st.subheader("✍️ Conclusion")
        st.markdown(conclusion)

def display_suggestions(suggestions):
    st.write("I found a few possible matches. Did you mean one of these?")
    for i, vehicle in enumerate(suggestions[:3]):
        with st.container(border=True):
            col1, col2 = st.columns([1, 3])
            with col1:
                if vehicle.get("Vehicle Image"):
                    st.image(vehicle["Vehicle Image"], use_container_width=True)
            with col2:
                st.subheader(vehicle.get('Vehicle Name') or vehicle.get('Model Name', 'Unknown Model'))
                details = ""
                if vehicle.get("Price"):
                    details += f"**Price:** {vehicle['Price']}<br>"
                if vehicle.get("Fuel Type"):
                    details += f"**Fuel Type:** {vehicle['Fuel Type']}<br>"
                st.markdown(details, unsafe_allow_html=True)
                if vehicle.get("End Point"):
                    st.link_button("Show Details", vehicle["End Point"])

def display_small_vehicle_card(vehicle_data):
    with st.container(border=True):
        cols = st.columns([1, 2])
        with cols[0]:
            if vehicle_data.get("Vehicle Image"):
                st.image(vehicle_data["Vehicle Image"], width=80)
        with cols[1]:
            st.markdown(f"**{vehicle_data.get('Vehicle Name', 'Unknown Vehicle')}**")
            price = vehicle_data.get("Price")
            if price:
                st.markdown(f"<span style='font-size:0.9em;'>Price: {price}</span>", unsafe_allow_html=True)
            fuel = vehicle_data.get("Fuel Type")
            if fuel:
                st.markdown(f"<span style='font-size:0.9em;'>Fuel: {fuel}</span>", unsafe_allow_html=True)
            if vehicle_data.get("End Point"):
                st.link_button("Details", vehicle_data["End Point"])

# --- Sidebar: Display Most Frequently Asked Questions ---
def display_faq_sidebar(df_qna=None):
    st.sidebar.markdown("""
    <div style='display: flex; align-items: center; gap: 10px;'>
        <img src='https://cdn-icons-png.flaticon.com/512/4712/4712035.png' width='36' style='border-radius: 50%;'>
        <span style='font-size: 1.3em; font-weight: bold;'>Quick Questions</span>
    </div>
    <div style='margin-top: 10px; margin-bottom: 18px; color: #555;'>
        <em>Your friendly commercial vehicle assistant!</em>
    </div>
    """, unsafe_allow_html=True)
    questions = [
        "What is 91Trucks?",
        "who is the founder of 91trucks",
        "how can i contact to the 91trucks",
        "How 91trucks can help me to get good vehicle",
        "compare tata intra v30 vs tata ace gold 2.0",
        "compare mahindra zeo vs mahindra jeeto",
        "tata trucks",
        "eicher trucks",
        "What is the primary focus of 91Trucks' digital platform?",
        "What is the significance of 91Trucks' physical retail stores?",
        "What is the role of 91Trucks in the broader logistics and transportation sectors?",
        "why choose 91trucks rather than other.",
        "what is the price of tata intra v30",
        "show me some tata buses, trucks, auto-rickshaw",
        "Most Expensive vehicle",
    ]
    for q in questions:
        if st.sidebar.button(q, key=f"faq_{q}"):
            st.session_state.user_input = q
            st.rerun()

# --- Streamlit UI and Main Logic ---

st.title("🚚 91Trucks")
st.caption("Your friendly assistant for all things commercial vehicles.")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'last_vehicle_name' not in st.session_state:
    st.session_state.last_vehicle_name = None
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

# Add a Clear Chat button for manual reset
if st.button("🧹 Clear Chat History"):
    st.session_state.chat_history = []
    st.session_state.last_vehicle_name = None
    st.rerun()

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Load Data and Models ---
df_main, df_qna = load_data()
display_faq_sidebar(df_qna)
vector_store = get_vector_store(df_main, df_qna)
qa_chain = get_qa_chain(vector_store)
display_names = set(df_main['Vehicle Name'].dropna().unique())
brand_names = set(df_main['Brand Name'].dropna().str.lower().unique())

# --- Main Query Handling ---
def normalize_vehicle_name(name):
    # Remove spaces, dashes, special characters, and lowercase
    if not isinstance(name, str):
        return ""
    name = unicodedata.normalize('NFKD', name)
    name = name.lower()
    name = re.sub(r'[^a-z0-9]', '', name)
    return name

def get_vehicle_context(data):
    # Use description if available, else build context from specs
    description = data.get("Description", "")
    if description and description.lower() not in ["not available", "nan", "n/a", ""]:
        return description
    # Build context from available specs
    specs = []
    for key in ["Price", "Power", "Fuel Type", "Average Rating", "Variant Name"]:
        val = data.get(key)
        if val and str(val).strip().lower() not in ["not available", "nan", "n/a", ""]:
            specs.append(f"{key}: {val}")
    return ". ".join(specs) if specs else None

def extract_numeric_price(price_str):
    numbers = [float(x.replace(',', '')) for x in re.findall(r'\d+\.?\d*', str(price_str))]
    if not numbers:
        return None
    return min(numbers)

def get_recommendations(main_vehicle, df, n=3, price=None):
    name = main_vehicle.get("Vehicle Name")
    if price is not None:
        base_price = price
    else:
        price = main_vehicle.get("Price", "")
        base_price = extract_numeric_price(price)
    price_numbers = [float(x.replace(',', '')) for x in re.findall(r'\d+\.?\d*', str(main_vehicle.get("Price", "")))]
    if not price_numbers:
        return df.iloc[0:0]
    min_price = min(price_numbers)
    max_price = max(price_numbers)
    # --- CATEGORY FILTERING ---
    category_col = 'Category Name'
    main_category = main_vehicle.get(category_col)
    df_valid = df[df['Price'].apply(lambda x: extract_numeric_price(x) is not None)]
    df_valid = df_valid[df_valid['Vehicle Name'] != name]
    if main_category:
        df_valid = df_valid[df_valid[category_col].str.lower() == str(main_category).lower()]
    df_valid = df_valid.copy()
    df_valid['price_num'] = df_valid['Price'].apply(extract_numeric_price)
    recs = df_valid[(df_valid['price_num'] >= min_price) & (df_valid['price_num'] <= max_price)]
    recs = recs.drop_duplicates(subset=['Vehicle Name'])
    return recs.head(n)

def process_query(prompt: str):
    response = ""
    prompt_lower = prompt.strip().lower()

    # 0. Handle greetings first (before any substring/brand/model logic)
    if is_greeting(prompt):
        goodbye_words = {"bye", "goodbye", "see you", "farewell", "take care"}
        prompt_clean = prompt.strip().lower()
        if any(word in prompt_clean for word in goodbye_words):
            goodbye_responses = [
                "Sad to see you go! If you need vehicle info again, I'll be here.",
                "Goodbye! Have a wonderful day ahead.",
                "Take care! Come back anytime for more vehicle advice.",
                "Farewell! Wishing you safe and happy journeys.",
                "See you soon! Hope I was helpful."
            ]
            bye_msg = random.choice(goodbye_responses)
            with st.chat_message("assistant"):
                st.markdown(bye_msg)
            st.session_state.chat_history.append({"role": "assistant", "content": bye_msg})
            return
        greeting_responses = [
            "Hey there! How can I assist with your vehicle search today?",
            "Hi! Looking for something specific in trucks or vans?",
            "Good day! Got a commercial vehicle in mind?",
            "Hey! I'm here to help you explore the best vehicles.",
            "Welcome! Tell me what kind of vehicle info you're after."
        ]
        greet_msg = random.choice(greeting_responses)
        with st.chat_message("assistant"):
            st.markdown(greet_msg)
        st.session_state.chat_history.append({"role": "assistant", "content": greet_msg})
        return

    # --- INTELLIGENT SPEC QUERY HANDLER (NEW) ---
    import difflib
    spec_patterns = [
        (r'price of (.+)', 'Price', "The price of {model} is {value}.", ['price']),
        (r'fuel type of (.+)', 'Fuel Type', "The fuel type of {model} is {value}.", ['fuel type', 'fuel']),
        (r'seating capacity of (.+)', 'Seating Capacity', "The seating capacity of {model} is {value}.", ['seating capacity', 'seating']),
        (r'power of (.+)', 'Power', "The power of {model} is {value}.", ['power']),
        (r'payload of (.+)', 'Payload', "The payload of {model} is {value}.", ['payload']),
        (r'engine of (.+)', 'Engine', "The engine of {model} is {value}.", ['engine']),
        (r'torque of (.+)', 'Torque', "The torque of {model} is {value}.", ['torque']),
        (r'gvw of (.+)', 'Gross Vehicle Weight', "The gross vehicle weight (GVW) of {model} is {value}.", ['gvw', 'gross vehicle weight']),
        (r'weight of (.+)', 'Gross Vehicle Weight', "The weight of {model} is {value}.", ['weight', 'gross vehicle weight']),
        (r'transmission of (.+)', 'Transmission', "The transmission of {model} is {value}.", ['transmission']),
        (r'wheelbase of (.+)', 'Wheelbase', "The wheelbase of {model} is {value}.", ['wheelbase']),
        (r'length of (.+)', 'Length', "The length of {model} is {value}.", ['length']),
        (r'width of (.+)', 'Width', "The width of {model} is {value}.", ['width']),
        (r'height of (.+)', 'Height', "The height of {model} is {value}.", ['height']),
        (r'tyre of (.+)', 'Tyre', "The tyre specification of {model} is {value}.", ['tyre']),
        (r'brake of (.+)', 'Brake', "The brake specification of {model} is {value}.", ['brake']),
        (r'suspension of (.+)', 'Suspension', "The suspension of {model} is {value}.", ['suspension']),
        (r'emission of (.+)', 'Emission Norms', "The emission norms of {model} is {value}.", ['emission', 'emission norms'])
    ]
    for pattern, field, template, alt_keywords in spec_patterns:
        match = re.search(pattern, prompt_lower)
        if match:
            model_query = match.group(1).strip()
            # Smart matching: try exact, substring, and fuzzy
            best_match = None
            best_score = 0
            for name in display_names:
                # Remove extra spaces, lower, ignore order
                norm_model_query = ' '.join(sorted(model_query.lower().split()))
                norm_name = ' '.join(sorted(name.lower().split()))
                score = difflib.SequenceMatcher(None, norm_model_query, norm_name).ratio()
                if score > best_score:
                    best_score = score
                    best_match = name
                # Also allow substring match
                if norm_model_query in norm_name:
                    best_match = name
                    best_score = 1.0
                    break
            if best_match and best_score > 0.7:
                vehicle_data = get_vehicle_data(best_match, df_main)
                value = vehicle_data.get(field, "Not available") if vehicle_data else "Not available"
                # Special handling for price: show 'Coming Soon' if not available
                if field.lower() == "price" and (not value or str(value).strip().lower() in ["not available", "n/a", "nan", ""]):
                    value = "Coming Soon"
                answer = template.format(model=best_match, value=value)
                with st.chat_message("assistant"):
                    st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                return
            else:
                answer = f"Sorry, I couldn't find the {field.lower()} for {model_query.title()}."
                with st.chat_message("assistant"):
                    st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                return

    # --- 1. Comparison logic (should come BEFORE the intent handler) ---
    if "vs" in prompt_lower or "compare" in prompt_lower:
        data, error = get_vehicle_comparison_data(prompt, df_main, display_names)
        if error:
            response = error
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        else:
            v1_data, v2_data = data[0], data[1]
            # Ensure we have data before proceeding
            if not v1_data or not v2_data:
                response = "I couldn't find the data for one or both vehicles. Please try again."
                with st.chat_message("assistant"):
                    st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                return
            with st.spinner("Analyzing and generating a conclusion..."):
                comparison_prompt = f"""
                Please provide a very brief conclusion (under 50 words) comparing the {v1_data['Vehicle Name']} and the {v2_data['Vehicle Name']}.
                - {v1_data['Vehicle Name']} Pros: {v1_data.get('Pros', 'N/A')}
                - {v2_data['Vehicle Name']} Pros: {v2_data.get('Pros', 'N/A')}
                Focus on the main differences to help a buyer choose.
                """
                conclusion_result = qa_chain.invoke({"question": comparison_prompt})
                conclusion = conclusion_result.get("result", "Could not generate a conclusion.")
            display_comparison(v1_data, v2_data, conclusion)
        return

    # --- INTENT HANDLER: Vehicle name in full sentence vs direct search ---
    found_vehicle = None
    for name in display_names:
        if name.lower() in prompt_lower:
            found_vehicle = name
            break
    if found_vehicle:
        # If query is exactly the vehicle name (or brand), show card as usual
        if prompt_lower.strip() == found_vehicle.lower().strip():
            data = get_vehicle_data(found_vehicle, df_main)
            if data:
                display_vehicle_card(data)
                # Show recommendations below the main card (filter by price and category)
                category_slug = None
                for col in ['Category Name', 'category_name', 'Vehicle Type', 'Category']:
                    if col in df_main.columns and col in data and data[col]:
                        category_slug = str(data[col]).lower()
                        break
                recs = get_recommendations(data, df_main)
                if category_slug:
                    recs = recs[[str(row.get(col, '')).lower() == category_slug for _, row in recs.iterrows()]]
                if not recs.empty:
                    st.subheader("Recommended Alternatives:")
                    rec_datas = [get_vehicle_data(rec_row['Vehicle Name'], df_main) for _, rec_row in recs.iterrows()]
                    cols = st.columns(len(rec_datas))
                    for i, rec_data in enumerate(rec_datas):
                        if rec_data:
                            with cols[i]:
                                display_small_vehicle_card(rec_data)
            else:
                # Vehicle not found: politely deny and suggest alternatives
                # Try to extract brand from the query
                brand = None
                for b in brand_names:
                    if b.lower() in prompt_lower:
                        brand = b
                        break
                st.warning(f"Sorry, I don't have information about '{found_vehicle}' at the moment.")
                if brand:
                    # Try to estimate price from query (if any number present)
                    price_numbers = [float(x.replace(',', '')) for x in re.findall(r'\d+\.?\d*', prompt)]
                    price_val = price_numbers[0] if price_numbers else None
                    # Try to extract category from query
                    category = extract_category(prompt_lower)
                    # Get alternatives from same brand, similar price, and category
                    brand_df = df_main[df_main['Brand Name'].str.lower() == brand.lower()]
                    if price_val:
                        brand_df = brand_df[brand_df['Price'].apply(lambda x: extract_numeric_price(x) is not None)]
                        brand_df['price_num'] = brand_df['Price'].apply(extract_numeric_price)
                        brand_df = brand_df[(brand_df['price_num'] >= price_val * 0.9) & (brand_df['price_num'] <= price_val * 1.1)]
                    if category:
                        for col in ['Category Name', 'category_name', 'Vehicle Type', 'Category']:
                            if col in brand_df.columns:
                                brand_df = brand_df[brand_df[col].str.lower().str.contains(category, na=False)]
                                break
                    brand_df = brand_df.drop_duplicates(subset=['Vehicle Name'])
                    if not brand_df.empty:
                        st.info(f"Here are some similar options from {brand.title()} in a similar price range:")
                        alt_vehicles = brand_df.head(5)
                        cols = st.columns(len(alt_vehicles))
                        for i, (_, row) in enumerate(alt_vehicles.iterrows()):
                            alt_data = get_vehicle_data(row['Vehicle Name'], df_main)
                            if alt_data:
                                with cols[i]:
                                    display_small_vehicle_card(alt_data)
                        return
                # If no brand or alternatives, just show polite denial
                st.info("Please try searching for another vehicle or brand.")
            return
        # If vehicle name is part of a longer sentence/question, use description/specs + LLM
        else:
            data = get_vehicle_data(found_vehicle, df_main)
            context = get_vehicle_context(data) if data else None
            if context:
                llm_prompt = (
                    f"User question: {prompt}\n"
                    f"Vehicle: {found_vehicle}\n"
                    f"Context: {context}\n"
                    "Answer the user's question using the context and your own knowledge. Be concise and helpful."
                )
                with st.spinner("Thinking..."):
                    result = qa_chain.invoke({"question": llm_prompt})
                    answer = result.get("result", "Sorry, I couldn't generate an answer.")
            else:
                answer = (
                    f"Sorry, I don't have enough information about {found_vehicle} to answer your question."
                )
            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            # Show recommendations below the LLM answer if data exists (filter by price and category)
            if data:
                category_slug = None
                for col in ['Category Name', 'category_name', 'Vehicle Type', 'Category']:
                    if col in df_main.columns and col in data and data[col]:
                        category_slug = str(data[col]).lower()
                        break
                recs = get_recommendations(data, df_main)
                if category_slug:
                    recs = recs[[str(row.get(col, '')).lower() == category_slug for _, row in recs.iterrows()]]
                if not recs.empty:
                    st.subheader("Recommended Alternatives:")
                    rec_datas = [get_vehicle_data(rec_row['Vehicle Name'], df_main) for _, rec_row in recs.iterrows()]
                    cols = st.columns(len(rec_datas))
                    for i, rec_data in enumerate(rec_datas):
                        if rec_data:
                            with cols[i]:
                                display_small_vehicle_card(rec_data)
            return

    # --- Brand match: show only top 5 vehicles for the brand ---
    normalized_prompt = normalize_vehicle_name(prompt)
    normalized_brands = {normalize_vehicle_name(brand): brand for brand in brand_names}
    if normalized_prompt in normalized_brands:
        brand = normalized_brands[normalized_prompt]
        brand_models = get_brand_models(brand, df_main)
        if brand_models:
            st.subheader(f"Top 5 vehicles for {brand.title()}:")
            for vehicle in brand_models:
                if vehicle:
                    display_vehicle_card(vehicle)
                else:
                    st.warning(f"I'm sorry, I don't have details for a vehicle in my current database.")
        else:
            st.warning(f"No vehicles found for brand '{brand.title()}'.")
        return

    # --- Normalized vehicle name matching (substring logic) ---
    normalized_display_names = {normalize_vehicle_name(name): name for name in display_names}
    substring_matches = [orig_name for norm_name, orig_name in normalized_display_names.items() if normalized_prompt in norm_name]
    if substring_matches:
        st.subheader(f"Found {len(substring_matches)} matching vehicle(s):")
        for name in substring_matches:
            data = get_vehicle_data(name, df_main)
            if data:
                display_vehicle_card(data)
                # Show recommendations below each card
                recs = get_recommendations(data, df_main)
                if not recs.empty:
                    st.subheader("Recommended Alternatives:")
                    rec_datas = [get_vehicle_data(rec_row['Vehicle Name'], df_main) for _, rec_row in recs.iterrows()]
                    cols = st.columns(len(rec_datas))
                    for i, rec_data in enumerate(rec_datas):
                        if rec_data:
                            with cols[i]:
                                display_small_vehicle_card(rec_data)
            else:
                st.warning(f"I'm sorry, I don't have details for '{name}' in my current database.")
        return

    # 2. Fuzzy matching for close matches (fallback)
    from difflib import SequenceMatcher
    scores = []
    for norm_name, orig_name in normalized_display_names.items():
        score = SequenceMatcher(None, normalized_prompt, norm_name).ratio()
        scores.append((score, orig_name))
    scores.sort(reverse=True)
    best_score, best_name = scores[0]
    close_matches = [(s, n) for s, n in scores if s > 0.75]
    if best_score > 0.85:
        data = get_vehicle_data(best_name, df_main)
        st.subheader(f"Best match (confidence: {best_score:.2f}):")
        if data:
            display_vehicle_card(data)
        else:
            st.warning(f"I'm sorry, I don't have details for '{best_name}' in my current database.")
        return
    elif close_matches:
        st.write("I found a few possible matches. Please confirm:")
        for score, name in close_matches[:3]:
            data = get_vehicle_data(name, df_main)
            st.markdown(f"**{name}** (confidence: {score:.2f})")
            if data:
                display_vehicle_card(data)
            else:
                st.warning(f"I'm sorry, I don't have details for '{name}' in my current database.")
        return

    # --- 1. Handle 'most expensive'/'cheapest' queries over the entire dataset ---
    # if any(kw in prompt_lower for kw in ["most expensive", "highest price", "costliest"]):
    #     # Always search the full DataFrame for numeric prices
    #     df_prices = df_main[pd.to_numeric(df_main['Price'], errors='coerce').notnull()].copy()
    #     if not df_prices.empty:
    #         max_price = df_prices['Price'].astype(float).max()
    #         rows = df_prices[df_prices['Price'].astype(float) == max_price]
    #         vehicles = [get_vehicle_data(row['Vehicle Name'], df_main) for _, row in rows.iterrows()]
    #         st.subheader("Most Expensive Vehicle(s):")
    #         for vehicle_info in vehicles:
    #             display_vehicle_card(vehicle_info)
    #         return
    #     else:
    #         response = "No valid price data available to determine the most expensive vehicle."
    #         with st.chat_message("assistant"):
    #             st.markdown(response)
    #         st.session_state.chat_history.append({"role": "assistant", "content": response})
    #         return
    # elif any(kw in prompt_lower for kw in ["least expensive", "cheapest", "lowest price"]):
    #     df_prices = df_main[pd.to_numeric(df_main['Price'], errors='coerce').notnull()].copy()
    #     if not df_prices.empty:
    #         min_price = df_prices['Price'].astype(float).min()
    #         rows = df_prices[df_prices['Price'].astype(float) == min_price]
    #         vehicles = [get_vehicle_data(row['Vehicle Name'], df_main) for _, row in rows.iterrows()]
    #         st.subheader("Cheapest Vehicle(s):")
    #         for vehicle_info in vehicles:
    #             display_vehicle_card(vehicle_info)
    #         return
    #     else:
    #         response = "No valid price data available to determine the cheapest vehicle."
    #         with st.chat_message("assistant"):
    #             st.markdown(response)
    #         st.session_state.chat_history.append({"role": "assistant", "content": response})
    #         return

    # --- 4. Always search the full DataFrame for all vehicle name matches ---
    # Find all vehicles whose name is mentioned in the query
    found_vehicles = []
    for name in display_names:
        if name.lower() in prompt_lower:
            data = get_vehicle_data(name, df_main)
            if data:
                found_vehicles.append(data)
    
    if found_vehicles:
        st.subheader(f"Found {len(found_vehicles)} matching vehicle(s) in your query:")
        for vehicle_info in found_vehicles:
            display_vehicle_card(vehicle_info)
        return
    
    # Fallback for partial name suggestions if no direct name was found
    suggestions = get_display_name_suggestion(prompt, display_names, df_main)
    if suggestions:
        display_suggestions(suggestions)
        return

    # --- 5. Brand/Family Query (set brand context, even if not in data) ---
    family_brand = is_family_query(prompt_lower, brand_names)
    category = extract_category(prompt_lower)
    if family_brand and family_brand in brand_names:
        st.session_state.last_brand_context = prompt_lower
        st.session_state.last_vehicle_name = None
        with st.spinner(f"Finding popular models for {family_brand.title()}..."):
            brand_models = get_brand_models(family_brand, df_main, category=category)
            if not brand_models:
                response = f"Sorry, I couldn't find any models for the brand {family_brand.title()} in the {category or 'selected'} category."
            else:
                display_brand_summary(family_brand, brand_models)
                return
        if response:
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            return

    # --- 6. Pronoun/helping verb follow-up (use brand context) ---
    brand_context = st.session_state.get('last_brand_context', None)
    if is_pronoun_followup(prompt_lower) and brand_context:
        with st.spinner(f"Finding popular models for your previous brand query..."):
            brand = is_family_query(brand_context, brand_names)
            cat = extract_category(brand_context)
            if brand and brand in brand_names:
                brand_models = get_brand_models(brand, df_main, category=cat)
                if not brand_models:
                    response = f"Sorry, I couldn't find any models for the brand {brand.title()} in the {cat or 'selected'} category."
                else:
                    display_brand_summary(brand, brand_models)
                    return
            else:
                response = "Sorry, I couldn't determine the previous brand context."
        if response:
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            return

    # --- 7. Brand Name Query (exact match) ---
    if prompt_lower in brand_names:
        st.session_state.last_brand_context = prompt_lower
        st.session_state.last_vehicle_name = None
        with st.spinner(f"Finding popular models for {prompt_lower.title()}..."):
            brand_models = get_brand_models(prompt_lower, df_main)
            if not brand_models:
                response = f"Sorry, I couldn't find any models for the brand {prompt_lower.title()}."
            else:
                display_brand_summary(prompt_lower, brand_models)
                return
        if response:
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            return

    # --- 8. Reasons to buy queries for specific vehicles ---
    if any(keyword in prompt_lower for keyword in ["reasons to buy", "why buy", "should i buy", "pros of", "benefits of"]) and (vehicle_name := find_vehicle_in_text(prompt, display_names)):
        st.session_state.last_vehicle_name = vehicle_name
        st.session_state.last_brand_context = None
        data = get_vehicle_data(vehicle_name, df_main)
        with st.spinner("Analyzing reasons to buy..."):
            reasons_prompt = f"What are the key reasons to buy the {vehicle_name}? Focus on its benefits, pros, and unique selling points. Keep it concise but informative."
            result = qa_chain.invoke({"question": reasons_prompt})
            answer = result.get("result", f"Here are some reasons to consider the {vehicle_name}.")
        display_vehicle_with_llm_response(data, answer)
        return

    # --- 9. Exact QnA Match ---
    qna_answer = find_qna_answer(prompt, df_qna)
    if qna_answer:
        with st.chat_message("assistant"):
            st.markdown(qna_answer)
        st.session_state.chat_history.append({"role": "assistant", "content": qna_answer})
        return

    # 4a. If no direct QnA match, use QnA CSV as context for LLM
    # Only do this for questions about 91trucks/company/brand (not for vehicle specs etc.)
    if any(x in prompt.lower() for x in ["91trucks", "91 trucks", "founder", "company", "contact", "about", "who is", "what is", "help", "support"]):
        # Limit to top 10 QnA pairs for context to avoid overloading the LLM
        qna_context = ""
        for _, row in df_qna.head(10).iterrows():
            qna_context += f"Q: {row['question']}\nA: {row['answer']}\n"
        llm_prompt = (
            f"User question: {prompt}\n"
            f"Here are some Q&A pairs from the 91Trucks dataset:\n{qna_context}\n"
            "Answer the user's question using the Q&A pairs above and your own knowledge. Be concise and helpful."
        )
        with st.spinner("Thinking..."):
            result = qa_chain.invoke({"question": llm_prompt})
            answer = result.get("result", "Sorry, I couldn't generate an answer.")
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        return

    # --- 10. Fallback to LLM for general queries, but always prefer CSV data ---
    commercial_keywords = [
        "truck", "trucks", "van", "vans", "pickup", "pickups", "auto", "autos", "bus", "buses", "vehicle", "vehicles", "commercial", "payload", "gvw", "engine", "mileage", "fuel", "specs", "specifications", "brand", "model", "variant", "power", "capacity", "91trucks"
    ]
    if not any(word in prompt_lower for word in commercial_keywords) and not is_greeting(prompt):
        response = "I'm a commercial vehicle and 91Trucks assistant. I can help with trucks, buses, autos, and related queries. Please ask about vehicles or 91Trucks."
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        return

    with st.spinner("Thinking..."):
        # LLM is only used to enhance, not replace, CSV data
        result = qa_chain.invoke({"question": prompt})
        answer = result.get("result") or "Sorry, I couldn't find an answer to that."
        answer = clean_llm_output(answer)
        response = answer
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.chat_history.append({"role": "assistant", "content": response})

# --- User Input Handling ---
if prompt := st.chat_input("Ask about a vehicle, e.g., 'Tata Ace vs Mahindra Jeeto'"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    process_query(prompt)
    # Only clear user_input after processing
    st.session_state.user_input = ""

if st.session_state.user_input:
    # Handle button clicks from suggestions (should not clear chat history)
    query = st.session_state.user_input
    st.session_state.user_input = "" # Clear only after processing
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    process_query(query)



# ***************************************************** Recommendation Old Code ****************************************************


# import streamlit as st
# import pandas as pd
# from langchain_community.vectorstores import FAISS
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
# from langchain_groq import ChatGroq
# import os
# from dotenv import load_dotenv
# import re
# import difflib 
# import random
# import unicodedata
# from sentence_transformers import SentenceTransformer
# from langchain_huggingface import HuggingFaceEmbeddings

# import math

# # Utility to clean meta/instructional lines from LLM output
# def clean_llm_output(text):
#     # Remove lines like (Note: I'll respond accordingly based on the user's query, following the structured process outlined above.)
#     lines = text.split('\n')
#     filtered = [line for line in lines if not re.search(r'\(note:.*structured process.*\)', line, re.IGNORECASE)]
#     return '\n'.join(filtered).strip()

# # --- Page Config ---
# st.set_page_config(page_title="91Trucks", layout="wide")

# # --- Load Environment Variables ---
# load_dotenv()
# HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN") or st.secrets.get("HUGGINGFACEHUB_API_TOKEN")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

# # --- Helper Functions (Copied and adapted from api.py) ---

# def clean_price(price):
#     if isinstance(price, str):
#         price_lower = price.lower().strip()
#         if 'coming soon' in price_lower:
#             return 'Coming Soon'
#         price = re.sub(r'[₹,]', '', price_lower)
#         if 'lakh' in price:
#             value = float(re.findall(r'(\d+\.?\d*)', price)[0])
#             return value
#         elif 'crore' in price:
#             value = float(re.findall(r'(\d+\.?\d*)', price)[0]) * 100
#             return value
#     return None

# @st.cache_data
# def load_data():
#     df_main = pd.read_csv('data/finalist_data.csv')
#     df_qna = pd.read_csv('data/91Trucks_QnAs.csv')
#     df_main["Vehicle Image"] = df_main["Vehicle Image"].fillna("")
#     df_main["Price"] = df_main["Vehicle Price"].apply(clean_price)
#     return df_main, df_qna

# @st.cache_resource
# def get_vector_store(_df_main, _df_qna):
#     with st.spinner("Initializing knowledge base... this may take a moment."):
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         docs = []
#         for _, row in _df_main.iterrows():
#             price_str = f"{row['Price']} Lakh" if pd.notnull(row['Price']) else "N/A"
#             text = (f"Brand: {row['Brand Name']}, Model: {row['Model Name']}, Vehicle Name: {row['Vehicle Name']}, "
#                     f"Electric: {row['Electric']}, Price: {price_str}, Fuel Type: {row['Fuel Type']}, "
#                     f"Variant: {row['Variant Name']}, Power: {row['Power']}, Image: {row['Vehicle Image']}, "
#                     f"End Point: {row.get('End Point', '')}")
#             docs.append(text)
#         for _, row in _df_qna.iterrows():
#             docs.append(f"Question: {row['question']} Answer: {row['answer']}")
#         documents = text_splitter.create_documents(docs)
#         embeddings = HuggingFaceEmbeddings(
#             model="sentence-transformers/all-MiniLM-L6-v2"
#         )
#         vector_store = FAISS.from_documents(documents, embeddings)
#     return vector_store

# @st.cache_resource
# def get_qa_chain(_vector_store):
# #     prompt_template = """
# #     As a commercial vehicle expert, your goal is to provide detailed information and recommendations about commercial vehicles based on the provided context.

# #     When a user asks a question, follow this structured process:

# #     1.  **Identify the User's Need:** Analyze the user's query to understand if they are asking about a specific vehicle, comparing vehicles, asking about a brand, or have a general question about suitability for a task (e.g., "best truck for construction").

# #     2.  **Gather Key Information:** Systematically extract all relevant details from the provided **Context**. The context is your only source of truth. Key information includes:
# #         *   Vehicle Name and Brand
# #         *   Price
# #         *   Key Specifications (Payload, GVW, Engine Power, Fuel Type, Mileage)
# #         *   Features and description.

# #     3.  **Synthesize and Structure the Analysis:** Present the gathered information in a clear, well-structured format. Use markdown for headings and lists. For example:

# #         *   **For a specific vehicle query:**
# #             ### [Vehicle Name]
# #             *   **Price:** ...
# #             *   **Payload:** ...
# #             *   **GVW:** ...
# #             *   **Fuel Type:** ...
# #             *   **Key Features:** ...
# #             only show those features where there is value present for features having not vavailable , dont shopw those features.

# #         *When a user asks about a particular vehicle brand (e.g. tata , mahindra , force, mahindra , euler, eicher ..etc, respond by showing only the top 5 most relevant or popular models from that brand. Do not display the full list, even if more models are available.

# # Instructions for generation:
# # Display only the top 5 model names (no more).

# # You can rank them based on relevance, popularity, or alphabetical order if popularity is unknown.

# # Keep the tone informative and helpful.

# # Do not mention that there are more models unless the user specifically asks.

# # Keep the reply concise (within 80 words).

# #         *   **For a comparison query ("A vs B"):**
# #             Present a side-by-side comparison table or list.

# #         *   **For a brand query:**
# #             List 3-4 popular models from that brand with their key specifications.

# #         *   **For "reasons to buy" or recommendation queries:**
# #             Always mention the specific vehicle name in your response. Focus on the key benefits, pros, and reasons why this vehicle would be a good choice. Highlight unique features, cost-effectiveness, reliability, and suitability for specific use cases.

# #     4.  **Analyze and Justify (if applicable):** Based *only* on the extracted information, provide a brief analysis.
# #         *   For example, if the user asks for a vehicle for a specific purpose (e.g., "city logistics"), analyze the specs to justify a recommendation: "With its compact size and good mileage, the [Vehicle Name] is well-suited for city logistics."
# #         *   Highlight key strengths or weaknesses based on the data. For instance, "This vehicle offers a high payload for its price segment."

# #     5.  **Provide a Recommendation Summary (if applicable):**
# #         *   If the user's query implies a need for a recommendation, conclude with a clear summary.
# #         *   Clearly state what the vehicle is best used for.
# #         *   Mention any potential trade-offs based on the data (e.g., "It has high power, but the mileage is lower than its competitors.").
# #         *   If you cannot make a recommendation from the context, state that. Do not make up information.

# #     **Important:** When recommending or discussing specific vehicles, always include the exact vehicle name in your response so it can be properly identified and displayed with additional information.

# #     **Context:**
# #     {context}

# #     **Question:**
# #     {question}

# #     **Answer:**
# #     """

#     prompt_template = """
#     As a commercial vehicle expert, your goal is to provide detailed information and recommendations about commercial vehicles based on the provided context.

#     When a user asks a question, follow this structured process:

#     1.  **Identify the User's Need:** Analyze the user's query to understand if they are asking about a specific vehicle, comparing vehicles, asking about a brand, or have a general question about suitability for a task (e.g., "best truck for construction").

#     2.  **Gather Key Information:** Systematically extract all relevant details from the provided **Context**. The context is your only source of truth. Key information includes:
#         *   Vehicle Name and Brand
#         *   Price
#         *   Key Specifications (Payload, GVW, Engine Power, Fuel Type, Mileage)
#         *   Features and description.

#     3.  **Synthesize and Structure the Analysis:** Present the gathered information in a clear, well-structured format. Use markdown for headings and lists. For example:

#         *   **For a specific vehicle query:**
#             ##[Vehicle Name]
#             *   **Price:** ...
#             *   **Payload:** ...
#             *   **GVW:** ...
#             *   **Fuel Type:** ...
#             *   **Key Features:** ...
#             only show those features where there is value present for features having not vavailable , dont shopw those features.

#         *When a user asks about a particular vehicle brand (e.g. tata , mahindra , force, mahindra , euler, eicher ..etc, respond by showing only the top 5 most relevant or popular models from that brand. Do not display the full list, even if more models are available.

# Instructions for generation:
# Display only the top 5 model names (no more).

# You can rank them based on relevance, popularity, or alphabetical order if popularity is unknown.

# Keep the tone informative and helpful.

# Do not mention that there are more models unless the user specifically asks.

# Keep the reply concise (within 80 words).

#         *   **For a comparison query ("A vs B"):**
#             Present a side-by-side comparison table or list.

#         *   **For a brand query:**
#             List 3-4 popular models from that brand with their key specifications.

#         *   **For "reasons to buy" or recommendation queries:**
#             Always mention the specific vehicle name in your response. Focus on the key benefits, pros, and reasons why this vehicle would be a good choice. Highlight unique features, cost-effectiveness, reliability, and suitability for specific use cases.

#     4.  **Analyze and Justify (if applicable):** Based *only* on the extracted information, provide a brief analysis.
#         *   For example, if the user asks for a vehicle for a specific purpose (e.g., "city logistics"), analyze the specs to justify a recommendation: "With its compact size and good mileage, the [Vehicle Name] is well-suited for city logistics."
#         *   Highlight key strengths or weaknesses based on the data. For instance, "This vehicle offers a high payload for its price segment."

#     5.  **Provide a Recommendation Summary (if applicable):**
#         *   If the user's query implies a need for a recommendation, conclude with a clear summary.
#         *   Clearly state what the vehicle is best used for.
#         *   Mention any potential trade-offs based on the data (e.g., "It has high power, but the mileage is lower than its competitors.").
#         *   If you cannot make a recommendation from the context, state that. Do not make up information.

#     **Important:** When recommending or discussing specific vehicles, always include the exact vehicle name in your response so it can be properly identified and displayed with additional information.

#     **Context:**
#     {context}

#     **Question:**
#     {question}

#     **Answer:**
#     """
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama3-70b-8192", temperature=0.2, max_tokens=2048)
#     return RetrievalQA.from_chain_type(llm=llm, retriever=_vector_store.as_retriever(), chain_type_kwargs={"prompt": prompt}, return_source_documents=True, input_key="question")

# def get_brand_models(brand_name, df, category=None):
#     brand_df = df[df['Brand Name'].str.lower() == brand_name.lower()]
#     if category:
#         found_category = False
#         for col in ['Category Name', 'category_name', 'Vehicle Type', 'Category']:
#             if col in brand_df.columns:
#                 # Only filter if the column is not all NaN
#                 if brand_df[col].notna().any():
#                     brand_df = brand_df[brand_df[col].str.lower().str.contains(category, na=False)]
#                     found_category = True
#         if not found_category:
#             st.warning(f"Sorry, I can't filter by category '{category}' because the data is missing category information.")
#             return []
#     if brand_df.empty:
#         return []
#     model_names = brand_df['Vehicle Name'].dropna().unique()
#     if 'Popularity' in brand_df.columns:
#         brand_df = brand_df.sort_values('Popularity', ascending=False)
#         model_names = brand_df['Vehicle Name'].dropna().unique()
#     else:
#         model_names = sorted(model_names)
#     top_model_names = model_names[:5]
#     models_data = []
#     for name in top_model_names:
#         vehicle_info = get_vehicle_data(name, df)
#         if vehicle_info:
#             models_data.append(vehicle_info)
#     return models_data

# def is_greeting(query):
#     GREETINGS = {"hi", "hello", "namaste", "hey", "good morning", "good afternoon", "good evening", "greetings", "good day", "hii", "hiii", "heyy", "hey there", "hello there", "yo", "sup", "hola", "bonjour", "ciao", "salaam", "shalom", "howdy", "hiya", "wassup", "what's up", "goodbye", "bye"}
#     q = query.strip().lower()
#     # Remove punctuation and extra whitespace
#     q = re.sub(r'[^a-zA-Z ]', '', q)
#     q = ' '.join(q.split())
#     # If all words in the query are greetings, treat as greeting
#     words = set(q.split())
#     if words and all(word in GREETINGS for word in words):
#         return True
#     # If the whole query matches a greeting
#     if q in GREETINGS:
#         return True
#     return False

# def extract_display_name(query, names):
#     query_clean = query.strip().lower()
#     for name in names:
#         if query_clean == name.strip().lower(): return name
#     return None

# def find_qna_answer(query, qna_df, threshold=0.85):
#     q = query.strip().lower()
#     best_score = 0
#     best_answer = None
#     for _, row in qna_df.iterrows():
#         question = str(row['question']).strip().lower()
#         score = difflib.SequenceMatcher(None, q, question).ratio()
#         if score > best_score:
#             best_score = score
#             best_answer = row['answer']
#     if best_score >= threshold:
#         return best_answer
#     return None

# def find_vehicle_in_text(text, names, threshold=0.7, exclude=None):
#     text_lower = text.lower().strip()
#     exclude = exclude or set()
#     # 1. Try exact match
#     for name in names:
#         if name.lower().strip() == text_lower and name not in exclude:
#             return name
#     # 2. Try full-string substring match
#     for name in names:
#         if text_lower in name.lower() and name not in exclude:
#             return name
#     # 3. Fuzzy match, but exclude already matched names
#     best_match = None
#     best_score = 0
#     for name in names:
#         if name in exclude:
#             continue
#         score = difflib.SequenceMatcher(None, text_lower, name.lower()).ratio()
#         if score > best_score:
#             best_score = score
#             best_match = name
#     if best_score >= threshold:
#         return best_match
#     return None

# def get_vehicle_data(display_name, df):
#     display_name_clean = display_name.lower().strip()
#     vehicle_data_df = df[df['Vehicle Name'].str.lower().str.strip().str.contains(display_name_clean, na=False)]
#     if not vehicle_data_df.empty:
#         prices = vehicle_data_df['Price'].dropna().tolist()
#         # Separate numeric prices and 'Coming Soon'
#         numeric_prices = [p for p in prices if isinstance(p, (int, float))]
#         coming_soon = any(isinstance(p, str) and 'coming soon' in p.lower() for p in prices)
#         # Show numeric price if available, otherwise Coming Soon, otherwise Not available
#         if numeric_prices:
#             min_price, max_price = min(numeric_prices), max(numeric_prices)
#             price_str = f"₹{min_price} Lakh" if min_price == max_price else f"₹{min_price} - {max_price} Lakh"
#         elif coming_soon:
#             price_str = "Coming Soon"
#         else:
#             price_str = "Not available"
#         fuel_types_str = ', '.join(vehicle_data_df['Fuel Type'].dropna().unique()) or "Not available"
#         power_str = ', '.join(vehicle_data_df['Power'].dropna().unique()) or "Not available"
#         variants_str = ', '.join(vehicle_data_df['Variant Name'].dropna().unique()) or "Not available"

#         pros_list = vehicle_data_df['Pros'].dropna().unique().tolist() if 'Pros' in vehicle_data_df.columns else []
#         cons_list = vehicle_data_df['Cons'].dropna().unique().tolist() if 'Cons' in vehicle_data_df.columns else []
#         pros_str = ' | '.join(str(p) for p in pros_list if str(p) != 'nan')
#         cons_str = ' | '.join(str(c) for c in cons_list if str(c) != 'nan')
        
#         row = vehicle_data_df.iloc[0].to_dict()
#         data = {
#             "Vehicle Name": row['Vehicle Name'],
#             "Brand Name": row['Brand Name'],
#             "Vehicle Image": row['Vehicle Image'],
#             "Description": row.get('Vehicle Description', "Not available"),
#             "Average Rating": row.get('Average Rating', "Not available"),
#             "End Point": row.get('End Point'),
#             "Price": price_str,
#             "Fuel Type": fuel_types_str,
#             "Power": power_str,
#             "Variant Name": variants_str,
#             "Pros": pros_str if pros_str else "Not available",
#             "Cons": cons_str if cons_str else "Not available",
#         }
#         return data
#     return None

# def get_display_name_suggestion(query, names, df):
#     query_clean = query.strip().lower()
#     suggestions = []
#     for name in names:
#         if query_clean in name.strip().lower() and query_clean != name.strip().lower():
#             vehicle_data_df = df[df['Vehicle Name'].str.lower().str.strip().str.contains(name.strip().lower(), na=False)]
#             if not vehicle_data_df.empty:
#                 prices = vehicle_data_df['Price'].dropna().tolist()
#                 numeric_prices = [p for p in prices if isinstance(p, (int, float))]
#                 coming_soon = any(isinstance(p, str) and 'coming soon' in p.lower() for p in prices)
#                 price_str = "Not available"
#                 if coming_soon:
#                     price_str = "Coming Soon"
#                 elif numeric_prices:
#                     min_price, max_price = min(numeric_prices), max(numeric_prices)
#                     price_str = f"₹{min_price} Lakh" if min_price == max_price else f"₹{min_price} - {max_price} Lakh"
#                 fuel_types = vehicle_data_df['Fuel Type'].dropna().unique().tolist()
#                 fuel_types_str = ', '.join(fuel_types) if fuel_types else "Not available"
#                 powers = []
#                 for p in vehicle_data_df['Power'].dropna().unique():
#                     try:
#                         powers.append(float(re.findall(r'\d+\.?\d*', str(p))[0]))
#                     except Exception:
#                         continue
#                 power_str = ', '.join(vehicle_data_df['Power'].dropna().unique()) if powers else "Not available"
#                 ratings = vehicle_data_df['Average Rating'].dropna().tolist()
#                 avg_rating = None
#                 if ratings:
#                     try:
#                         avg_rating = sum([float(r) for r in ratings]) / len(ratings)
#                     except Exception:
#                         avg_rating = None
#                 rating_str = f"{avg_rating:.1f}" if avg_rating and avg_rating >= 3 else None
#                 vehicle_image = vehicle_data_df['Vehicle Image'].iloc[0] if 'Vehicle Image' in vehicle_data_df.columns else None
#                 end_point = vehicle_data_df['End Point'].iloc[0] if 'End Point' in vehicle_data_df.columns else None
#                 suggestion = {
#                     "Model Name": name,
#                     "Price": price_str,
#                     "Fuel Type": fuel_types_str,
#                     "Power": power_str,
#                 }
#                 if rating_str:
#                     suggestion["Average Rating"] = rating_str
#                 if vehicle_image:
#                     suggestion["Vehicle Image"] = vehicle_image
#                 if end_point:
#                     suggestion["End Point"] = end_point
#                 suggestions.append(clean_vehicle_row(suggestion))
#     return suggestions if suggestions else None

# def get_vehicle_comparison_data(query, df, names_set):
#     # Split on 'vs' or 'versus', strip and lower-case both sides
#     parts = re.split(r'\s+vs\s+|\s+versus\s+', query.lower())
#     if len(parts) != 2:
#         return None, "Please provide two vehicles to compare using 'vs', like 'Tata Ace vs Mahindra Jeeto'."
#     v1_name = find_vehicle_in_text(parts[0].strip(), names_set)
#     v2_name = find_vehicle_in_text(parts[1].strip(), names_set, exclude={v1_name} if v1_name else set())
#     if not v1_name or not v2_name:
#         return None, "I couldn't identify one or both of the vehicles. Please check the names and try again."
#     v1_data = get_vehicle_data(v1_name, df)
#     v2_data = get_vehicle_data(v2_name, df)
#     return [v1_data, v2_data], None

# def clean_vehicle_row(row):
#     result = {}
#     for key, value in row.items():
#         val = str(value).strip().lower() if value is not None else ""
#         if val not in ["not available", "n/a", "-", ""] and not (isinstance(value, float) and math.isnan(value)):
#             result[key] = value
#     return result

# # --- Helper Functions (add at the top, after imports) ---
# def is_pronoun_followup(query):
#     pronouns = ["it", "this", "that", "these", "those"]
#     helping_verbs = ["is it", "does it", "can it", "will it", "has it", "have it", "was it", "were it"]
#     q = query.lower()
#     return any(p in q.split() for p in pronouns) or any(hv in q for hv in helping_verbs)

# def is_family_query(query, brand_names):
#     q = query.lower()
#     # If query contains 'truck' or 'trucks' and at least one other word, treat as family/brand query
#     if ("truck" in q or "trucks" in q) and len(q.split()) > 1:
#         return q.split()[0]  # Return the first word as the likely brand
#     for brand in brand_names:
#         if brand in q and ("truck" in q or "trucks" in q or "family" in q or "range" in q):
#             return brand
#     return None

# # --- Helper Functions (add after is_family_query) ---
# def extract_category(query):
#     categories = ["truck", "trucks", "van", "vans", "pickup", "pickups", "auto", "autos", "bus", "buses"]
#     for cat in categories:
#         if cat in query:
#             return cat.rstrip('s')  # normalize to singular
#     return None

# # --- Helper Functions (add after extract_category) ---
# def is_new_intent_query(query):
#     keywords = [
#         "cheapest", "best", "most expensive", "top", "lowest", "highest", "price under", "price below", "budget", "affordable"
#     ]
#     q = query.lower()
#     return any(kw in q for kw in keywords)

# # --- UI Rendering Functions ---

# def display_vehicle_card(vehicle_data):
#     with st.container(border=True):
#         if vehicle_data.get("Vehicle Image"):
#             st.image(vehicle_data["Vehicle Image"], width=200)
#         st.subheader(vehicle_data.get("Vehicle Name", "Unknown Vehicle"))
        
#         details_map = {
#             "Price": vehicle_data.get("Price"),
#             "Power": vehicle_data.get("Power"),
#             "Fuel Type": vehicle_data.get("Fuel Type"),
#             "Average Rating": vehicle_data.get("Average Rating"),
#             "Variant(s)": vehicle_data.get("Variant Name")
#         }
        
#         details = ""
#         for key, value in details_map.items():
#             if value and str(value).strip().lower() not in ["not available", "nan", "n/a", ""]:
#                 details += f"**{key}:** {value}<br>"
#         st.markdown(details, unsafe_allow_html=True)
        
#         description = vehicle_data.get("Description")
#         if description and str(description).strip().lower() not in ["not available", "nan", "n/a", ""]:
#             with st.expander("View Description"):
#                 st.markdown(description)

#         if vehicle_data.get("Pros") and vehicle_data.get("Pros") != "Not available":
#             with st.expander("View Pros"):
#                 for pro in vehicle_data["Pros"].split(' | '):
#                     st.markdown(f"• {pro.strip()}")
                
#         if vehicle_data.get("Cons") and vehicle_data.get("Cons") != "Not available":
#             with st.expander("View Cons"):
#                 for con in vehicle_data["Cons"].split(' | '):
#                     st.markdown(f"• {con.strip()}")

#         if vehicle_data.get("End Point"):
#             st.link_button("Learn More", vehicle_data["End Point"])

# def display_vehicle_with_llm_response(vehicle_data, llm_response):
#     """Display vehicle information with LLM response, small image, and See More button. Show fallback if missing."""
#     with st.container(border=True):
#         col1, col2 = st.columns([1, 3])
#         with col1:
#             if vehicle_data.get("Vehicle Image"):
#                 st.image(vehicle_data["Vehicle Image"], width=150)
#             else:
#                 st.write(":camera: No image available.")
#         with col2:
#             st.subheader(vehicle_data["Vehicle Name"])
#             st.markdown(llm_response)
#             if vehicle_data.get("End Point"):
#                 st.link_button("See More", vehicle_data["End Point"])
#             else:
#                 st.write(":link: No additional link available.")

# def display_brand_summary(brand_name, models):
#     st.subheader(f"Here are some popular models from {brand_name.title()}:")
    
#     for model in models:
#         with st.container(border=True):
#             col1, col2 = st.columns([1, 2])
#             with col1:
#                 if model.get("Vehicle Image"):
#                     st.image(model["Vehicle Image"], use_container_width=True)
#             with col2:
#                 st.markdown(f"**{model['Vehicle Name']}**")
#                 details = ""
#                 if model.get("Price"):
#                     details += f"**Price:** {model['Price']}<br>"
#                 if model.get("Fuel Type"):
#                     details += f"**Fuel Type:** {model['Fuel Type']}<br>"
#                 st.markdown(details, unsafe_allow_html=True)

#                 if model.get("End Point"):
#                     st.link_button("See more", model["End Point"])

# def display_comparison_table(v1_data, v2_data):
#     st.subheader("Side-by-Side Comparison")
    
#     # Define which keys to show in the table and their display order
#     display_keys = [
#         "Price", "Power", "Fuel Type", "Average Rating", "Variant Name"
#     ]
    
#     # Use the full vehicle name for the table headers
#     v1_name = v1_data.get("Vehicle Name", "Vehicle 1")
#     v2_name = v2_data.get("Vehicle Name", "Vehicle 2")

#     table_data = {"Feature": [], v1_name: [], v2_name: []}
    
#     for key in display_keys:
#         v1_value = v1_data.get(key, "Not available")
#         v2_value = v2_data.get(key, "Not available")
        
#         # Only add row if at least one vehicle has a non-empty value for this feature
#         v1_valid = v1_value and str(v1_value).strip().lower() not in ["not available", "nan", "n/a", ""]
#         v2_valid = v2_value and str(v2_value).strip().lower() not in ["not available", "nan", "n/a", ""]

#         if v1_valid or v2_valid:
#             table_data["Feature"].append(key.replace("_", " ").title())
#             table_data[v1_name].append(v1_value if v1_valid else "—")
#             table_data[v2_name].append(v2_value if v2_valid else "—")
            
#     if table_data["Feature"]:
#         df_compare = pd.DataFrame(table_data)
#         st.table(df_compare.set_index("Feature"))
#     else:
#         st.info("No comparable features found between these two vehicles.")

# def display_comparison(v1_data, v2_data, conclusion):
#     st.subheader("Vehicles for Comparison")
#     col1, col2 = st.columns(2)
#     with col1:
#         display_vehicle_card(v1_data)
#     with col2:
#         display_vehicle_card(v2_data)
    
#     st.markdown("---")
    
#     # Display the side-by-side comparison table
#     display_comparison_table(v1_data, v2_data)
    
#     if conclusion:
#         st.markdown("---")
#         st.subheader("✍️ Conclusion")
#         st.markdown(conclusion)

# def display_suggestions(suggestions):
#     st.write("I found a few possible matches. Did you mean one of these?")
#     for i, vehicle in enumerate(suggestions[:3]):
#         with st.container(border=True):
#             col1, col2 = st.columns([1, 3])
#             with col1:
#                 if vehicle.get("Vehicle Image"):
#                     st.image(vehicle["Vehicle Image"], use_container_width=True)
#             with col2:
#                 st.subheader(vehicle.get('Vehicle Name') or vehicle.get('Model Name', 'Unknown Model'))
#                 details = ""
#                 if vehicle.get("Price"):
#                     details += f"**Price:** {vehicle['Price']}<br>"
#                 if vehicle.get("Fuel Type"):
#                     details += f"**Fuel Type:** {vehicle['Fuel Type']}<br>"
#                 st.markdown(details, unsafe_allow_html=True)
#                 if vehicle.get("End Point"):
#                     st.link_button("Show Details", vehicle["End Point"])

# def display_small_vehicle_card(vehicle_data):
#     with st.container(border=True):
#         cols = st.columns([1, 2])
#         with cols[0]:
#             if vehicle_data.get("Vehicle Image"):
#                 st.image(vehicle_data["Vehicle Image"], width=80)
#         with cols[1]:
#             st.markdown(f"**{vehicle_data.get('Vehicle Name', 'Unknown Vehicle')}**")
#             price = vehicle_data.get("Price")
#             if price:
#                 st.markdown(f"<span style='font-size:0.9em;'>Price: {price}</span>", unsafe_allow_html=True)
#             fuel = vehicle_data.get("Fuel Type")
#             if fuel:
#                 st.markdown(f"<span style='font-size:0.9em;'>Fuel: {fuel}</span>", unsafe_allow_html=True)
#             if vehicle_data.get("End Point"):
#                 st.link_button("Details", vehicle_data["End Point"])

# # --- Sidebar: Display Most Frequently Asked Questions ---
# def display_faq_sidebar(df_qna=None):
#     st.sidebar.markdown("""
#     <div style='display: flex; align-items: center; gap: 10px;'>
#         <img src='https://cdn-icons-png.flaticon.com/512/4712/4712035.png' width='36' style='border-radius: 50%;'>
#         <span style='font-size: 1.3em; font-weight: bold;'>Quick Questions</span>
#     </div>
#     <div style='margin-top: 10px; margin-bottom: 18px; color: #555;'>
#         <em>Your friendly commercial vehicle assistant!</em>
#     </div>
#     """, unsafe_allow_html=True)
#     questions = [
#         "What is 91Trucks?",
#         "who is the founder of 91trucks",
#         "how can i contact to the 91trucks",
#         "How 91trucks can help me to get good vehicle",
#         "compare tata intra v30 vs tata ace gold 2.0",
#         "compare mahindra zeo vs mahindra jeeto",
#         "tata trucks",
#         "eicher trucks",
#         "What is the primary focus of 91Trucks' digital platform?",
#         "What is the significance of 91Trucks' physical retail stores?",
#         "What is the role of 91Trucks in the broader logistics and transportation sectors?",
#         "why choose 91trucks rather than other.",
#         "what is the price of tata intra v30",
#         "show me some tata buses, trucks, auto-rickshaw",
#         "Most Expensive vehicle",
#     ]
#     for q in questions:
#         if st.sidebar.button(q, key=f"faq_{q}"):
#             st.session_state.user_input = q
#             st.rerun()

# # --- Streamlit UI and Main Logic ---

# st.title("🚚 91Trucks")
# st.caption("Your friendly assistant for all things commercial vehicles.")

# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []
# if 'last_vehicle_name' not in st.session_state:
#     st.session_state.last_vehicle_name = None
# if 'user_input' not in st.session_state:
#     st.session_state.user_input = ""

# # Add a Clear Chat button for manual reset
# if st.button("🧹 Clear Chat History"):
#     st.session_state.chat_history = []
#     st.session_state.last_vehicle_name = None
#     st.rerun()

# for message in st.session_state.chat_history:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # --- Load Data and Models ---
# df_main, df_qna = load_data()
# display_faq_sidebar(df_qna)
# vector_store = get_vector_store(df_main, df_qna)
# qa_chain = get_qa_chain(vector_store)
# display_names = set(df_main['Vehicle Name'].dropna().unique())
# brand_names = set(df_main['Brand Name'].dropna().str.lower().unique())

# # --- Main Query Handling ---
# def normalize_vehicle_name(name):
#     # Remove spaces, dashes, special characters, and lowercase
#     if not isinstance(name, str):
#         return ""
#     name = unicodedata.normalize('NFKD', name)
#     name = name.lower()
#     name = re.sub(r'[^a-z0-9]', '', name)
#     return name

# def get_vehicle_context(data):
#     # Use description if available, else build context from specs
#     description = data.get("Description", "")
#     if description and description.lower() not in ["not available", "nan", "n/a", ""]:
#         return description
#     # Build context from available specs
#     specs = []
#     for key in ["Price", "Power", "Fuel Type", "Average Rating", "Variant Name"]:
#         val = data.get(key)
#         if val and str(val).strip().lower() not in ["not available", "nan", "n/a", ""]:
#             specs.append(f"{key}: {val}")
#     return ". ".join(specs) if specs else None

# def extract_numeric_price(price_str):
#     numbers = [float(x.replace(',', '')) for x in re.findall(r'\d+\.?\d*', str(price_str))]
#     if not numbers:
#         return None
#     return min(numbers)

# def get_recommendations(main_vehicle, df, n=3, price=None):
#     name = main_vehicle.get("Vehicle Name")
#     if price is not None:
#         base_price = price
#     else:
#         price = main_vehicle.get("Price", "")
#         base_price = extract_numeric_price(price)
#     price_numbers = [float(x.replace(',', '')) for x in re.findall(r'\d+\.?\d*', str(main_vehicle.get("Price", "")))]
#     if not price_numbers:
#         return df.iloc[0:0]
#     min_price = min(price_numbers)
#     max_price = max(price_numbers)
#     df_valid = df[df['Price'].apply(lambda x: extract_numeric_price(x) is not None)]
#     df_valid = df_valid[df_valid['Vehicle Name'] != name]
#     df_valid = df_valid.copy()
#     df_valid['price_num'] = df_valid['Price'].apply(extract_numeric_price)
#     recs = df_valid[(df_valid['price_num'] >= min_price) & (df_valid['price_num'] <= max_price)]
#     recs = recs.drop_duplicates(subset=['Vehicle Name'])
#     return recs.head(n)

# def process_query(prompt: str):
#     response = ""
#     prompt_lower = prompt.strip().lower()

#     # 0. Handle greetings first (before any substring/brand/model logic)
#     if is_greeting(prompt):
#         goodbye_words = {"bye", "goodbye", "see you", "farewell", "take care"}
#         prompt_clean = prompt.strip().lower()
#         if any(word in prompt_clean for word in goodbye_words):
#             goodbye_responses = [
#                 "Sad to see you go! If you need vehicle info again, I'll be here.",
#                 "Goodbye! Have a wonderful day ahead.",
#                 "Take care! Come back anytime for more vehicle advice.",
#                 "Farewell! Wishing you safe and happy journeys.",
#                 "See you soon! Hope I was helpful."
#             ]
#             bye_msg = random.choice(goodbye_responses)
#             with st.chat_message("assistant"):
#                 st.markdown(bye_msg)
#             st.session_state.chat_history.append({"role": "assistant", "content": bye_msg})
#             return
#         greeting_responses = [
#             "Hey there! How can I assist with your vehicle search today?",
#             "Hi! Looking for something specific in trucks or vans?",
#             "Good day! Got a commercial vehicle in mind?",
#             "Hey! I'm here to help you explore the best vehicles.",
#             "Welcome! Tell me what kind of vehicle info you're after."
#         ]
#         greet_msg = random.choice(greeting_responses)
#         with st.chat_message("assistant"):
#             st.markdown(greet_msg)
#         st.session_state.chat_history.append({"role": "assistant", "content": greet_msg})
#         return

#     # --- INTELLIGENT SPEC QUERY HANDLER (NEW) ---
#     import difflib
#     spec_patterns = [
#         (r'price of (.+)', 'Price', "The price of {model} is {value}.", ['price']),
#         (r'fuel type of (.+)', 'Fuel Type', "The fuel type of {model} is {value}.", ['fuel type', 'fuel']),
#         (r'seating capacity of (.+)', 'Seating Capacity', "The seating capacity of {model} is {value}.", ['seating capacity', 'seating']),
#         (r'power of (.+)', 'Power', "The power of {model} is {value}.", ['power']),
#         (r'payload of (.+)', 'Payload', "The payload of {model} is {value}.", ['payload']),
#         (r'engine of (.+)', 'Engine', "The engine of {model} is {value}.", ['engine']),
#         (r'torque of (.+)', 'Torque', "The torque of {model} is {value}.", ['torque']),
#         (r'gvw of (.+)', 'Gross Vehicle Weight', "The gross vehicle weight (GVW) of {model} is {value}.", ['gvw', 'gross vehicle weight']),
#         (r'weight of (.+)', 'Gross Vehicle Weight', "The weight of {model} is {value}.", ['weight', 'gross vehicle weight']),
#         (r'transmission of (.+)', 'Transmission', "The transmission of {model} is {value}.", ['transmission']),
#         (r'wheelbase of (.+)', 'Wheelbase', "The wheelbase of {model} is {value}.", ['wheelbase']),
#         (r'length of (.+)', 'Length', "The length of {model} is {value}.", ['length']),
#         (r'width of (.+)', 'Width', "The width of {model} is {value}.", ['width']),
#         (r'height of (.+)', 'Height', "The height of {model} is {value}.", ['height']),
#         (r'tyre of (.+)', 'Tyre', "The tyre specification of {model} is {value}.", ['tyre']),
#         (r'brake of (.+)', 'Brake', "The brake specification of {model} is {value}.", ['brake']),
#         (r'suspension of (.+)', 'Suspension', "The suspension of {model} is {value}.", ['suspension']),
#         (r'emission of (.+)', 'Emission Norms', "The emission norms of {model} is {value}.", ['emission', 'emission norms'])
#     ]
#     for pattern, field, template, alt_keywords in spec_patterns:
#         match = re.search(pattern, prompt_lower)
#         if match:
#             model_query = match.group(1).strip()
#             # Smart matching: try exact, substring, and fuzzy
#             best_match = None
#             best_score = 0
#             for name in display_names:
#                 # Remove extra spaces, lower, ignore order
#                 norm_model_query = ' '.join(sorted(model_query.lower().split()))
#                 norm_name = ' '.join(sorted(name.lower().split()))
#                 score = difflib.SequenceMatcher(None, norm_model_query, norm_name).ratio()
#                 if score > best_score:
#                     best_score = score
#                     best_match = name
#                 # Also allow substring match
#                 if norm_model_query in norm_name:
#                     best_match = name
#                     best_score = 1.0
#                     break
#             if best_match and best_score > 0.7:
#                 vehicle_data = get_vehicle_data(best_match, df_main)
#                 value = vehicle_data.get(field, "Not available") if vehicle_data else "Not available"
#                 # Special handling for price: show 'Coming Soon' if not available
#                 if field.lower() == "price" and (not value or str(value).strip().lower() in ["not available", "n/a", "nan", ""]):
#                     value = "Coming Soon"
#                 answer = template.format(model=best_match, value=value)
#                 with st.chat_message("assistant"):
#                     st.markdown(answer)
#                 st.session_state.chat_history.append({"role": "assistant", "content": answer})
#                 return
#             else:
#                 answer = f"Sorry, I couldn't find the {field.lower()} for {model_query.title()}."
#                 with st.chat_message("assistant"):
#                     st.markdown(answer)
#                 st.session_state.chat_history.append({"role": "assistant", "content": answer})
#                 return

#     # --- 1. Comparison logic (should come BEFORE the intent handler) ---
#     if "vs" in prompt_lower or "compare" in prompt_lower:
#         data, error = get_vehicle_comparison_data(prompt, df_main, display_names)
#         if error:
#             response = error
#             with st.chat_message("assistant"):
#                 st.markdown(response)
#             st.session_state.chat_history.append({"role": "assistant", "content": response})
#         else:
#             v1_data, v2_data = data[0], data[1]
#             # Ensure we have data before proceeding
#             if not v1_data or not v2_data:
#                 response = "I couldn't find the data for one or both vehicles. Please try again."
#                 with st.chat_message("assistant"):
#                     st.markdown(response)
#                 st.session_state.chat_history.append({"role": "assistant", "content": response})
#                 return
#             with st.spinner("Analyzing and generating a conclusion..."):
#                 comparison_prompt = f"""
#                 Please provide a very brief conclusion (under 50 words) comparing the {v1_data['Vehicle Name']} and the {v2_data['Vehicle Name']}.
#                 - {v1_data['Vehicle Name']} Pros: {v1_data.get('Pros', 'N/A')}
#                 - {v2_data['Vehicle Name']} Pros: {v2_data.get('Pros', 'N/A')}
#                 Focus on the main differences to help a buyer choose.
#                 """
#                 conclusion_result = qa_chain.invoke({"question": comparison_prompt})
#                 conclusion = conclusion_result.get("result", "Could not generate a conclusion.")
#             display_comparison(v1_data, v2_data, conclusion)
#         return

#     # --- INTENT HANDLER: Vehicle name in full sentence vs direct search ---
#     found_vehicle = None
#     for name in display_names:
#         if name.lower() in prompt_lower:
#             found_vehicle = name
#             break
#     if found_vehicle:
#         # If query is exactly the vehicle name (or brand), show card as usual
#         if prompt_lower.strip() == found_vehicle.lower().strip():
#             data = get_vehicle_data(found_vehicle, df_main)
#             if data:
#                 display_vehicle_card(data)
#                 # Show recommendations below the main card
#                 recs = get_recommendations(data, df_main)
#                 if not recs.empty:
#                     st.subheader("Recommended Alternatives:")
#                     rec_datas = [get_vehicle_data(rec_row['Vehicle Name'], df_main) for _, rec_row in recs.iterrows()]
#                     cols = st.columns(len(rec_datas))
#                     for i, rec_data in enumerate(rec_datas):
#                         if rec_data:
#                             with cols[i]:
#                                 display_small_vehicle_card(rec_data)
#             else:
#                 st.warning(f"I'm sorry, I don't have details for '{found_vehicle}' in my current database.")
#             return
#         # If vehicle name is part of a longer sentence/question, use description/specs + LLM
#         else:
#             data = get_vehicle_data(found_vehicle, df_main)
#             context = get_vehicle_context(data) if data else None
#             if context:
#                 llm_prompt = (
#                     f"User question: {prompt}\n"
#                     f"Vehicle: {found_vehicle}\n"
#                     f"Context: {context}\n"
#                     "Answer the user's question using the context and your own knowledge. Be concise and helpful."
#                 )
#                 with st.spinner("Thinking..."):
#                     result = qa_chain.invoke({"question": llm_prompt})
#                     answer = result.get("result", "Sorry, I couldn't generate an answer.")
#             else:
#                 answer = (
#                     f"Sorry, I don't have enough information about {found_vehicle} to answer your question."
#                 )
#             with st.chat_message("assistant"):
#                 st.markdown(answer)
#             st.session_state.chat_history.append({"role": "assistant", "content": answer})
#             # Show recommendations below the LLM answer if data exists
#             if data:
#                 recs = get_recommendations(data, df_main)
#                 if not recs.empty:
#                     st.subheader("Recommended Alternatives:")
#                     rec_datas = [get_vehicle_data(rec_row['Vehicle Name'], df_main) for _, rec_row in recs.iterrows()]
#                     cols = st.columns(len(rec_datas))
#                     for i, rec_data in enumerate(rec_datas):
#                         if rec_data:
#                             with cols[i]:
#                                 display_small_vehicle_card(rec_data)
#             return

#     # --- Brand match: show only top 5 vehicles for the brand ---
#     normalized_prompt = normalize_vehicle_name(prompt)
#     normalized_brands = {normalize_vehicle_name(brand): brand for brand in brand_names}
#     if normalized_prompt in normalized_brands:
#         brand = normalized_brands[normalized_prompt]
#         brand_models = get_brand_models(brand, df_main)
#         if brand_models:
#             st.subheader(f"Top 5 vehicles for {brand.title()}:")
#             for vehicle in brand_models:
#                 if vehicle:
#                     display_vehicle_card(vehicle)
#                 else:
#                     st.warning(f"I'm sorry, I don't have details for a vehicle in my current database.")
#         else:
#             st.warning(f"No vehicles found for brand '{brand.title()}'.")
#         return

#     # --- Normalized vehicle name matching (substring logic) ---
#     normalized_display_names = {normalize_vehicle_name(name): name for name in display_names}
#     substring_matches = [orig_name for norm_name, orig_name in normalized_display_names.items() if normalized_prompt in norm_name]
#     if substring_matches:
#         st.subheader(f"Found {len(substring_matches)} matching vehicle(s):")
#         for name in substring_matches:
#             data = get_vehicle_data(name, df_main)
#             if data:
#                 display_vehicle_card(data)
#                 # Show recommendations below each card
#                 recs = get_recommendations(data, df_main)
#                 if not recs.empty:
#                     st.subheader("Recommended Alternatives:")
#                     rec_datas = [get_vehicle_data(rec_row['Vehicle Name'], df_main) for _, rec_row in recs.iterrows()]
#                     cols = st.columns(len(rec_datas))
#                     for i, rec_data in enumerate(rec_datas):
#                         if rec_data:
#                             with cols[i]:
#                                 display_small_vehicle_card(rec_data)
#             else:
#                 st.warning(f"I'm sorry, I don't have details for '{name}' in my current database.")
#         return

#     # 2. Fuzzy matching for close matches (fallback)
#     from difflib import SequenceMatcher
#     scores = []
#     for norm_name, orig_name in normalized_display_names.items():
#         score = SequenceMatcher(None, normalized_prompt, norm_name).ratio()
#         scores.append((score, orig_name))
#     scores.sort(reverse=True)
#     best_score, best_name = scores[0]
#     close_matches = [(s, n) for s, n in scores if s > 0.75]
#     if best_score > 0.85:
#         data = get_vehicle_data(best_name, df_main)
#         st.subheader(f"Best match (confidence: {best_score:.2f}):")
#         if data:
#             display_vehicle_card(data)
#         else:
#             st.warning(f"I'm sorry, I don't have details for '{best_name}' in my current database.")
#         return
#     elif close_matches:
#         st.write("I found a few possible matches. Please confirm:")
#         for score, name in close_matches[:3]:
#             data = get_vehicle_data(name, df_main)
#             st.markdown(f"**{name}** (confidence: {score:.2f})")
#             if data:
#                 display_vehicle_card(data)
#             else:
#                 st.warning(f"I'm sorry, I don't have details for '{name}' in my current database.")
#         return

#     # --- 1. Handle 'most expensive'/'cheapest' queries over the entire dataset ---
#     # if any(kw in prompt_lower for kw in ["most expensive", "highest price", "costliest"]):
#     #     # Always search the full DataFrame for numeric prices
#     #     df_prices = df_main[pd.to_numeric(df_main['Price'], errors='coerce').notnull()].copy()
#     #     if not df_prices.empty:
#     #         max_price = df_prices['Price'].astype(float).max()
#     #         rows = df_prices[df_prices['Price'].astype(float) == max_price]
#     #         vehicles = [get_vehicle_data(row['Vehicle Name'], df_main) for _, row in rows.iterrows()]
#     #         st.subheader("Most Expensive Vehicle(s):")
#     #         for vehicle_info in vehicles:
#     #             display_vehicle_card(vehicle_info)
#     #         return
#     #     else:
#     #         response = "No valid price data available to determine the most expensive vehicle."
#     #         with st.chat_message("assistant"):
#     #             st.markdown(response)
#     #         st.session_state.chat_history.append({"role": "assistant", "content": response})
#     #         return
#     # elif any(kw in prompt_lower for kw in ["least expensive", "cheapest", "lowest price"]):
#     #     df_prices = df_main[pd.to_numeric(df_main['Price'], errors='coerce').notnull()].copy()
#     #     if not df_prices.empty:
#     #         min_price = df_prices['Price'].astype(float).min()
#     #         rows = df_prices[df_prices['Price'].astype(float) == min_price]
#     #         vehicles = [get_vehicle_data(row['Vehicle Name'], df_main) for _, row in rows.iterrows()]
#     #         st.subheader("Cheapest Vehicle(s):")
#     #         for vehicle_info in vehicles:
#     #             display_vehicle_card(vehicle_info)
#     #         return
#     #     else:
#     #         response = "No valid price data available to determine the cheapest vehicle."
#     #         with st.chat_message("assistant"):
#     #             st.markdown(response)
#     #         st.session_state.chat_history.append({"role": "assistant", "content": response})
#     #         return

#     # --- 4. Always search the full DataFrame for all vehicle name matches ---
#     # Find all vehicles whose name is mentioned in the query
#     found_vehicles = []
#     for name in display_names:
#         if name.lower() in prompt_lower:
#             data = get_vehicle_data(name, df_main)
#             if data:
#                 found_vehicles.append(data)
    
#     if found_vehicles:
#         st.subheader(f"Found {len(found_vehicles)} matching vehicle(s) in your query:")
#         for vehicle_info in found_vehicles:
#             display_vehicle_card(vehicle_info)
#         return
    
#     # Fallback for partial name suggestions if no direct name was found
#     suggestions = get_display_name_suggestion(prompt, display_names, df_main)
#     if suggestions:
#         display_suggestions(suggestions)
#         return

#     # --- 5. Brand/Family Query (set brand context, even if not in data) ---
#     family_brand = is_family_query(prompt_lower, brand_names)
#     category = extract_category(prompt_lower)
#     if family_brand and family_brand in brand_names:
#         st.session_state.last_brand_context = prompt_lower
#         st.session_state.last_vehicle_name = None
#         with st.spinner(f"Finding popular models for {family_brand.title()}..."):
#             brand_models = get_brand_models(family_brand, df_main, category=category)
#             if not brand_models:
#                 response = f"Sorry, I couldn't find any models for the brand {family_brand.title()} in the {category or 'selected'} category."
#             else:
#                 display_brand_summary(family_brand, brand_models)
#                 return
#         if response:
#             with st.chat_message("assistant"):
#                 st.markdown(response)
#             st.session_state.chat_history.append({"role": "assistant", "content": response})
#             return

#     # --- 6. Pronoun/helping verb follow-up (use brand context) ---
#     brand_context = st.session_state.get('last_brand_context', None)
#     if is_pronoun_followup(prompt_lower) and brand_context:
#         with st.spinner(f"Finding popular models for your previous brand query..."):
#             brand = is_family_query(brand_context, brand_names)
#             cat = extract_category(brand_context)
#             if brand and brand in brand_names:
#                 brand_models = get_brand_models(brand, df_main, category=cat)
#                 if not brand_models:
#                     response = f"Sorry, I couldn't find any models for the brand {brand.title()} in the {cat or 'selected'} category."
#                 else:
#                     display_brand_summary(brand, brand_models)
#                     return
#             else:
#                 response = "Sorry, I couldn't determine the previous brand context."
#         if response:
#             with st.chat_message("assistant"):
#                 st.markdown(response)
#             st.session_state.chat_history.append({"role": "assistant", "content": response})
#             return

#     # --- 7. Brand Name Query (exact match) ---
#     if prompt_lower in brand_names:
#         st.session_state.last_brand_context = prompt_lower
#         st.session_state.last_vehicle_name = None
#         with st.spinner(f"Finding popular models for {prompt_lower.title()}..."):
#             brand_models = get_brand_models(prompt_lower, df_main)
#             if not brand_models:
#                 response = f"Sorry, I couldn't find any models for the brand {prompt_lower.title()}."
#             else:
#                 display_brand_summary(prompt_lower, brand_models)
#                 return
#         if response:
#             with st.chat_message("assistant"):
#                 st.markdown(response)
#             st.session_state.chat_history.append({"role": "assistant", "content": response})
#             return

#     # --- 8. Reasons to buy queries for specific vehicles ---
#     if any(keyword in prompt_lower for keyword in ["reasons to buy", "why buy", "should i buy", "pros of", "benefits of"]) and (vehicle_name := find_vehicle_in_text(prompt, display_names)):
#         st.session_state.last_vehicle_name = vehicle_name
#         st.session_state.last_brand_context = None
#         data = get_vehicle_data(vehicle_name, df_main)
#         with st.spinner("Analyzing reasons to buy..."):
#             reasons_prompt = f"What are the key reasons to buy the {vehicle_name}? Focus on its benefits, pros, and unique selling points. Keep it concise but informative."
#             result = qa_chain.invoke({"question": reasons_prompt})
#             answer = result.get("result", f"Here are some reasons to consider the {vehicle_name}.")
#         display_vehicle_with_llm_response(data, answer)
#         return

#     # --- 9. Exact QnA Match ---
#     qna_answer = find_qna_answer(prompt, df_qna)
#     if qna_answer:
#         with st.chat_message("assistant"):
#             st.markdown(qna_answer)
#         st.session_state.chat_history.append({"role": "assistant", "content": qna_answer})
#         return

#     # 4a. If no direct QnA match, use QnA CSV as context for LLM
#     # Only do this for questions about 91trucks/company/brand (not for vehicle specs etc.)
#     if any(x in prompt.lower() for x in ["91trucks", "91 trucks", "founder", "company", "contact", "about", "who is", "what is", "help", "support"]):
#         # Limit to top 10 QnA pairs for context to avoid overloading the LLM
#         qna_context = ""
#         for _, row in df_qna.head(10).iterrows():
#             qna_context += f"Q: {row['question']}\nA: {row['answer']}\n"
#         llm_prompt = (
#             f"User question: {prompt}\n"
#             f"Here are some Q&A pairs from the 91Trucks dataset:\n{qna_context}\n"
#             "Answer the user's question using the Q&A pairs above and your own knowledge. Be concise and helpful."
#         )
#         with st.spinner("Thinking..."):
#             result = qa_chain.invoke({"question": llm_prompt})
#             answer = result.get("result", "Sorry, I couldn't generate an answer.")
#         with st.chat_message("assistant"):
#             st.markdown(answer)
#         st.session_state.chat_history.append({"role": "assistant", "content": answer})
#         return

#     # --- 10. Fallback to LLM for general queries, but always prefer CSV data ---
#     commercial_keywords = [
#         "truck", "trucks", "van", "vans", "pickup", "pickups", "auto", "autos", "bus", "buses", "vehicle", "vehicles", "commercial", "payload", "gvw", "engine", "mileage", "fuel", "specs", "specifications", "brand", "model", "variant", "power", "capacity", "91trucks"
#     ]
#     if not any(word in prompt_lower for word in commercial_keywords) and not is_greeting(prompt):
#         response = "Hi! I'm your 91Trucks Assistant — here to help with anything related to trucks, vans, and commercial vehicles. Ask me anything about specs, prices, or models!"
#         with st.chat_message("assistant"):
#             st.markdown(response)
#         st.session_state.chat_history.append({"role": "assistant", "content": response})
#         return

#     with st.spinner("Thinking..."):
#         # LLM is only used to enhance, not replace, CSV data
#         result = qa_chain.invoke({"question": prompt})
#         answer = result.get("result") or "Sorry, I couldn't find an answer to that."
#         answer = clean_llm_output(answer)
#         response = answer
#     with st.chat_message("assistant"):
#         st.markdown(response)
#     st.session_state.chat_history.append({"role": "assistant", "content": response})

# # --- User Input Handling ---
# if prompt := st.chat_input("Ask about a vehicle, e.g., 'Tata Intra v30 vs Mahindra Jeeto'"):
#     st.session_state.chat_history.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)
#     process_query(prompt)
#     # Only clear user_input after processing
#     st.session_state.user_input = ""

# if st.session_state.user_input:
#     # Handle button clicks from suggestions (should not clear chat history)
#     query = st.session_state.user_input
#     st.session_state.user_input = "" # Clear only after processing
#     st.session_state.chat_history.append({"role": "user", "content": query})
#     with st.chat_message("user"):
#         st.markdown(query)
#     process_query(query)