import streamlit as st
import pandas as pd
import os
import re # For extracting code blocks from LLM output
import matplotlib.pyplot as plt # For running Matplotlib/Seaborn plots
import plotly.express as px # For running Plotly plots
import seaborn as sns # For running Seaborn plots
import textwrap # Used for proper indentation in exec() code execution

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage, AIMessage

from dotenv import load_dotenv

# Load environment variables from .env file (e.g., NVIDIA_API_KEY)
load_dotenv()

# =================================================================
# --- 0. FILE READING AND CODE EXECUTION FUNCTIONS ---
# =================================================================

def read_uploaded_file(uploaded_file):
    """Reads the uploaded file based on its extension."""
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension == 'csv':
        df = pd.read_csv(uploaded_file)
    elif file_extension in ['xlsx', 'xls']:
        # Requires the openpyxl library
        df = pd.read_excel(uploaded_file, engine='openpyxl')
    elif file_extension == 'json':
        df = pd.read_json(uploaded_file)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")
        
    return df


def extract_and_execute_code(response_content, df):
    """
    Extracts a Python code block from the LLM response, modifies it to ensure 
    a figure object is returned, and safely executes it.
    """
    # Find all Python code blocks (using re.DOTALL to capture multiline blocks)
    code_blocks = re.findall(r'```python\n(.*?)```', response_content, re.DOTALL)
    
    if code_blocks:
        # Use an expander for clean display of code and visual
        with st.expander("📊 Generated Visual & Code", expanded=True):
            original_code = code_blocks[0]
            fig_variable_name = None 
            
            # --- 1. CODE MODIFICATION LOGIC ---
            
            # Matplotlib/Seaborn plotting logic
            if 'plt.' in original_code or 'sns.' in original_code:
                # Attempt to find the variable assigned to the figure
                fig_match = re.search(r'(\w+)\s*=\s*(?:plt\.|sns\.)', original_code)
                fig_variable_name = fig_match.group(1) if fig_match else 'plt.gcf()'

                # Clean up display commands and wrap the code in a function
                modified_code = original_code.replace("plt.show()", "")
                modified_code = re.sub(r'st\.pyplot\s*\(.*?\)', '', modified_code)
                
                code_to_execute = (
                    "import matplotlib.pyplot as plt\n"
                    "import seaborn as sns\n"
                    "def generate_plot(data, pd, st, plt, px, sns):\n"
                    f"{textwrap.indent(modified_code, '    ')}\n"
                    f"    return {fig_variable_name}" # Force return of the figure
                )
                display_function = st.pyplot

            # Plotly plotting logic
            elif 'px.' in original_code or 'go.' in original_code:
                # Attempt to find the variable assigned to the chart
                fig_match = re.search(r'(\w+)\s*=\s*(?:px\.|go\.)', original_code)
                fig_variable_name = fig_match.group(1) if fig_match else 'fig'

                # Clean up display commands and wrap the code in a function
                modified_code = original_code
                modified_code = re.sub(r'st\.plotly_chart\s*\(.*?\)', '', modified_code)
                
                code_to_execute = (
                    "import plotly.express as px\n"
                    "import plotly.graph_objects as go\n"
                    "def generate_plot(data, pd, st, plt, px, sns):\n"
                    f"{textwrap.indent(modified_code, '    ')}\n"
                    f"    return {fig_variable_name}" # Force return of the chart
                )
                display_function = st.plotly_chart

            else:
                # Non-plotting code (should be rare/empty due to new system prompt rule)
                code_to_execute = original_code
                display_function = None

            # --- 2. EXECUTION ---
            # Set up the execution environment (globals)
            exec_globals = {
                'data': df, 
                'st': st,
                'pd': pd,
                'plt': plt,
                'px': px,
                'sns': sns
            }
            
            st.code(code_to_execute, language="python") # Display the modified code
            
            try:
                # Execute the code/function definition
                exec(code_to_execute, exec_globals)
                
                if display_function:
                    # If a function was defined, call it to get the plot object
                    plot_object = exec_globals['generate_plot'](
                        df, pd, st, plt, px, sns
                    )
                    
                    # Display the plot/chart
                    display_function(plot_object)
                    
                    # Clean up Matplotlib state to prevent display issues
                    if display_function == st.pyplot:
                        plt.close('all')

            except Exception as e:
                # Display execution errors clearly to the user
                st.error(f"❌ Code Execution Failed! The generated code could not run.")
                st.error(f"Error: {e}")
                st.markdown(f"**Code attempted:**\n```python\n{code_to_execute}\n```")


# =================================================================
# --- 1. CONFIGURATION AND INITIALIZATION ---
# =================================================================

nvidia_api_key = os.getenv("NVIDIA_API_KEY")

# --- LangChain LLM Setup ---
if "NVIDIA_API_KEY" not in os.environ and not st.session_state.get("api_key_entered"):
    st.error("Please set the NVIDIA_API_KEY environment variable or enter it in the sidebar.")
    st.stop()

# Initialize the LLM (Llama 3 8B on NVIDIA NIM)
try:
    api_key = st.session_state.get("api_key_entered") or os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        st.error("NVIDIA API Key is missing.")
        st.stop()

    llm = ChatNVIDIA(
        model="meta/llama3-8b-instruct",
        nvidia_api_key=api_key,
        temperature=0.1
    )
except Exception as e:
    st.error(f"Error initializing LLM: {e}")
    st.stop()

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="Hello! Upload a data file (CSV, Excel, or JSON) on the left sidebar to begin, and I will summarize it. Then, feel free to ask me questions about the data.")
    ]
if "summary" not in st.session_state:
    st.session_state.summary = None
if "insight" not in st.session_state: # New state for storing deep insights
    st.session_state.insight = None


# =================================================================
# --- 2. CORE FUNCTIONS (SYSTEM PROMPT) ---
# =================================================================

def generate_data_summary(df):
    """Generates a structural summary of the DataFrame using the LLM."""
    try:
        summary_text = f"""
        The uploaded dataset has {len(df)} rows and {len(df.columns)} columns.
        The column names and their first few non-null values are:
        {df.head(2).to_markdown(index=False)}

        The data types are:
        {df.dtypes.to_frame('Data Type').to_markdown()}
        """

        system_prompt = f"""
        You are an expert Data Analyst AI.
        Your task is to provide a concise, insightful summary of the **structure and content** of the dataset based on the details provided.
        Do not generate any code or plot.

        DATASET CONTEXT:
        {summary_text}
        """

        prompt = ChatPromptTemplate.from_messages([("system", system_prompt)])
        chain = prompt | llm

        response = chain.invoke({})
        return response.content

    except Exception as e:
        return f"An error occurred during summarization: {e}"

def generate_data_insights(df):
    """Generates deep, analytical insights from the DataFrame using the LLM."""
    try:
        summary_text = f"""
        The uploaded dataset has {len(df)} rows and {len(df.columns)} columns.
        The column names, first few non-null values, and data types are:
        {df.head(2).to_markdown(index=False)}

        Data Types:
        {df.dtypes.to_frame('Data Type').to_markdown()}
        """

        # System prompt focused on analysis and insights
        system_prompt = f"""
        You are an expert Data Scientist AI.
        Your task is to provide 3-5 deep, analytical **insights** and **observations** about the following dataset.
        Focus on: potential relationships, trends, outliers, and initial hypotheses.
        Do not generate any code or plot. Just provide the analytical text.

        DATASET CONTEXT:
        {summary_text}
        """

        prompt = ChatPromptTemplate.from_messages([("system", system_prompt)])
        chain = prompt | llm

        response = chain.invoke({})
        return response.content

    except Exception as e:
        return f"An error occurred during insight generation: {e}"


def generate_chat_response(prompt_text, dataframe_content):
    """Generates a response to a user's question, including data context and code execution rules."""
    
    # CRITICAL: Define the available packages and enforce output format rules
    system_prompt = f"""
    You are an AI assistant specialized in analyzing and chatting about data.
    The current dataset is available as a DataFrame named 'data'.

    If the user asks for a visual (scatterplot, histogram, etc.), you MUST output the Python plotting code
    in a markdown block (```python...```) using the 'data' variable.
    
    **CRITICAL RULE 1: Only use the following pre-installed packages: pandas, numpy, matplotlib.pyplot (as plt), seaborn (as sns), plotly.express (as px), and statsmodels.api (as sm).**
    
    **CRITICAL RULE 2: If the user asks for a table of statistical results (e.g., correlation matrix, descriptive statistics, regression summary), you MUST calculate and present those results directly as a well-formatted MARKDOWN TABLE in your response text, NOT within a Python code block. Only use a Python block for VISUALIZATIONS.**

    **CRITICAL VISUALIZATION STEP**: When creating plots that use many columns (like pairplots), always get the list of numerical columns using the robust method: `numerical_cols = data.select_dtypes(include=['number']).columns.tolist()`. This prevents ambiguous truth value errors.

    DATASET (First 10 rows):
    {dataframe_content}
    
    NOTE: The user is asking a direct question about this data. Use only this data context.
    """

    # Build the full list of messages for the LLM context
    history = [msg for msg in st.session_state.messages if isinstance(msg, (HumanMessage, AIMessage))]
    # Prepend the system prompt and append the new user prompt
    full_messages = [HumanMessage(content=system_prompt)] + history + [HumanMessage(content=prompt_text)]
    
    # Invoke the model directly with the list of messages
    response = llm.invoke(full_messages)
    return response.content


# =================================================================
# --- 3. STREAMLIT LAYOUT ---
# =================================================================

st.set_page_config(layout="wide", page_title="NVIDIA NIM Data Analyst")
st.title("📊 NVIDIA NIM Data Analyst Assistant") # Removed [Image of a data visualization dashboard] for clarity

# --- LEFT SIDEBAR: Data Upload and Analysis Control ---
with st.sidebar:
    st.header("1. Upload Data (CSV, Excel, JSON)")
    uploaded_file = st.file_uploader("Choose a data file", type=["csv", "xlsx", "xls", "json"])

    if 'api_key_entered' not in st.session_state:
        st.session_state.api_key_entered = None

    # Handle file upload and analysis trigger
    if uploaded_file is not None:
        try:
            df = read_uploaded_file(uploaded_file)
            st.session_state.df = df
            st.success("File uploaded successfully!")
            st.subheader("Data Preview")
            st.dataframe(df.head())

            st.subheader("2. Initial Analysis (Llama 3)")
            
            # --- Button for Summary (Structural Overview) ---
            if st.button("Generate Summary (Overview)"):
                with st.spinner("Analyzing data structure..."):
                    summary = generate_data_summary(df)
                    st.session_state.summary = summary
                    st.session_state.insight = None # Clear insights when generating summary
                
                st.session_state.messages.append(AIMessage(content=summary))
                st.rerun() # Refresh chat window
            
            # --- Button for Insights (NEW: Deep Analysis) ---
            if st.button("Generate Insights (Deep Dive)"):
                with st.spinner("Generating deep analytical insights..."):
                    insights = generate_data_insights(df)
                    st.session_state.insight = insights
                    st.session_state.summary = None # Clear summary when generating insights

                st.session_state.messages.append(AIMessage(content=insights))
                st.rerun()

            # --- Display Previous Analysis ---
            if st.session_state.get("summary"):
                st.markdown("**Previous Summary:**")
                st.info(st.session_state.summary)
            elif st.session_state.get("insight"):
                st.markdown("**Previous Insights:**")
                st.info(st.session_state.insight)

        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.session_state.df = None
            st.session_state.summary = None
            st.session_state.insight = None # Also clear insights

    else:
        st.session_state.df = None
        st.session_state.summary = None
        st.session_state.insight = None
        st.info("Upload a data file to enable the chat feature.")


# --- MAIN COLUMN: Discussion and Chat ---

st.header("💬 Data Discussion (Q&A)")

# Display chat messages from history
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Handle user input for the chat
if st.session_state.get("df") is not None:
    if prompt := st.chat_input("Ask a question about your uploaded data..."):
        df_current = st.session_state.df
        
        # 1. Display user message immediately
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Get LLM response and display it
        with st.chat_message("assistant"):
            with st.spinner("NIM is thinking..."):
                # Pass the first 10 rows as context
                data_context = df_current.head(10).to_markdown(index=False)
                full_response = generate_chat_response(prompt, data_context)
                st.markdown(full_response)
        
        # 3. Add assistant response to history
        st.session_state.messages.append(AIMessage(content=full_response))
        
        # 4. Extract and execute code from the response
        extract_and_execute_code(full_response, df_current)
        
        # NOTE: st.rerun() is removed here to allow the visual to persist.
