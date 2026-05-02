import os
import google.generativeai as genai
import gradio as gr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-3-flash-preview")


def fact_check(claim):
    try:
        prompt = f"""
                    You are an expert fact-checking system.

                    Do NOT assume binary truth.
                    If a claim contains partially correct elements, classify it as:
                    - Half True
                    - Partially True
                    - Misleading
                    - Needs Context

                    Analyze carefully.

                    Claim: {claim}

                    Return:
                    1. Verdict (choose from: True, False, Half True, Partially True, Misleading, Not Enough Evidence)
                    2. Detailed reasoning
                    3. Confidence (Low/Medium/High)
                    """


        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return f"Error: {str(e)}"

# Gradio UI
interface = gr.Interface(
    fn=fact_check,
    inputs=gr.Textbox(label="Enter a claim"),
    outputs=gr.Textbox(label="Fact Check Result"),
    title="Fake News checking Model",
)
""
if __name__ == "__main__":
    interface.launch()
