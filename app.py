import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

st.markdown(
    """
    <style>
    .stApp {
        background: url("https://source.unsplash.com/1600x900/?technology,abstract") no-repeat center center fixed;
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)


import streamlit as st
import base64

def add_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
add_bg_from_local("img.avif")

# Load the model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Function to summarize text
def summarize(text):
    input_text = "summarize: " + text
    inputs = tokenizer.encode(input_text, return_tensors='pt', max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Streamlit app interface
import streamlit as st

st.markdown(
    """
    <div style="text-align: center;">
        <h1 style="color: #FFFFFF;">ClauseWise Legisum</h1>
    </div>
    """,
    unsafe_allow_html=True
)


# Text input field for user to input the Terms & Conditions text
text_input = st.text_area("Paste your Terms and Conditions here:", height=300)

# Button to generate the summary
if st.button("✨Generate Summary"):
    if text_input:  # Ensure the input is not empty
        summary = summarize(text_input)
        st.write("Summary:")
        st.text(summary)
    else:
        st.warning("Please paste the Terms and Conditions text first.")
