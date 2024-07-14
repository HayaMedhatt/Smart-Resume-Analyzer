import spacy
import streamlit as st
from transformers import pipeline, AutoModel, AutoTokenizer
import spacy
import os, fitz
import google.generativeai as genai
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# Load models
summarizer = pipeline(task="summarization", model="facebook/bart-large-cnn")
job_recommender = pipeline("text-classification", model="Apizhai/Albert-IT-JobRecommendation")

# Configure the API key for Gemini
genai.configure(api_key='') # add your API key

# Configuration for text generation
generation_config = {
    "temperature": 0.5,
    "top_k": 1,
    "max_output_tokens": 2848
}

# Initialize the generative model
model = genai.GenerativeModel(model_name="gemini-1.0-pro")
chat = model.start_chat(history=[])

def extract_all_information(nlp, text):
    doc = nlp(text)
    extracted_data = []
    for ent in doc.ents:
        extracted_data.append((ent.label_, ent.text))
    return extracted_data

def extract_text_from_pdf(uploaded_file):
    try:
        # Save the uploaded file locally
        file_path = "uploaded_resume.pdf"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Process the PDF using PyMuPDF
        text = ""
        with fitz.open(file_path) as pdf:
            for page_num in range(len(pdf)):
                page = pdf.load_page(page_num)
                text += page.get_text()
        text = " ".join(text.split('\n'))

        # Remove the temporary file
        os.remove(file_path)

        return text
    except Exception as e:
        st.error(f"Error processing the PDF: {e}")
        return None

def summarize_resume(text):
    if len(text) >= 1024:
        text = text[:1024]
    summarized_text = summarizer(text, min_length=90, max_length=140)[0]['summary_text']
    return summarized_text


def recommend_job_position(text):
    recommendations = job_recommender(text,top_k = 3)
    return recommendations

def get_skill_recommendations_and_courses(job, skills):
    # Construct the prompt
    prompt = (f"I am applying for a {job} position. I currently have the following skills: {', '.join(skills)}. "
              f"Please recommend additional skills that are more suitable and related to the {job} role, "
              f"and provide some relevant YouTube course videos for these skills.")
    
    # Send the prompt to the Gemini API and get the response
    response = chat.send_message(prompt)
    
    # Return the response text
    return response.text


# Streamlit web app
st.title("Resume Analyzer and Skill Recommendation")

uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

if uploaded_file is not None:
    resume_text = extract_text_from_pdf(uploaded_file)
    
    # Extract information from resume using NER model
    ner_model = spacy.load("nlp_ner_model2")
    extracted_info = extract_all_information(ner_model, resume_text)

    # Display all extracted entities and their labels
    all_extracted_info = extract_all_information(ner_model, resume_text)
    st.header("All Extracted Entities")
    for label, text in all_extracted_info:
        st.write(f"**{label}:** {text}")
        
        # Summarize resume text
    summarized_text = summarize_resume(resume_text)
    st.header("Summarized Resume")
    st.write(summarized_text)
    
    # Recommend job positions based on summarized text
    recommended_job = recommend_job_position(summarized_text)
    st.header("Recommended Job Position")
    for recommendation in recommended_job:
        st.markdown(f"- {recommendation['label']}")
     
       
    # Get skill recommendations and courses
    st.header("Skill Recommendations and Courses")
    skills = [info[1] for info in all_extracted_info if info[0] == "Skills"]
    for recommendation in recommended_job:
        job = recommendation['label']
        #print(skills)
        recommendations = get_skill_recommendations_and_courses(job, skills)
        st.write(recommendations)
        st.write("-----------------------------------")

st.sidebar.header("About")
st.sidebar.write("This app extracts information from a resume and recommends skills based on the extracted designation using a trained deep learning model.")
