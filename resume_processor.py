import os
import json
import re
import pandas as pd
import numpy as np
import spacy
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import PyPDF2
import docx
import subprocess
import sys

# Silent install of models if missing - Critical for "run in one go" deployment
def download_spacy_model(model_name="en_core_web_sm"):
    try:
        if not spacy.util.is_package(model_name):
            print(f"Downloading {model_name}...")
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
    except Exception as e:
        print(f"Failed to auto-download spacy model: {e}")

download_spacy_model()

# Ensure NLTK data is downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class ResumeProcessor:
    def __init__(self, skills_dict_path='skills_dict.json'):
        """Initialize the processor with skills dictionary and NLP models."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
             # Fallback if load fails even after check
             download_spacy_model()
             self.nlp = spacy.load("en_core_web_sm")
             
        self.stop_words = set(stopwords.words('english'))
        
        # Load skills dictionary
        try:
            with open(skills_dict_path, 'r') as f:
                self.skills_dict = json.load(f)
        except FileNotFoundError:
            # Fallback path if running from app logic
             with open(os.path.join(os.path.dirname(__file__), skills_dict_path), 'r') as f:
                self.skills_dict = json.load(f)
                
        # Flatten skills for easier searching
        self.all_skills = set()
        for category in self.skills_dict.values():
            for skill in category:
                self.all_skills.add(skill.lower())

    def extract_text_from_pdf(self, file_obj):
        """Extract text from PDF file object."""
        try:
            pdf_reader = PyPDF2.PdfReader(file_obj)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            return f"Error extraction PDF: {str(e)}"

    def extract_text_from_docx(self, file_obj):
        """Extract text from DOCX file object."""
        try:
            doc = docx.Document(file_obj)
            text = []
            for paragraph in doc.paragraphs:
                text.append(paragraph.text)
            return "\n".join(text)
        except Exception as e:
            return f"Error extracting DOCX: {str(e)}"

    def extract_text_from_txt(self, file_obj):
        """Extract text from TXT file object."""
        try:
            return file_obj.read().decode('utf-8')
        except Exception as e:
             # Try different encoding if utf-8 fails
            try:
                return file_obj.read().decode('latin-1')
            except:
                return f"Error extracting TXT: {str(e)}"

    def preprocess_text(self, text):
        """Clean and preprocess text using SpaCy."""
        # specific cleaning for technical terms preservation
        text = text.replace('\n', ' ')
        
        # Helper to preserve C++, C#, .NET during tokenization or cleaning if needed
        # SpaCy generally handles these okay, but we clean surrounding noise.
        
        doc = self.nlp(text)
        
        clean_tokens = []
        for token in doc:
            # Keep meaningful tokens
            if not token.is_stop and not token.is_punct and not token.is_space:
                clean_tokens.append(token.lemma_.lower())
                
        return " ".join(clean_tokens)

    def extract_skills(self, text):
        """Extract skills from text based on dictionary."""
        text_lower = text.lower()
        found_skills = []
        
        # We search for exact matches of skills
        # For multi-word skills, this is simple substring match. 
        # For single word, we might want boundary checks, but fast loop is okay for now.
        # Improvement: Use regex with word boundaries for efficiency and accuracy
        
        for skill in self.all_skills:
            # Simple regex for word boundary to avoid partial matches like 'java' in 'javascript' NO, 'javascript' contains 'java'
            # But 'java' shouldn't match 'javascript'.
            # Use \b pattern. Escape special regex chars in skill names (like C++, C#)
            
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                # mapped back to original casing if possible, or just append title case/original from dict
                # For now, let's just use the skill string from the set (which is lower) or find original.
                # Let's simple append the detected skill string (we'll title case it for display)
                found_skills.append(skill.capitalize()) 
                
        # Handle special cases that \b might miss or mess up like C++ or .NET if not carefully handled
        # manual check for C++, C#, .NET
        if "c++" in text_lower and "c++" in self.all_skills: found_skills.append("C++")
        if "c#" in text_lower and "c#" in self.all_skills: found_skills.append("C#")
        if ".net" in text_lower and ".net" in self.all_skills: found_skills.append(".NET")
        
        return list(set(found_skills))

    def vectorize_text(self, texts):
        """Convert list of texts to TF-IDF vectors."""
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=1,
            max_df=1.0  # Relaxed max_df for small batch sizes
        )
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            return tfidf_matrix, vectorizer
        except ValueError:
            # Handle empty vocabulary case
            return None, None

    def calculate_similarity(self, resume_vector, jd_vector):
        """Calculate cosine similarity between resume and JD."""
        if resume_vector is None or jd_vector is None:
            return 0.0
        return cosine_similarity(resume_vector, jd_vector)[0][0]

    def calculate_skill_match_score(self, resume_skills, jd_skills):
        """Calculate percentage of JD skills found in resume."""
        if not jd_skills:
            return 0.0, []
            
        resume_skills_set = set(s.lower() for s in resume_skills)
        jd_skills_set = set(s.lower() for s in jd_skills)
        
        matched_skills = resume_skills_set.intersection(jd_skills_set)
        
        score = (len(matched_skills) / len(jd_skills_set)) * 100
        
        # Return display friendly matched skills
        display_matched = [s.capitalize() for s in matched_skills]
        return score, display_matched

    def calculate_keyword_density(self, resume_text, jd_text):
        """Calculate density of top JD keywords in resume."""
        # Simple implementation: extract top keywords from JD using TF-IDF or Frequency
        # Here we'll use frequency of significant words
        
        # clean jd
        jd_doc = self.nlp(jd_text.lower())
        jd_words = [token.text for token in jd_doc if not token.is_stop and not token.is_punct and token.is_alpha]
        
        if not jd_words:
            return 0.0
            
        common_words = Counter(jd_words).most_common(10)
        top_keywords = [word for word, count in common_words]
        
        resume_lower = resume_text.lower()
        found_count = 0
        for word in top_keywords:
            if word in resume_lower:
                found_count += 1
                
        # Score is what % of top 10 keywords are present
        return (found_count / len(top_keywords)) * 100 if top_keywords else 0.0

    def calculate_composite_score(self, tfidf_sim_score, skill_match_score, keyword_density_score):
        """Calculate final composite score."""
        # Weights
        w_tfidf = 0.40
        w_skill = 0.35
        w_keyword = 0.10
        w_exp = 0.15 # Experience placeholder
        
        # Normalize tfidf (0-1) to 0-100
        tfidf_100 = tfidf_sim_score * 100
        
        # Assuming experience is neutral (50) for now as we don't have explicit exp extraction
        exp_score = 50 
        
        final_score = (w_tfidf * tfidf_100) + \
                      (w_skill * skill_match_score) + \
                      (w_keyword * keyword_density_score) + \
                      (w_exp * exp_score)
                      
        return min(100.0, max(0.0, final_score))

    def process_resumes(self, resume_files, job_description):
        """Main pipeline to process resumes against a job description."""
        
        results = []
        
        # 1. Process JD
        jd_clean = self.preprocess_text(job_description)
        jd_skills = self.extract_skills(job_description)
        
        # 2. Process all resumes
        corpus = [jd_clean] # Index 0 is JD
        resume_data = []
        
        for file_obj in resume_files:
            # Extract text
            filename = file_obj.name
            file_ext = os.path.splitext(filename)[1].lower()
            
            if file_ext == '.pdf':
                text = self.extract_text_from_pdf(file_obj)
            elif file_ext == '.docx':
                text = self.extract_text_from_docx(file_obj)
            elif file_ext == '.txt':
                text = self.extract_text_from_txt(file_obj)
            else:
                text = "" # Skip or handle error
                
            clean_text = self.preprocess_text(text)
            skills = self.extract_skills(text)
            
            resume_data.append({
                'name': filename,
                'raw_text': text,
                'clean_text': clean_text,
                'skills': skills
            })
            corpus.append(clean_text)
            
        # 3. Vectorize
        if len(corpus) > 1:
            tfidf_matrix, _ = self.vectorize_text(corpus)
            jd_vector = tfidf_matrix[0]
            
            # 4. Score and Build Results
            for i, data in enumerate(resume_data):
                resume_vector = tfidf_matrix[i+1] # +1 because 0 is JD
                
                # Scores
                sim_score = self.calculate_similarity(resume_vector, jd_vector)
                skill_score, matched_skills = self.calculate_skill_match_score(data['skills'], jd_skills)
                keyword_score = self.calculate_keyword_density(data['raw_text'], job_description)
                
                final_score = self.calculate_composite_score(sim_score, skill_score, keyword_score)
                
                results.append({
                    'Candidate Name': data['name'],
                    'Score': round(final_score, 1),
                    'TF-IDF Score': round(sim_score * 100, 1),
                    'Skill Match %': round(skill_score, 1),
                    'Keyword Density': round(keyword_score, 1),
                    'Matched Skills': matched_skills,
                    'All Extracted Skills': data['skills']
                })
        else:
             # handle case where vectorization fails or no resumes
             pass

        # Create DataFrame
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values(by='Score', ascending=False)
            
        return df
