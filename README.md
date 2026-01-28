# AI-Powered Resume Screening System
[![Streamlit Live App](https://img.shields.io/badge/Streamlit-Live%20App-5314C4?logo=streamlit)](https://ai-resumeranker-dm.streamlit.app/)

Automatically analyze, score, and rank candidate resumes against job descriptions using NLP and Machine Learning.

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
2. The application will open in your default web browser.

3. **In the App:**
   - **Step 1:** Paste a job description or upload a JD file (TXT/DOCX).
   - **Step 2:** Upload one or more resumes (PDF, DOCX, TXT).
   - **Step 3:** Click "Analyze & Rank Candidates".
   - **Step 4:** Review the table of ranked candidates and expand rows for detailed scoring breakdowns.



## Features

- **Multi-Format Support:** Handles PDF, DOCX, and TXT resume formats.
- **Skill Extraction:** Identifies programming languages, frameworks, databases, and soft skills using a predefined dictionary.
- **Intelligent Scoring:**
    - **TF-IDF Similarity (40%):** Matches overall content relevance.
    - **Skill Match (35%):** Checks for required technical skills.
    - **Keyword Density (10%):** Frequency of key job terms.
    - **Experience (15%):** Baseline score for experience presence.
- **Visualizations:** Score distribution charts and color-coded rankings.
- **Export:** Download results as CSV.

