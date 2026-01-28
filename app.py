import streamlit as st
import pandas as pd
from resume_processor import ResumeProcessor
import plotly.express as px
import time

# Page config
st.set_page_config(page_title="AI Resume Screener", layout="wide", page_icon="📄")

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize processor
@st.cache_resource
def load_processor():
    return ResumeProcessor()

try:
    processor = load_processor()
except Exception as e:
    st.error(f"Failed to load processor: {e}")
    st.info("If on Streamlit Cloud, add the model URL to `requirements.txt`. Locally, run: `python -m spacy download en_core_web_sm`")
    st.stop()

# Title and description
st.title("🤖 AI-Powered Resume Screening System")
st.markdown("Upload resumes and job description to get instant candidate rankings based on NLP matching.")

# Sidebar for inputs
with st.sidebar:
    st.header("📥 Input Section")
    
    # Job Description Input
    st.subheader("1. Job Description")
    jd_input_method = st.radio("Input method:", ["Paste Text", "Upload File"])
    
    job_description = ""
    if jd_input_method == "Paste Text":
        job_description = st.text_area("Paste job description here:", height=200, help="Copy and paste the full job description.")
    else:
        jd_file = st.file_uploader("Upload JD file", type=['txt', 'docx'])
        if jd_file:
            # Extract text from uploaded JD file based on extension
            if jd_file.name.endswith('.docx'):
                job_description = processor.extract_text_from_docx(jd_file)
            elif jd_file.name.endswith('.txt'):
                 job_description = jd_file.read().decode('utf-8')
            
            if job_description:
                st.success("JD Loaded successfully!")
            else:
                st.error("Failed to read JD file.")
    
    # Resume Upload
    st.subheader("2. Upload Resumes")
    resume_files = st.file_uploader(
        "Upload resumes (PDF, DOCX, TXT)", 
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True
    )
    
    # Process button
    process_button = st.button("🚀 Analyze & Rank Candidates", type="primary")

# Main content area
if process_button:
    if not job_description:
        st.error("❌ Please provide a job description")
    elif not resume_files:
        st.error("❌ Please upload at least one resume")
    else:
        with st.spinner("🔄 Processing resumes... This may take a moment."):
            # Add a small delay for visual feedback or just process
            start_time = time.time()
            
            # Process all resumes
            try:
                results_df = processor.process_resumes(resume_files, job_description)
                
                # Store in session state
                st.session_state['results'] = results_df
                st.success(f"Processed in {time.time() - start_time:.2f} seconds.")
                
            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")

# Display results if available
if 'results' in st.session_state:
    results_df = st.session_state['results']
    
    if results_df.empty:
        st.warning("No results to display. Please check your inputs.")
    else:
        st.success(f"✅ Analyzed {len(results_df)} candidates")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Candidates", len(results_df))
        col2.metric("Avg Score", f"{results_df['Score'].mean():.1f}")
        col3.metric("Top Score", f"{results_df['Score'].max():.1f}")
        col4.metric("Qualified (>70)", len(results_df[results_df['Score'] > 70]))
        
        # Results table
        st.subheader("📊 Ranked Candidates")
        
        # Format the dataframe for display
        display_df = results_df.copy()
        display_df['Rank'] = range(1, len(display_df) + 1)
        
        # Reorder columns
        cols = ['Rank', 'Candidate Name', 'Score', 'Skill Match %', 'TF-IDF Score', 'Matched Skills']
        display_df = display_df[cols]
        
        # Color code scores
        def color_score(val):
            if val >= 80:
                return 'background-color: #d4edda; color: black;'  # Green
            elif val >= 60:
                return 'background-color: #fff3cd; color: black;'  # Yellow
            else:
                return 'background-color: #f8d7da; color: black;'  # Red
        
        # Apply style
        # Note: Pandas Styler might act differently in Streamlit depending on version, generic fallback usually ok
        try:
            styled_df = display_df.style.map(color_score, subset=['Score'])
        except:
             # Fallback for older pandas
             styled_df = display_df.style.applymap(color_score, subset=['Score'])
             
        st.dataframe(styled_df, use_container_width=True)
        
        # Detailed view - expandable for each candidate
        st.subheader("📋 Detailed Candidate Breakdown")
        for idx, row in results_df.iterrows():
            with st.expander(f"#{idx+1} - {row['Candidate Name']} (Score: {row['Score']:.1f})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Score Breakdown:**")
                    st.write(f"- TF-IDF Similarity: {row['TF-IDF Score']:.1f}% (Content Match)")
                    st.write(f"- Skill Match: {row['Skill Match %']:.1f}%")
                    st.write(f"- Keyword Density: {row['Keyword Density']:.1f}%")
                
                with col2:
                    st.write("**Matched Skills:**")
                    matched_skills = row['Matched Skills']
                    if matched_skills:
                        st.write(", ".join(matched_skills))
                    else:
                        st.write("No specific skill matches found")
                    
                    st.write("**All Extracted Skills:**")
                    st.write(", ".join(row['All Extracted Skills']))
        
        # Visualization
        st.subheader("📈 Score Distribution")
        if len(results_df) > 1:
            fig = px.histogram(results_df, x='Score', nbins=10, 
                            title="Candidate Score Distribution",
                            labels={'Score': 'Score', 'count': 'Number of Candidates'},
                            color_discrete_sequence=['#4CAF50'])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data for distribution plot.")
        
        # Export functionality
        st.subheader("💾 Export Results")
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="candidate_rankings.csv",
            mime="text/csv"
        )

else:
    # Show instructions when no results
    st.info("👈 Use the sidebar to upload job description and resumes, then click 'Analyze & Rank Candidates'")
    
    # Show example/demo section
    st.markdown("""
    ### How It Works:
    1. **Upload Job Description**: Paste or upload the JD for the role
    2. **Upload Resumes**: Add multiple resume files (PDF, DOCX, or TXT)
    3. **Click Analyze**: The AI processes all resumes instantly based on content similarity and skill matching.
    4. **Review Rankings**: See candidates ranked by relevance score
    
    ### Key Features:
    - **Skill Extraction**: Automatically identifies tech stacks and soft skills.
    - **Smart Scoring**: Combines keyword matching with semantic similarity (TF-IDF).
    - **Privacy Focused**: Processing happens locally in session.
    """)
