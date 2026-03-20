# AI Resume Screener 📄

An ultra-lightweight, lightning-fast application built with Python and **Streamlit** that screens resumes against a Job Description using pure **OpenAI** API integration.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red)
![OpenAI](https://img.shields.io/badge/OpenAI-Native-black)

## ✨ Core Features
- **1-Call Architecture:** Makes exactly ONE API call per resume to keep costs virtually zero (using `gpt-4o-mini`).
- **Strict Scoring Logic:** Analyzes Skills (40%), Experience (30%), Keywords (20%), and Gaps (10%) to automatically categorize candidates into **Strong Fit**, **Moderate Fit**, or **Not Fit**.
- **The Executive TL;DR**: The AI generates a sharp, one-sentence "Value Proposition" summarizing the candidate's exact fit.
- **Dynamic Interview Questions**: Automatically generates targeted behavioral/technical interview questions specifically designed to grill the candidate on their weak points or verify their biggest claims.
- **No Heavy Frameworks**: Built entirely using the standard `requests` library without the bloat of LangChain or LlamaIndex.

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/ai-resume-screener.git
   cd ai-resume-screener
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your OpenAI API Key:**
   Create a `.env` file in the root directory and add your key:
   ```env
   OPENAI_API_KEY="sk-proj-..."
   ```

4. **Run the App:**
   ```bash
   streamlit run app.py
   ```

## 🔒 Security
- **API Keys:** The application pulls standard keys natively from the `.env` file and does not leak or display them in the UI. Ensure `.env` is never committed.
- **XSS Protection:** All scraped PDF data and AI-generated outputs are strictly HTML-escaped before being rendered onto the Streamlit canvas.
