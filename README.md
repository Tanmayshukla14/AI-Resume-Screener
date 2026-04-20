# Precision Analyst 🎯 — Enterprise AI Resume Screener

An enterprise-grade, lightning-fast application built with Python and **Streamlit**. Precision Analyst evaluates resumes against Job Descriptions using **LangChain** and **OpenRouter** (`openai/gpt-4o-mini`), enforcing strict structured logic via **Pydantic**. 

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red)
![LangChain](https://img.shields.io/badge/LangChain-Native-black)
![OpenRouter](https://img.shields.io/badge/OpenRouter-AI-purple)

## ✨ Core Features
- **Structured AI Reasoning:** Replaces raw prompt engineering with strict Pydantic JSON schemas, guaranteeing reliable extraction of 11 candidate data points.
- **5-Factor Deterministic Scoring:** Prevents LLM scoring hallucinations by applying a deterministic, weighted mathematical calculation: Skills (35%), Experience (30%), Keywords (20%), Gaps (10%), and Seniority (5%).
- **Deep Security Hardening (`security.py`):** 
  - **Magic Byte Validation:** Halts execution if a non-PDF file disguised as a `.pdf` is uploaded.
  - **Rate Limiting:** Configured sliding-window memory (max 50 calls/hour) to prevent API abuse.
  - **Sanitization:** Deep path-traversal prevention and strict HTML-escaping against XSS injections.
- **Ultra-Premium UI:** Re-architected with a clean, Vercel-inspired aesthetic. Bypasses standard Streamlit markdown parsing using `st.html()` to inject pristine custom HTML components including interactive SVG score rings, runner-up grids, and expandable Candidate Deep-Dives.
- **Actionable Output:** Includes one-sentence executive TL;DRs and custom-generated interview probing questions targeting the candidate's exact weaknesses.

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Tanmayshukla14/AI-Resume-Screener.git
   cd AI-Resume-Screener
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Keys:**
   Create a `.env` file in the root directory. This project uses **OpenRouter** to handle model inference.
   ```env
   # Add your OpenRouter Key (sk-or-v1-...)
   OPENAI_API_KEY="sk-or-v1-xxxxxxxx"
   ```

4. **Launch Application:**
   ```bash
   streamlit run app.py
   ```

## 🏗️ Architecture Overview
* `app.py`: The presentation layer. Renders the Vercel-style UI, handles concurrent file parsing loops, and coordinates the components natively without iframes.
* `llm_engine.py`: The brain. Interfaces with LangChain and Pydantic. Tracks error trees cleanly separating Auth/Network/Quota triggers.
* `scoring.py`: The calculator. Null-safe algorithms that output consistent 0-100 `int` scores and color-coded recommendation tiers.
* `security.py`: The defensive layer. Enforces rules regarding file payload size, rate-limiting, missing files, and injection defense.
* `parser.py`: The extractor. Utilizes `pdfplumber` to pull text payloads from multi-page PDFs cleanly while handling silent OCR warnings securely.
