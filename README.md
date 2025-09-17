# AI-Powered-JSON-Translator
# 🚀 Smart JSON Multilingual Translator — Glossary & Review Built-In

## 📋 Overview  
This tool provides an easy way to translate the *values* in your JSON files from English to multiple languages, while keeping the keys unchanged.

Features:
- Supports Spanish, German, French, Romanian, Russian, Polish, Italian, Arabic, Mandarin (SG + CN), and Hindi.
- Editable glossary of domain-specific terms (AML, KYC, OTP, etc.).
- Review and adjust translations before final download.
- Caching for faster repeated translations.

## ✅ Usage Steps
1. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate   # Linux/Mac
    venv\Scripts\activate      # Windows
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the app:
    ```bash
    streamlit run app.py
    ```

4. Open in browser:  
    [http://localhost:8501](http://localhost:8501)

5. Upload your English JSON → Select target languages → Review glossary → Translate → Download translated JSON.

## ⚡ Hosting
- Works locally or on your internal network.  
- Example network URL: `http://172.16.1.186:8502`

## ⚠️ Notes
- Translations use deep-translator (Google NMT unofficial).
- For production, consider switching to OpenAI or Google Cloud Translation API for higher reliability.

---

## 📄 License  
MIT License
