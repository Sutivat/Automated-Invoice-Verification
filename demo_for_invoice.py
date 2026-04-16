import streamlit as st
import pandas as pd
import json
import requests
import re
import os
import time
import tempfile
from typhoon_ocr import ocr_document
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. CONSTANTS & RATE LIMITS
# ==========================================
# 20 req/min = 1 request every 3 seconds. 
# We use 3.1 to be safe against network jitter.
API_RATE_LIMIT_DELAY = 3.1 

# ==========================================
# 2. APP CONFIG & SESSION STATE
# ==========================================
st.set_page_config(page_title="Typhoon Invoice AI", layout="wide")

# Persistent storage for results so they don't disappear on re-run
if "invoice_results" not in st.session_state:
    st.session_state.invoice_results = []

# --- Sidebar Configuration ---
st.sidebar.title("⚙️ System Configuration")
mode = st.sidebar.radio("Processing Mode", ["LOCAL (Ollama)", "API (Typhoon Cloud)"])

if mode == "LOCAL (Ollama)":
    OCR_BASE_URL = "http://localhost:11434/v1"
    OCR_MODEL = "scb10x/typhoon-ocr1.5-3b"
    LLM_BASE_URL = "http://localhost:11434/v1"
    LLM_MODEL = "scb10x/typhoon2.5-qwen3-4b"
    API_KEY = "ollama"
    st.sidebar.info("Running locally - No rate limits applied.")
else:
    OCR_BASE_URL = "https://api.opentyphoon.ai/v1"
    OCR_MODEL = "typhoon-ocr"
    LLM_BASE_URL = "https://api.opentyphoon.ai/v1"
    LLM_MODEL = "typhoon-v2.5-30b-a3b-instruct"
    API_KEY = os.getenv("typhoon_api_key")
    st.sidebar.success("Cloud API Mode - 20 req/m limit enforced.")

client = OpenAI(api_key=API_KEY, base_url=LLM_BASE_URL)

# Mock Database
DATABASE_PO = {
    "PO-2023-001": {"vendor": "บริษัท วัสดุก่อสร้าง จำกัด", "approved_amount": 10700.00},
    "PO-2023-002": {"vendor": "ร้านเหล็กไทย", "approved_amount": 50000.00},
    "IV1806-0002": {"vendor": "surebattstore", "approved_amount": 410.00}
}

# ==========================================
# 3. UTILITY FUNCTIONS
# ==========================================
def clean_json_response(text):
    text = re.sub(r'```json', '', text)
    text = re.sub(r'```', '', text)
    return text.strip()

def safe_float_convert(value):
    """Cleans currency strings like '1,200.50' to float."""
    try:
        if isinstance(value, str):
            value = value.replace(',', '').strip()
        return float(value)
    except:
        return 0.0

# ==========================================
# 4. CORE PROCESSING LOGIC
# ==========================================
def process_single_invoice(uploaded_file):
    """Processes a single file with built-in retry logic for Rate Limits."""
    # Use tempfile to avoid file collision
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:
        # Step 1: OCR
        markdown_text = ocr_document(
            tmp_path, 
            base_url=OCR_BASE_URL, 
            api_key=API_KEY,
            model=OCR_MODEL, 
            page_num=1
        )

        # Step 2: Extraction
        prompt = f"Return JSON only: 'invoice_number', 'po_reference', 'total_amount', 'vendor_name'. Text: {markdown_text}"
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=4096
        )
        
        raw_json = clean_json_response(response.choices[0].message.content)
        data = json.loads(raw_json)
        
        # Step 3: Verification
        po_ref = data.get("po_reference")
        inv_amt = safe_float_convert(data.get("total_amount", 0))
        
        status = "❌ FAILED"
        if po_ref in DATABASE_PO:
            db_po = DATABASE_PO[po_ref]
            if abs(inv_amt - db_po["approved_amount"]) < 0.01:
                status = "✅ PASSED"
            else:
                status = f"⚠️ MISMATCH (DB: {db_po['approved_amount']})"
        else:
            status = "❓ PO NOT FOUND"
            
        return {
            "filename": uploaded_file.name,
            "invoice_number": data.get("invoice_number"),
            "po_reference": po_ref,
            "total_amount": inv_amt,
            "vendor_name": data.get("vendor_name"),
            "verification_status": status,
            "approved": False
        }

    except Exception as e:
        # Check specifically for rate limit errors
        err_msg = str(e)
        if "429" in err_msg:
            return "RATE_LIMIT_ERROR"
        return {"filename": uploaded_file.name, "verification_status": f"Error: {err_msg}", "approved": False}
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# ==========================================
# 5. UI LAYOUT
# ==========================================
st.title("📑 Smart Finance Verifier (Pro)")
st.caption(f"Engine: {LLM_MODEL} | Protection: Active")

files = st.file_uploader("Upload Invoices", accept_multiple_files=True, type=['png', 'jpg', 'pdf'])

if files:
    if st.button("🚀 Start Batch Processing"):
        st.session_state.invoice_results = [] # Clear previous
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, file in enumerate(files):
            status_text.text(f"Processing {i+1}/{len(files)}: {file.name}")
            
            # Record start time for throttling
            start_time = time.time()
            
            # Process the invoice
            result = process_single_invoice(file)
            
            # Handle Retry if hit rate limit
            if result == "RATE_LIMIT_ERROR":
                st.warning("Hit API Limit. Cooling down for 10s...")
                time.sleep(10)
                result = process_single_invoice(file) # Try once more

            st.session_state.invoice_results.append(result)
            
            # ENFORCE RATE LIMIT (Only in API Mode)
            if mode == "API (Typhoon Cloud)":
                elapsed = time.time() - start_time
                if elapsed < API_RATE_LIMIT_DELAY:
                    time.sleep(API_RATE_LIMIT_DELAY - elapsed)
            
            progress_bar.progress((i + 1) / len(files))
        
        status_text.success("Batch Processing Complete!")

# ==========================================
# 6. HUMAN-IN-THE-LOOP & EXPORT
# ==========================================
if st.session_state.invoice_results:
    st.divider()
    st.subheader("👨‍💻 Human Review & Approval")
    
    df = pd.DataFrame(st.session_state.invoice_results)
    
    # Editable table for corrections
    edited_df = st.data_editor(
        df,
        column_config={
            "approved": st.column_config.CheckboxColumn("Approve?", default=False),
            "total_amount": st.column_config.NumberColumn("Amount (THB)", format="%.2f"),
            "verification_status": st.column_config.TextColumn("Status", disabled=True),
            "filename": st.column_config.TextColumn("Source File", disabled=True),
        },
        hide_index=True,
        use_container_width=True
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Finalize & Save Approved"):
            count = len(edited_df[edited_df['approved'] == True])
            st.balloons()
            st.success(f"Finalized {count} invoices for payment.")
            
    with col2:
        # Use utf-8-sig for Excel Thai language support
        csv = edited_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 Export to CSV for Excel",
            data=csv,
            file_name=f"finance_audit_{int(time.time())}.csv",
            mime="text/csv"
        )