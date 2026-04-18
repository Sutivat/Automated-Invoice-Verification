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
API_RATE_LIMIT_DELAY = 3.1 

# ==========================================
# 2. APP CONFIG & SESSION STATE
# ==========================================
st.set_page_config(page_title="Typhoon Invoice AI (Pro Edition)", layout="wide")

if "invoice_results" not in st.session_state:
    st.session_state.invoice_results = []

st.sidebar.title("⚙️ System Configuration")
mode = st.sidebar.radio("Processing Mode", ["LOCAL (Ollama)", "API (Typhoon Cloud)"])

if mode == "LOCAL (Ollama)":
    OCR_BASE_URL = "http://localhost:11434/v1"
    OCR_MODEL = "scb10x/typhoon-ocr1.5-3b"
    LLM_BASE_URL = "http://localhost:11434/v1"
    LLM_MODEL = "scb10x/typhoon2.5-qwen3-4b"
    API_KEY = "ollama"
else:
    OCR_BASE_URL = "https://api.opentyphoon.ai/v1"
    OCR_MODEL = "typhoon-ocr"
    LLM_BASE_URL = "https://api.opentyphoon.ai/v1"
    LLM_MODEL = "typhoon-v2.5-30b-a3b-instruct"
    API_KEY = os.getenv("typhoon_api_key")

client = OpenAI(api_key=API_KEY, base_url=LLM_BASE_URL)

# Mock Database (จำลองฐานข้อมูล ERP)
DATABASE_PO = {
    "2312/2019": {"vendor": "East Repair Inc.", "tax_id": "0105555555555", "approved_amount": 154.06},
    "xxx-xxxx": {"vendor": "Surebatt Store", "tax_id": "0104444444444", "approved_amount": 50000.00},
    "IV1806-0002": {"vendor": "surebattstore", "tax_id": "0103333333333", "approved_amount": 410.00}
}

# จำลองฐานข้อมูลบิลที่เคยจ่ายเงินไปแล้ว (กันจ่ายซ้ำ)
PAID_INVOICES = {"01234", "INV-9999", "IV1806-0001"}

# ==========================================
# 3. UTILITY FUNCTIONS
# ==========================================
def clean_json_response(text):
    text = re.sub(r'```json', '', text)
    text = re.sub(r'```', '', text)
    return text.strip()

def safe_float_convert(value):
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
    
    # 1. ดึงนามสกุลไฟล์ที่แท้จริงจากชื่อไฟล์ที่อัปโหลด (เช่น .pdf, .jpg, .png)
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    
    # 2. ใช้นามสกุลไฟล์ของจริงในการสร้าง Temp File
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:
        # Step 1: OCR
        markdown_text = ocr_document(
            tmp_path, 
            base_url=OCR_BASE_URL, 
            api_key=API_KEY,
            model=OCR_MODEL, 
            page_num=1 # ถ้าเป็น PDF จะดึงหน้าแรกมา ถ้าเป็นรูปพารามิเตอร์นี้อาจจะถูกข้ามไป
        )
        
        # Step 2: Extraction (เพิ่มการดึง Tax ID, Date, Subtotal, VAT)
        prompt = f"""
        Return JSON only with exact keys: 'invoice_number', 'invoice_date' (YYYY-MM-DD format), 
        'po_reference', 'tax_id', 'vendor_name', 'subtotal' (number), 'vat_amount' (number), 'total_amount' (number). 
        Text: {markdown_text}
        """
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=4096
        )
        
        raw_json = clean_json_response(response.choices[0].message.content)
        data = json.loads(raw_json)
        
        # Step 3: Verification (เพิ่มลอจิกเช็คบิลซ้ำ และ 2-way matching)
        inv_no = data.get("invoice_number", "")
        po_ref = data.get("po_reference")
        inv_amt = safe_float_convert(data.get("total_amount", 0))
        
        status = "❌ FAILED"
        default_action = "Pending"
        remarks = ""

        if inv_no in PAID_INVOICES:
            status = "🚨 DUPLICATE INVOICE"
            default_action = "Reject"
            remarks = "พบประวัติการจ่ายเงินบิลนี้แล้ว"
        elif po_ref in DATABASE_PO:
            db_po = DATABASE_PO[po_ref]
            if abs(inv_amt - db_po["approved_amount"]) < 0.01:
                status = "✅ MATCHED"
                default_action = "Approve"
            else:
                status = f"⚠️ AMT MISMATCH (PO: {db_po['approved_amount']})"
                default_action = "Hold"
                remarks = "ยอดเงินไม่ตรงกับ PO"
        else:
            status = "❓ PO NOT FOUND"
            default_action = "Hold"
            remarks = "ไม่พบเลข PO ในระบบ"
            
        return {
            "filename": uploaded_file.name,
            "action": default_action, # Approve, Hold, Reject
            "invoice_date": data.get("invoice_date", ""),
            "invoice_number": inv_no,
            "po_reference": po_ref,
            "vendor_name": data.get("vendor_name"),
            "tax_id": data.get("tax_id", ""),
            "subtotal": safe_float_convert(data.get("subtotal", 0)),
            "vat_amount": safe_float_convert(data.get("vat_amount", 0)),
            "total_amount": inv_amt,
            "verification_status": status,
            "remarks": remarks
        }

    except Exception as e:
        err_msg = str(e)
        if "429" in err_msg: return "RATE_LIMIT_ERROR"
        return {"filename": uploaded_file.name, "verification_status": f"Error: {err_msg}", "action": "Reject", "remarks": "AI Processing Error"}
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)

# ==========================================
# 5. UI LAYOUT
# ==========================================
st.title("📑 Smart Accounts Payable (AP) Verifier")
st.caption("Automated 2-Way Matching & Duplicate Detection")

files = st.file_uploader("Upload Invoices", accept_multiple_files=True, type=['png', 'jpg', 'pdf'])

if files:
    if st.button("🚀 Start Batch Processing"):
        st.session_state.invoice_results = [] 
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, file in enumerate(files):
            status_text.text(f"Processing {i+1}/{len(files)}: {file.name}")
            start_time = time.time()
            
            result = process_single_invoice(file)
            if result == "RATE_LIMIT_ERROR":
                st.warning("Hit API Limit. Cooling down for 10s...")
                time.sleep(10)
                result = process_single_invoice(file) 

            st.session_state.invoice_results.append(result)
            
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
    st.subheader("👨‍💻 AP Review & Action")
    
    df = pd.DataFrame(st.session_state.invoice_results)
    
    # Editable table for corrections (ปรับให้เป็นระบบบัญชีมากขึ้น)
    edited_df = st.data_editor(
        df,
        column_config={
            "action": st.column_config.SelectboxColumn("Action", options=["Approve", "Hold", "Reject"], required=True),
            "remarks": st.column_config.TextColumn("Remarks (เหตุผล)"),
            "subtotal": st.column_config.NumberColumn("Subtotal", format="%.2f"),
            "vat_amount": st.column_config.NumberColumn("VAT 7%", format="%.2f"),
            "total_amount": st.column_config.NumberColumn("Total", format="%.2f"),
            "verification_status": st.column_config.TextColumn("AI Status", disabled=True),
            "filename": st.column_config.TextColumn("Source", disabled=True),
        },
        hide_index=True,
        use_container_width=True
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Finalize Data"):
            approved_count = len(edited_df[edited_df['action'] == 'Approve'])
            st.success(f"บันทึกข้อมูลเรียบร้อย: รอจ่ายเงิน {approved_count} รายการ")
            
    with col2:
        # เตรียมข้อมูลสำหรับ Export เข้า ERP โดยเฉพาะ
        export_df = edited_df.copy()
        # กรองเอาเฉพาะข้อมูลที่พนักงานบัญชีตรวจแล้วไม่ Reject
        export_df = export_df[export_df['action'] != 'Reject']
        
        # ลบคอลัมน์ที่ไม่จำเป็นสำหรับโปรแกรมบัญชี
        export_df = export_df.drop(columns=['filename', 'verification_status'])
        
        # เปลี่ยนชื่อหัวคอลัมน์ให้เป็น Standard ERP Format
        export_df = export_df.rename(columns={
            "action": "Status",
            "invoice_date": "DocDate",
            "invoice_number": "InvoiceNo",
            "po_reference": "PONumber",
            "vendor_name": "VendorName",
            "tax_id": "VendorTaxID",
            "subtotal": "AmountBeforeTax",
            "vat_amount": "TaxAmount",
            "total_amount": "TotalAmount",
            "remarks": "Memo"
        })
        
        # สร้าง CSV แบบ utf-8-sig เพื่อให้เปิดใน Excel แล้วภาษาไทยไม่เพี้ยน
        csv = export_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 Export to ERP (CSV)",
            data=csv,
            file_name=f"AP_Import_{int(time.time())}.csv",
            mime="text/csv"
        )