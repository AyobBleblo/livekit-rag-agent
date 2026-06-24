"""
build_rag.py — يقرأ ملفات JSON من rag_data/ ويبني قاعدة بيانات ChromaDB مع embeddings.
شغّل هذا الملف أولاً قبل تشغيل agent.py.
"""

import json
import os
import sys
from pathlib import Path
import chromadb
from google import genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("[ERROR] لم يتم العثور على GEMINI_API_KEY أو GOOGLE_API_KEY في ملف .env")
    sys.exit(1)

client_genai = genai.Client(api_key=api_key)

RAG_DIR = Path("rag_data")
CHROMA_DIR = Path("chroma_db")

# ---------- chunking functions ----------

def chunk_clinic_info(data: dict) -> list[dict]:
    chunks = []
    hours = "\n".join([f"{day}: {time}" for day, time in data["working_hours"].items()])
    chunks.append({
        "id": "clinic_main",
        "text": (
            f"اسم العيادة: {data['clinic_name']}\n"
            f"العنوان: {data['address']}\n"
            f"الهاتف: {data['phone']}\n"
            f"واتساب: {data['whatsapp']}\n"
            f"أوقات العمل:\n{hours}\n"
            f"عن العيادة: {data['about']}"
        )
    })
    return chunks


def chunk_doctors(doctors: list) -> list[dict]:
    chunks = []
    for doc in doctors:
        days = "، ".join(doc["schedule"]["days"])
        text = (
            f"الدكتور/ة: {doc['name']}\n"
            f"التخصص: {doc['specialty']} — {doc['sub_specialty']}\n"
            f"الخبرة: {doc['experience_years']} سنوات\n"
            f"أيام العمل: {days}\n"
            f"ساعات العمل: من {doc['schedule']['start_time']} إلى {doc['schedule']['end_time']}\n"
            f"سعر الاستشارة: {doc['consultation_price_LYD']} دينار ليبي\n"
            f"يستقبل مرضى جدد: {'نعم' if doc['accepts_new_patients'] else 'لا حالياً'}\n"
            f"نبذة: {doc['bio']}"
        )
        chunks.append({"id": doc["id"], "text": text})
    return chunks


def chunk_services(services: list) -> list[dict]:
    chunks = []
    for svc in services:
        text = (
            f"الخدمة: {svc['name']}\n"
            f"التصنيف: {svc['category']}\n"
            f"الوصف: {svc['description']}\n"
            f"المدة: {svc['duration_minutes']} دقيقة\n"
            f"السعر: {svc['price_LYD']} دينار ليبي\n"
            f"مناسبة لـ: {', '.join(svc['suitable_for'])}"
        )
        chunks.append({"id": svc["id"], "text": text})
    return chunks


def chunk_faq(faqs: list) -> list[dict]:
    return [
        {"id": f["id"], "text": f"سؤال: {f['question']}\nجواب: {f['answer']}"}
        for f in faqs
    ]


def chunk_booking_rules(data: dict) -> list[dict]:
    text = (
        f"طرق الحجز: {', '.join(data['booking_methods'])}\n"
        f"الحجز المسبق: قبل {data['advance_booking_hours']} ساعة على الأقل\n"
        f"إلغاء مجاني: قبل {data['cancellation_policy']['free_cancellation_before_hours']} ساعة\n"
        f"رسوم الإلغاء المتأخر: {data['cancellation_policy']['late_cancellation_fee_LYD']} دينار\n"
        f"طرق الدفع: {', '.join(data['payment_methods'])}\n"
        f"تعليمات المريض الجديد: {data['new_patient_instructions']}"
    )
    return [{"id": "booking_rules", "text": text}]


# ---------- embedding helper ----------

def get_embedding(text: str) -> list[float]:
    """Generate embedding using the new google-genai SDK."""
    response = client_genai.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
    )
    return response.embeddings[0].values


# ---------- main ----------

LOADERS = {
    "clinic_info.json":   (chunk_clinic_info,   False),
    "doctors.json":       (chunk_doctors,        True),
    "services.json":      (chunk_services,       True),
    "faq.json":           (chunk_faq,            True),
    "booking_rules.json": (chunk_booking_rules,  False),
}


def main():
    # Validate RAG data files exist
    missing = [f for f in LOADERS if not (RAG_DIR / f).exists()]
    if missing:
        print(f"[ERROR] ملفات ناقصة في {RAG_DIR}/: {', '.join(missing)}")
        sys.exit(1)

    # Build chunks from all JSON files
    all_chunks = []
    for filename, (chunker, is_list) in LOADERS.items():
        data = json.loads((RAG_DIR / filename).read_text(encoding="utf-8"))
        chunks = chunker(data)
        all_chunks.extend(chunks)
        print(f"  [+] {filename} -> {len(chunks)} chunks")

    print(f"\n[TOTAL] chunks: {len(all_chunks)}")

    # Create / reset ChromaDB collection
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # Delete existing collection to rebuild fresh
    try:
        chroma_client.delete_collection(name="clinic_knowledge")
        print("[RESET] deleted old collection")
    except Exception:
        pass  # Collection doesn't exist yet
    collection = chroma_client.create_collection(name="clinic_knowledge")

    # Embed and store each chunk
    for i, chunk in enumerate(all_chunks):
        print(f"  [{i+1}/{len(all_chunks)}] embedding: {chunk['id']}")
        embedding = get_embedding(chunk["text"])
        collection.add(
            ids=[chunk["id"]],
            documents=[chunk["text"]],
            embeddings=[embedding]
        )

    print(f"\n[DONE] ChromaDB built successfully in {CHROMA_DIR}/")
    print(f"   Total documents: {collection.count()}")


if __name__ == "__main__":
    main()