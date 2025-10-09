import io
from typing import List, Optional


import streamlit as st
from supabase import Client, create_client


BUCKET_NAME = st.secrets.get("SUPABASE_BUCKET", "butterfly")
TABLE_NAME = st.secrets.get("SUPABASE_TABLE", "butterflyentry")


def record_logo_entry(client: Client, email: str, class_label: str, image_name: str ):
    payload = {
        "email": email,
        "class_label": class_label,
        "image_name": image_name,
    }
    try:
        response = client.table(TABLE_NAME).insert(payload).execute()
    except Exception as exc:
        st.error(f"Storing image metadata failed: {exc}")
        return False

    error = getattr(response, "error", None)
    if error:
        message = getattr(error, "message", str(error))
        st.error(f"Storing image metadata failed: {message}")
        return False

    return None


def upload_file(client: Client, uploaded_file, class_label: str, email: str) -> Optional[str]:
    image_name = uploaded_file.name
    storage_path = f"{class_label}/{image_name}"
    file_bytes = uploaded_file.getvalue()
    try:
        client.storage.from_(BUCKET_NAME).upload(path=storage_path,file=file_bytes)
        record_logo_entry(client, email, class_label, image_name)
        return st.success(f"Uploaded to {storage_path}") 
    except Exception as exc:
        st.error(f"Upload failed: {exc}")
        return None
