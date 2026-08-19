# RAG Tourist Assistant

A local LM-driven tourist assistant using SQLite + SQL retrieval + T5 encoder
semantic ranking + T5 encoder-decoder generation + LangChain + Streamlit.

## Architecture

User NLP query
 -> intent extraction
 -> SQL retrieval from hotels/restaurants/places
 -> T5 encoder embeddings for semantic reranking
 -> header + retrieved records + query
 -> T5 encoder
 -> T5 decoder through cross-attention
 -> grounded natural-language recommendation

T5 is an encoder-decoder model. The normal generation path is to provide the
combined prompt to the encoder and let the decoder attend to the encoder hidden
states while generating the answer.

## Setup

Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/build_database.py
streamlit run app.py
```

Linux/macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_database.py
streamlit run app.py
```

The first model load downloads `google-t5/t5-small` from Hugging Face.

Example queries:
- I have a budget of 800 SAR per night in city. Recommend a highly rated hotel.
- Recommend cheap Saudi restaurants in city under 100 SAR.
- What are the best cultural places to visit in city?
- I want a 2-day city trip with a hotel, restaurants, and attractions.

For a stronger portfolio/research version, replace the demo CSVs with a larger
dataset and fine-tune T5 on `(user query + retrieved context) -> answer`.
