# Legal Document Assistant Backend

The backend provides:
- JWT-based authentication and user profile APIs
- PDF upload and indexing to Upstash Vector
- RAG question answering with Groq (`ask-groq`)
- User analytics endpoints
- OpenAPI docs via drf-spectacular

## Project Structure

- `core/` Django project settings and URL routing
- `authuser/` auth and user profile APIs
- `fileUpload/` upload, vector indexing, RAG query, analytics APIs
- `globalutils/` shared response/pagination/error utilities

## Prerequisites

- Python 3.12+
- pip
- Virtual environment support
- Upstash Vector index credentials (for upload + retrieval)
- Groq API key (for answering document questions)

## Backend Setup (Windows PowerShell)

1. Create and activate virtual environment:

```powershell
python -m venv venv
./venv/Scripts/Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Create `.env` in the repository root (you can copy from `sample.env`):

```env
DEBUG=true
SECRET_KEY=your-secret-key
JWT_ACCESS_TOKEN_LIFETIME_HRS=2
JWT_REFRESH_TOKEN_LIFETIME_HRS=24
JWT_KEY=your-jwt-signing-key
PASSWORD_MIN_LENGTH=8
THROTTLE_RATES_IN_DAYS=5000
SUPER_ADMIN_PASSWORD=Admin@123
PROJECT_NAME=legal-doc

UPSTASH_VECTOR_REST_URL=https://your-vector-index.upstash.io
UPSTASH_VECTOR_REST_TOKEN=your-upstash-token

# Optional (analytics persistence)
UPSTASH_REDIS_REST_URL=
UPSTASH_REDIS_REST_TOKEN=

# Required for ask-groq
GROQ_API_KEY=your-groq-key
GROQ_MAX_TOKENS=1200

# Optional LangChain tuning
LANGCHAIN_EMBEDDING_MODEL=huggingface
LANGCHAIN_CHUNK_SIZE=1000
LANGCHAIN_CHUNK_OVERLAP=100
LANGCHAIN_EMBEDDING_DIMENSION=1024
```

4. Run migrations:

```powershell
python manage.py migrate
```

5. Start backend server:

```powershell
python manage.py runserver
```

Backend runs at `http://127.0.0.1:8000/`.

## API Documentation

- Swagger: `http://127.0.0.1:8000/api/swagger/`
- Redoc: `http://127.0.0.1:8000/api/redoc/`
- OpenAPI schema: `http://127.0.0.1:8000/api/schema/`
- Django admin: `http://127.0.0.1:8000/admin/`

## Authentication Flow

Most protected endpoints require:

```http
Authorization: Bearer <access_token>
```

User auth endpoints are mounted under:

- `/api/v1/<PROJECT_NAME>/...` for register/login/logout/refresh
- `/api/v1/...` for user detail/info/password endpoints

## API Endpoints (Current)

### Auth (`authuser`)

- `POST /api/v1/<PROJECT_NAME>/user/register/`
- `POST /api/v1/<PROJECT_NAME>/user/login/`
- `POST /api/v1/<PROJECT_NAME>/user/logout/`
- `POST /api/v1/<PROJECT_NAME>/refresh-token/`
- `PUT /api/v1/logged-in-user/change-password/`
- `POST /api/v1/user-detail/`
- `PUT /api/v1/user-detail/`
- `GET /api/v1/user-info/`

### File Upload + RAG (`fileUpload`)

- `POST /api/v1/upload/`
- `GET /api/v1/get-files/`
- `POST /api/v1/ask-groq/`
- `DELETE /api/v1/remove-file/<file_id>/`
- `DELETE /api/v1/remove-files/`
- `GET /api/v1/analytics/me/`
- `DELETE /api/v1/analytics/me/`

## Upload and RAG Notes

- Current upload endpoint enforces PDF uploads (`application/pdf`) in `UploadFileView`.
- Max upload size is validated via `fileUpload.services.file_validator`.
- Upload pipeline:
   1. validate file
   2. extract text with LangChain
   3. split into chunks
   4. create embeddings
   5. store in Upstash Vector
- `ask-groq` supports models:
   - `llama-3.1-8b-instant` (default)
   - `llama-3.3-70b-versatile`
- `chat_history` is accepted, validated, and capped to the latest 20 messages.

## Running Tests

Run all tests:

```powershell
pytest
```

Run specific test file:

```powershell
pytest fileUpload/tests/test_fileupload_api_endpoints.py
```

Run one test by keyword:

```powershell
pytest -k test_upload_file_with_valid_data
```

## Troubleshooting

### `500` on upload with `No chunks created from documents`

If you see logs similar to:

- `Document chunking failed: No chunks created from documents`
- `File upload failed: No chunks created from documents`

check the following:

1. The uploaded file is a real, non-empty PDF.
2. Text extraction succeeded (scanned/image-only PDFs may produce no extractable text).
3. Upstash env vars are configured correctly.
4. LangChain dependencies are installed from `requirements.txt`.

This error usually means extraction/chunking produced zero content, so the endpoint returns a server error.

### Hugging Face unauthenticated warning

The warning about unauthenticated Hugging Face Hub requests is non-fatal. You can set `HF_TOKEN` to increase rate limits and improve model download reliability.

## Useful Commands

Create admin user:

```powershell
python manage.py createsuperuser
```

Create new migrations:

```powershell
python manage.py makemigrations
```

Apply migrations:

```powershell
python manage.py migrate
```
