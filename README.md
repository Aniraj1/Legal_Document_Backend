# Backend

Django REST backend for the project.

## Requirements
- Python 3.12+
- Virtual environment (recommended)

## Setup
1. Create and activate a virtual environment:
   - Windows PowerShell:
     - `virtualenv venv`
     - `venv\Scripts\Activate.ps1`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Create a `.env` file in the project root (use `.env` keys referenced in `core/settings.py`).

## Database
- Create migrations:
  - `python manage.py makemigrations`
- Apply migrations:
  - `python manage.py migrate`

## Run the server
- `python manage.py runserver`
- Open: `http://127.0.0.1:8000/`

## Admin
1. Create a superuser:
   - `python manage.py createsuperuser`
2. Log in at:
   - `http://127.0.0.1:8000/admin/`

## API Docs
- Swagger UI: `http://127.0.0.1:8000/api/swagger/`
- Redoc: `http://127.0.0.1:8000/api/redoc/`
- Schema: `http://127.0.0.1:8000/api/schema/`
- admin: `http://127.0.0.1:8000/admin`
