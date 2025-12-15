# Climate RAG Frontend

Vue 3 Single Page Application for the Climate Data ETL Pipeline.

## Features

- 🌙 Dark mode UI with TailwindCSS
- 📊 Dashboard with stats overview
- 💬 AI-powered chat with RAG
- 📁 Data source management
- 🔢 Embedding statistics
- 🔐 Simple authentication

## Tech Stack

- Vue 3 (Composition API)
- Vue Router 4
- Pinia (State Management)
- TailwindCSS
- Vite

## Development

```bash
# Install dependencies
npm install

# Start dev server (with hot reload)
npm run dev

# The dev server proxies API requests to http://localhost:8000
```

## Production Build

```bash
# Build for production
npm run build

# Files are output to ./dist/
# FastAPI serves these at /app route
```

## Docker Deployment

The frontend is built as part of the Docker image:

```dockerfile
# In web-api Dockerfile, add:
WORKDIR /app/web_api/frontend
RUN npm install && npm run build
```

## Authentication

Credentials are configured via environment variables:

```env
AUTH_USERNAME=admin
AUTH_PASSWORD=climate2024
```

## Routes

| Route | Description |
|-------|-------------|
| `/app` | Dashboard |
| `/app/chat` | RAG Chat Interface |
| `/app/sources` | Data Sources List |
| `/app/sources/create` | Create New Source |
| `/app/embeddings` | Embedding Stats |
| `/app/login` | Login Page |

## API Proxy (Development)

During development, Vite proxies these paths to the FastAPI backend:

- `/rag/*` → RAG endpoints
- `/embeddings/*` → Embedding endpoints
- `/sources/*` → Source management
- `/auth/*` → Authentication
- `/health` → Health check
