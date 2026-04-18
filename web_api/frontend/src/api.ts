// Thin wrapper around `fetch` that:
//   1. Injects the bearer token from localStorage on every request.
//   2. Redirects to /app/login when the backend returns 401 (token expired
//      or server restart cleared the in-memory token store).
//
// Usage: `import { apiFetch } from '../api'` then call `apiFetch('/sources/')`
// exactly like you would `fetch`. It returns the raw `Response` object.

export type ApiInput = RequestInfo | URL
export type ApiInit = RequestInit

function readToken(): string | null {
  const raw = localStorage.getItem('auth_token')
  if (!raw || raw === 'undefined' || raw === 'null') return null
  return raw
}

function redirectToLogin(): void {
  if (typeof window === 'undefined') return
  // Avoid an infinite loop if we're already there.
  if (window.location.pathname.startsWith('/app/login')) return
  // Clear stale credentials so the guard kicks in on the next load.
  localStorage.removeItem('auth_token')
  localStorage.removeItem('auth_user')
  window.location.assign('/app/login')
}

function inputToString(input: ApiInput): string {
  if (typeof input === 'string') return input
  if (input instanceof URL) return input.toString()
  return input.url
}

export async function apiFetch(input: ApiInput, init: ApiInit = {}): Promise<Response> {
  const token = readToken()
  const headers = new Headers(init.headers || {})
  if (token && !headers.has('Authorization')) {
    headers.set('Authorization', `Bearer ${token}`)
  }
  const resp = await fetch(input, { ...init, headers })
  if (resp.status === 401) {
    // Don't treat /auth/* responses as needing redirect — the login view has
    // its own error handling.
    const url = inputToString(input)
    if (!url.startsWith('/auth/')) {
      redirectToLogin()
    }
  }
  return resp
}

// Convenience wrappers — small surface area, keeps view code tidy.
export const apiGet = (url: ApiInput, init: ApiInit = {}): Promise<Response> =>
  apiFetch(url, { ...init, method: 'GET' })

export const apiPost = (url: ApiInput, body?: unknown, init: ApiInit = {}): Promise<Response> =>
  apiFetch(url, {
    ...init,
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...(init.headers || {}) },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  })

export const apiPut = (url: ApiInput, body?: unknown, init: ApiInit = {}): Promise<Response> =>
  apiFetch(url, {
    ...init,
    method: 'PUT',
    headers: { 'Content-Type': 'application/json', ...(init.headers || {}) },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  })

export const apiDelete = (url: ApiInput, init: ApiInit = {}): Promise<Response> =>
  apiFetch(url, { ...init, method: 'DELETE' })
