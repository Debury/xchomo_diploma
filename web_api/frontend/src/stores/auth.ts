import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export interface AuthUser {
  username: string
}

interface LoginSuccessResponse {
  success: true
  token: string
  username?: string
  message?: string
}

interface LoginErrorResponse {
  success?: false
  detail?: string
  message?: string
}

type LoginResponse = LoginSuccessResponse | LoginErrorResponse

function readStoredToken(): string | null {
  const raw = localStorage.getItem('auth_token')
  if (!raw || raw === 'undefined' || raw === 'null') return null
  return raw
}

function readStoredUser(): AuthUser | null {
  const raw = localStorage.getItem('auth_user')
  if (!raw || raw === 'undefined' || raw === 'null') return null
  try {
    return JSON.parse(raw) as AuthUser
  } catch {
    return null
  }
}

export const useAuthStore = defineStore('auth', () => {
  const token = ref<string | null>(readStoredToken())
  const user = ref<AuthUser | null>(readStoredUser())
  const initialized = ref(false)

  const isAuthenticated = computed(() => Boolean(token.value))

  // Verify the cached token against the server BEFORE the router's first
  // navigation decision. Without this, a stale localStorage token flashes
  // the protected UI for ~one API round-trip while apiFetch catches a 401
  // and hard-navigates to /login — visible as a flicker of the dashboard.
  //
  // Resolves once initialization is done. Idempotent: subsequent calls
  // return the original promise.
  let _initPromise: Promise<void> | null = null

  function initialize(): Promise<void> {
    if (_initPromise) return _initPromise
    _initPromise = (async () => {
      if (!token.value) {
        initialized.value = true
        return
      }
      try {
        const resp = await fetch('/auth/verify', {
          method: 'GET',
          headers: { Authorization: `Bearer ${token.value}` },
        })
        const data = resp.ok ? await resp.json() : null
        if (!data || data.valid !== true) {
          // Stale or revoked token — clear and let the guard redirect.
          token.value = null
          user.value = null
          localStorage.removeItem('auth_token')
          localStorage.removeItem('auth_user')
        } else if (data.username && (!user.value || user.value.username !== data.username)) {
          user.value = { username: data.username }
          localStorage.setItem('auth_user', JSON.stringify(user.value))
        }
      } catch {
        // Network error — treat token as unverified rather than wiping it,
        // so a flaky network on reload doesn't log the user out. apiFetch
        // will clear it the next time the server actually rejects.
      } finally {
        initialized.value = true
      }
    })()
    return _initPromise
  }

  async function login(username: string, password: string): Promise<true> {
    let resp: Response
    try {
      resp = await fetch('/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      })
    } catch {
      throw new Error('Cannot reach the authentication service. Check your connection and try again.')
    }

    let data: LoginResponse | null = null
    try {
      data = (await resp.json()) as LoginResponse
    } catch {
      // fall through — handled below
    }

    if (!resp.ok) {
      const detail = data && ((data as LoginErrorResponse).detail || (data as LoginErrorResponse).message)
      if (resp.status === 503) {
        throw new Error(detail || 'Authentication is not configured on the server (contact an administrator).')
      }
      if (resp.status === 401) {
        throw new Error('Invalid username or password')
      }
      throw new Error(detail || `Login failed (status ${resp.status})`)
    }

    const success = data as LoginSuccessResponse | null
    if (!success || !success.success || !success.token) {
      const msg = (data && (data as LoginErrorResponse).message) || 'Invalid username or password'
      throw new Error(msg)
    }

    token.value = success.token
    user.value = { username: success.username || username }

    localStorage.setItem('auth_token', success.token)
    localStorage.setItem('auth_user', JSON.stringify(user.value))

    return true
  }

  function logout(): void {
    token.value = null
    user.value = null
    localStorage.removeItem('auth_token')
    localStorage.removeItem('auth_user')
  }

  return { token, user, initialized, isAuthenticated, initialize, login, logout }
})
