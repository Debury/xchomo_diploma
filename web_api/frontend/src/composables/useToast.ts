import { reactive } from 'vue'

export type ToastKind = 'error' | 'success' | 'info'

export interface Toast {
  id: number
  kind: ToastKind
  message: string
  retry?: () => void | Promise<void>
  timeoutId?: ReturnType<typeof setTimeout>
}

interface ToastState {
  items: Toast[]
}

const state = reactive<ToastState>({ items: [] })
let nextId = 1

function remove(id: number): void {
  const idx = state.items.findIndex(t => t.id === id)
  if (idx === -1) return
  const t = state.items[idx]
  if (t.timeoutId) clearTimeout(t.timeoutId)
  state.items.splice(idx, 1)
}

function push(kind: ToastKind, message: string, opts: { retry?: Toast['retry']; duration?: number } = {}): number {
  const id = nextId++
  const duration = opts.duration ?? (kind === 'error' ? 8000 : 4000)
  const toast: Toast = { id, kind, message, retry: opts.retry }
  if (duration > 0) {
    toast.timeoutId = setTimeout(() => remove(id), duration)
  }
  state.items.push(toast)
  return id
}

export function useToast() {
  return {
    toasts: state,
    remove,
    error: (message: string, retry?: Toast['retry']) => push('error', message, { retry }),
    success: (message: string) => push('success', message),
    info: (message: string) => push('info', message),
  }
}
