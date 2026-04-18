import { reactive } from 'vue'

export interface ConfirmOptions {
  title?: string
  message: string
  confirmText?: string
  cancelText?: string
  danger?: boolean
}

interface ConfirmState extends ConfirmOptions {
  open: boolean
  resolver: ((value: boolean) => void) | null
}

const state = reactive<ConfirmState>({
  open: false,
  title: '',
  message: '',
  confirmText: 'Confirm',
  cancelText: 'Cancel',
  danger: false,
  resolver: null,
})

function ask(opts: ConfirmOptions): Promise<boolean> {
  return new Promise((resolve) => {
    if (state.resolver) state.resolver(false)
    state.title = opts.title ?? 'Please confirm'
    state.message = opts.message
    state.confirmText = opts.confirmText ?? 'Confirm'
    state.cancelText = opts.cancelText ?? 'Cancel'
    state.danger = opts.danger ?? false
    state.resolver = resolve
    state.open = true
  })
}

function resolve(value: boolean): void {
  state.open = false
  const r = state.resolver
  state.resolver = null
  r?.(value)
}

export function useConfirm() {
  return {
    state,
    confirm: ask,
    accept: () => resolve(true),
    cancel: () => resolve(false),
  }
}
