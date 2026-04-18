<template>
  <Teleport to="body">
    <div
      class="fixed top-4 right-4 z-[60] flex flex-col gap-2 max-w-sm w-[calc(100%-2rem)] sm:w-auto pointer-events-none"
      aria-live="polite"
      aria-atomic="false"
      role="region"
      aria-label="Notifications"
    >
      <TransitionGroup name="toast">
        <div
          v-for="t in toasts.items"
          :key="t.id"
          :class="[
            'pointer-events-auto flex items-start gap-3 rounded-lg border shadow-sm p-3 text-sm bg-white',
            toneBorder(t.kind),
          ]"
          role="status"
        >
          <span :class="['mt-0.5 w-2 h-2 rounded-full flex-shrink-0', toneDot(t.kind)]" aria-hidden="true"></span>
          <div class="flex-1 min-w-0">
            <p class="text-mendelu-black break-words">{{ t.message }}</p>
            <button
              v-if="t.retry"
              type="button"
              class="mt-1 text-xs font-medium text-mendelu-green hover:text-mendelu-green-hover"
              @click="handleRetry(t)"
            >
              Retry
            </button>
          </div>
          <button
            type="button"
            class="text-mendelu-gray-dark hover:text-mendelu-black text-lg leading-none flex-shrink-0 -mt-0.5"
            aria-label="Dismiss notification"
            @click="remove(t.id)"
          >&times;</button>
        </div>
      </TransitionGroup>
    </div>
  </Teleport>
</template>

<script setup lang="ts">
import { useToast, type Toast, type ToastKind } from '../composables/useToast'

const { toasts, remove } = useToast()

function toneBorder(kind: ToastKind): string {
  if (kind === 'error') return 'border-mendelu-alert/40'
  if (kind === 'success') return 'border-mendelu-success/40'
  return 'border-mendelu-gray-semi'
}
function toneDot(kind: ToastKind): string {
  if (kind === 'error') return 'bg-mendelu-alert'
  if (kind === 'success') return 'bg-mendelu-success'
  return 'bg-mendelu-green'
}

async function handleRetry(t: Toast) {
  try {
    await t.retry?.()
  } finally {
    remove(t.id)
  }
}
</script>

<style scoped>
.toast-enter-active,
.toast-leave-active {
  transition: all 150ms ease;
}
.toast-enter-from {
  opacity: 0;
  transform: translateY(-6px);
}
.toast-leave-to {
  opacity: 0;
  transform: translateX(8px);
}
</style>
