<template>
  <Teleport to="body">
    <Transition name="modal-fade">
      <div
        v-if="open"
        ref="backdropRef"
        class="fixed inset-0 bg-black/40 flex items-center justify-center z-50 p-4"
        @click.self="onBackdrop"
      >
        <div
          ref="dialogRef"
          role="dialog"
          aria-modal="true"
          :aria-labelledby="titleId"
          :aria-describedby="subtitle ? subtitleId : undefined"
          tabindex="-1"
          :class="[
            'bg-white border border-mendelu-gray-semi rounded-xl shadow-lg w-full mx-4 outline-none',
            widthClass,
            scrollable ? 'max-h-[85vh] flex flex-col' : '',
          ]"
          @keydown.esc.stop="emit('close')"
          @keydown.tab="onTab"
        >
          <div class="flex items-start justify-between gap-4 p-6 pb-4">
            <div class="flex-1 min-w-0">
              <h2 :id="titleId" class="text-lg font-semibold text-mendelu-black">
                <slot name="title">{{ title }}</slot>
              </h2>
              <p v-if="subtitle || $slots.subtitle" :id="subtitleId" class="text-xs text-mendelu-gray-dark mt-0.5">
                <slot name="subtitle">{{ subtitle }}</slot>
              </p>
            </div>
            <button
              type="button"
              class="btn-ghost !px-2 !py-1 flex-shrink-0"
              aria-label="Close dialog"
              @click="emit('close')"
            >&times;</button>
          </div>

          <div
            :class="[
              'px-6 pb-6',
              scrollable ? 'overflow-y-auto' : '',
            ]"
          >
            <slot />
          </div>

          <div v-if="$slots.footer" class="px-6 pb-6 pt-0 border-t border-mendelu-gray-semi">
            <slot name="footer" />
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<script setup lang="ts">
import { computed, nextTick, ref, watch, onBeforeUnmount } from 'vue'

const props = withDefaults(defineProps<{
  open: boolean
  title?: string
  subtitle?: string
  maxWidth?: 'sm' | 'md' | 'lg' | 'xl' | '2xl'
  scrollable?: boolean
  closeOnBackdrop?: boolean
}>(), {
  title: '',
  subtitle: '',
  maxWidth: 'md',
  scrollable: true,
  closeOnBackdrop: true,
})

const emit = defineEmits<{
  (e: 'close'): void
}>()

const uid = Math.random().toString(36).slice(2, 9)
const titleId = `modal-title-${uid}`
const subtitleId = `modal-subtitle-${uid}`

const backdropRef = ref<HTMLElement | null>(null)
const dialogRef = ref<HTMLElement | null>(null)
let previouslyFocused: HTMLElement | null = null
let previousBodyOverflow = ''

const widthClass = computed(() => {
  switch (props.maxWidth) {
    case 'sm': return 'max-w-sm'
    case 'md': return 'max-w-md'
    case 'lg': return 'max-w-lg'
    case 'xl': return 'max-w-xl'
    case '2xl': return 'max-w-2xl'
    default: return 'max-w-md'
  }
})

function onBackdrop() {
  if (props.closeOnBackdrop) emit('close')
}

function getFocusable(): HTMLElement[] {
  if (!dialogRef.value) return []
  const sel = 'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
  return Array.from(dialogRef.value.querySelectorAll<HTMLElement>(sel))
    .filter(el => !el.hasAttribute('disabled') && el.offsetParent !== null)
}

function onTab(e: KeyboardEvent) {
  const focusable = getFocusable()
  if (focusable.length === 0) {
    e.preventDefault()
    dialogRef.value?.focus()
    return
  }
  const first = focusable[0]
  const last = focusable[focusable.length - 1]
  const active = document.activeElement as HTMLElement | null
  if (e.shiftKey && active === first) {
    e.preventDefault()
    last.focus()
  } else if (!e.shiftKey && active === last) {
    e.preventDefault()
    first.focus()
  }
}

watch(() => props.open, async (isOpen) => {
  if (isOpen) {
    previouslyFocused = (document.activeElement as HTMLElement) ?? null
    previousBodyOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    await nextTick()
    const focusable = getFocusable()
    ;(focusable[0] ?? dialogRef.value)?.focus()
  } else {
    document.body.style.overflow = previousBodyOverflow
    previouslyFocused?.focus?.()
    previouslyFocused = null
  }
}, { immediate: true })

onBeforeUnmount(() => {
  if (props.open) {
    document.body.style.overflow = previousBodyOverflow
    previouslyFocused?.focus?.()
  }
})
</script>

<style scoped>
.modal-fade-enter-active,
.modal-fade-leave-active {
  transition: opacity 120ms ease;
}
.modal-fade-enter-from,
.modal-fade-leave-to {
  opacity: 0;
}
</style>
