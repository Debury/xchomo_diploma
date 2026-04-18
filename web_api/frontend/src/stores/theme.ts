import { defineStore } from 'pinia'
import { ref, watch } from 'vue'

export type Theme = 'light' | 'dark'

function readTheme(): Theme {
  const raw = localStorage.getItem('theme')
  return raw === 'dark' ? 'dark' : 'light'
}

export const useThemeStore = defineStore('theme', () => {
  const theme = ref<Theme>(readTheme())

  function apply(t: Theme): void {
    document.documentElement.classList.toggle('dark', t === 'dark')
  }

  function toggle(): void {
    theme.value = theme.value === 'dark' ? 'light' : 'dark'
  }

  watch(theme, (val) => {
    localStorage.setItem('theme', val)
    apply(val)
  })

  // Apply on init
  apply(theme.value)

  return { theme, toggle }
})
