import { defineStore } from 'pinia'
import { ref, watch } from 'vue'

export const useThemeStore = defineStore('theme', () => {
  const theme = ref(localStorage.getItem('theme') || 'light')

  function apply(t) {
    document.documentElement.classList.toggle('dark', t === 'dark')
  }

  function toggle() {
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
