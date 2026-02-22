import { ref, watch, onMounted } from 'vue'

export function useCountUp(targetRef, duration = 800) {
  const displayValue = ref(0)
  let animationId = null

  function easeOutCubic(t) {
    return 1 - Math.pow(1 - t, 3)
  }

  function animate(from, to) {
    if (animationId) cancelAnimationFrame(animationId)
    if (to === 0) { displayValue.value = 0; return }

    const start = performance.now()

    function step(now) {
      const elapsed = now - start
      const progress = Math.min(elapsed / duration, 1)
      const eased = easeOutCubic(progress)
      displayValue.value = Math.round(from + (to - from) * eased)

      if (progress < 1) {
        animationId = requestAnimationFrame(step)
      } else {
        displayValue.value = to
      }
    }

    animationId = requestAnimationFrame(step)
  }

  watch(targetRef, (newVal, oldVal) => {
    const to = typeof newVal === 'number' ? newVal : parseInt(newVal) || 0
    const from = typeof oldVal === 'number' ? oldVal : parseInt(oldVal) || 0
    animate(from, to)
  })

  onMounted(() => {
    const val = typeof targetRef.value === 'number' ? targetRef.value : parseInt(targetRef.value) || 0
    if (val > 0) animate(0, val)
  })

  return { displayValue }
}
