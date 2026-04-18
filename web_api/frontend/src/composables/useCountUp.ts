import { ref, watch, onMounted, type Ref } from 'vue'

type NumericInput = Ref<number | string | null | undefined>

export function useCountUp(targetRef: NumericInput, duration = 800) {
  const displayValue = ref(0)
  let animationId: number | null = null

  function easeOutCubic(t: number): number {
    return 1 - Math.pow(1 - t, 3)
  }

  function toNumber(val: unknown): number {
    if (typeof val === 'number') return val
    if (typeof val === 'string') {
      const n = parseInt(val, 10)
      return Number.isFinite(n) ? n : 0
    }
    return 0
  }

  function animate(from: number, to: number): void {
    if (animationId) cancelAnimationFrame(animationId)
    if (to === 0) {
      displayValue.value = 0
      return
    }

    const start = performance.now()

    function step(now: number): void {
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
    animate(toNumber(oldVal), toNumber(newVal))
  })

  onMounted(() => {
    const val = toNumber(targetRef.value)
    if (val > 0) animate(0, val)
  })

  return { displayValue }
}
