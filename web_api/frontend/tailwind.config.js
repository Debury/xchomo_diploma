/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{vue,js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        display: ['DM Sans', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'SF Mono', 'Fira Code', 'monospace'],
      },
      colors: {
        mendelu: {
          green: '#79be15',
          'green-hover': '#6aaa10',
          'green-light': '#e8f5d0',
          'green-muted': 'rgba(121, 190, 21, 0.08)',
          success: '#82c55b',
          alert: '#d53e3c',
          black: '#1c1c1c',
          white: '#ffffff',
          'gray-dark': '#535a5d',
          'gray-semi': '#dce3e4',
          'gray-light': '#f7f8fa',
        },
        dark: {
          bg: '#0f172a',
          card: '#1e293b',
          border: '#334155',
          hover: '#475569',
        },
        surface: {
          DEFAULT: '#ffffff',
          secondary: '#f7f8fa',
          tertiary: '#f8fafb',
        },
        muted: {
          DEFAULT: '#535a5d',
          light: '#dce3e4',
        },
        ring: {
          DEFAULT: 'rgba(121, 190, 21, 0.3)',
        },
      },
      keyframes: {
        skeleton: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.4' },
        },
        fadeSlideUp: {
          from: { opacity: '0', transform: 'translateY(12px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
      },
      animation: {
        skeleton: 'skeleton 1.5s ease-in-out infinite',
        'fade-slide-up': 'fadeSlideUp 0.5s cubic-bezier(0.16, 1, 0.3, 1) both',
      },
      borderRadius: {
        '2xl': '1rem',
        '3xl': '1.25rem',
      },
    },
  },
  plugins: [],
}
