/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{vue,js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        mendelu: {
          green: '#79be15',
          'green-hover': '#6aaa10',
          success: '#82c55b',
          alert: '#d53e3c',
          black: '#1c1c1c',
          white: '#ffffff',
          'gray-dark': '#535a5d',
          'gray-semi': '#dce3e4',
          'gray-light': '#eff3f4',
        },
        dark: {
          bg: '#0f172a',
          card: '#1e293b',
          border: '#334155',
          hover: '#475569'
        }
      }
    },
  },
  plugins: [],
}
