/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      keyframes: {
        shimmer: {
          "100%": { transform: "translateX(100%)" },
        },
        dash: {
          to: { strokeDashoffset: "-8" },
        },
        scan: {
          "0%, 100%": { transform: "translateY(0)" },
          "50%": { transform: "translateY(100%)" },
        },
      },
      animation: {
        shimmer: "shimmer 2s infinite",
        dash: "dash 3s linear infinite",
        scan: "scan 3s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
