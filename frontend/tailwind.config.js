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
      },
      animation: {
        shimmer: "shimmer 2s infinite",
        dash: "dash 3s linear infinite",
      },
    },
  },
  plugins: [],
};
