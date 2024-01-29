import daisyui from "daisyui";

/** @type {import('tailwindcss').Config} */
export default {
  content: ["./src/**/*.{vue,ts,js}"],
  theme: {
    extend: {},
  },
  plugins: [require(daisyui)],
}

