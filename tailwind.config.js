import daisyui from "daisyui";

/** @type {import('tailwindcss').Config} */
export default {
    content: ["./src/**/*.{vue,ts,js}"],
    theme: {
        extend: {},
    }, daisyui: {themes: ["night"]},
    plugins: [daisyui],
}

