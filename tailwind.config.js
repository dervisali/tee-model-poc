/** Tailwind v3 config for the FastAPI/HTMX front-end (web/).
 *  Build:  npx --yes tailwindcss@3 -c tailwind.config.js \
 *            -i web/static/tailwind.src.css -o web/static/tailwind.css --minify
 */
module.exports = {
  content: ["./web/templates/**/*.html"],
  // Classes assembled in JS strings (kept even if the scanner misses a variant).
  safelist: [
    "!bg-emerald-500", "!bg-rose-500", "!text-white",
    "ring-2", "ring-emerald-300", "ring-rose-300", "cursor-default",
    "border-emerald-400", "border-rose-300", "border-emerald-200", "border-rose-200",
    "bg-emerald-50", "bg-rose-50", "bg-emerald-100", "bg-rose-100",
    "text-emerald-700", "text-emerald-800", "text-rose-600", "text-rose-700", "text-rose-800",
  ],
  theme: {
    extend: {
      colors: {
        brand: { 50:"#eef3fb",100:"#d6e2f5",200:"#adc4ea",300:"#7e9fdb",
                 400:"#5179c6",500:"#345aa8",600:"#274785",700:"#21396a",
                 800:"#1b2f57",900:"#162b4e",950:"#0f1b34" },
        accent: { 500:"#c8102e", 600:"#a60d26" },
      },
      fontFamily: { sans: ["Inter","ui-sans-serif","system-ui","sans-serif"] },
      boxShadow: {
        card: "0 1px 2px rgba(16,24,40,.06),0 1px 3px rgba(16,24,40,.10)",
        lift: "0 8px 24px -8px rgba(16,24,40,.18)",
      },
    },
  },
  plugins: [],
};
