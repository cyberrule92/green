import js from "@eslint/js";
import globals from "globals";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";

export default [
  { ignores: ["dist"] },
  {
    // Build/tooling config runs in Node, not the browser.
    files: ["*.config.js"],
    languageOptions: {
      globals: globals.node,
      parserOptions: { ecmaVersion: "latest", sourceType: "module" },
    },
    rules: { ...js.configs.recommended.rules },
  },
  {
    files: ["src/**/*.{js,jsx}"],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
      parserOptions: {
        ecmaVersion: "latest",
        ecmaFeatures: { jsx: true },
        sourceType: "module",
      },
    },
    plugins: {
      "react-hooks": reactHooks,
      "react-refresh": reactRefresh,
    },
    rules: {
      ...js.configs.recommended.rules,
      ...reactHooks.configs.recommended.rules,
      // `eslint-plugin-react` is not a dependency here, so core no-unused-vars
      // cannot see that `<Section/>` uses the `Section` binding. Exempt
      // capitalised identifiers — by React convention those are components,
      // which are overwhelmingly consumed in JSX rather than called directly.
      "no-unused-vars": [
        "error",
        { varsIgnorePattern: "^[A-Z_]", argsIgnorePattern: "^[A-Z_]" },
      ],
      "react-refresh/only-export-components": [
        "warn",
        { allowConstantExport: true },
      ],
    },
  },
];
