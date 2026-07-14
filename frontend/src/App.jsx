import React from "react";
import { Grommet } from "grommet";
import { GreenAIChat } from "./Components/GreenAIChat";
import "./App.css";

const theme = {
  global: {
    colors: {
      brand: "#01a982",
      background: "#f4f8f7",
      "text-strong": "#0b1f1f",
      "text-soft": "#5f7272",
      border: "#d4e3df",
      accent: "#0f5f59",
    },
    font: {
      family: '"MetricHPE", "Aptos", "Trebuchet MS", sans-serif',
      size: "16px",
      height: "22px",
    },
  },
};

function App() {
  return (
    <Grommet full theme={theme}>
      <GreenAIChat />
    </Grommet>
  );
}

export default App;
