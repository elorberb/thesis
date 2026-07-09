import { createContext, useContext, useState, useEffect, ReactNode } from "react";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { ColorScheme, ThemeColors, GradientsByTheme, MaturityColorsByTheme, MaturityStage } from "../constants/theme";

const STORAGE_KEY = "@loupelab_theme";

type ThemeContextValue = {
  scheme: ColorScheme;
  Colors: typeof ThemeColors.dark;
  Gradients: { vitality: [string, string] };
  toggleTheme: () => void;
  getMaturityColors: (stage: string) => { background: string; text: string; label: string };
};

const ThemeContext = createContext<ThemeContextValue | null>(null);

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [scheme, setScheme] = useState<ColorScheme>("dark");

  useEffect(() => {
    AsyncStorage.getItem(STORAGE_KEY)
      .then((saved) => { if (saved === "light" || saved === "dark") setScheme(saved); })
      .catch(() => {});
  }, []);

  const toggleTheme = () => {
    setScheme((prev) => {
      const next: ColorScheme = prev === "dark" ? "light" : "dark";
      AsyncStorage.setItem(STORAGE_KEY, next).catch(() => {});
      return next;
    });
  };

  const getMaturityColors = (stage: string) => {
    return MaturityColorsByTheme[scheme][stage as MaturityStage] ?? MaturityColorsByTheme[scheme].early;
  };

  return (
    <ThemeContext.Provider value={{
      scheme,
      Colors: ThemeColors[scheme],
      Gradients: GradientsByTheme[scheme],
      toggleTheme,
      getMaturityColors,
    }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme must be used inside ThemeProvider");
  return ctx;
}
