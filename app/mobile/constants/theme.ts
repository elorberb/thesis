export type ColorScheme = "dark" | "light";

const DarkColors = {
  background: "#111213",
  surface: "#1a1b1e",
  surfaceElevated: "#222428",
  surfaceHighest: "#2a2d32",
  border: "#34373d",
  borderSubtle: "rgba(60,65,75,0.20)",
  textPrimary: "#e2e5ea",
  textSecondary: "#9aa0ac",
  textMuted: "#646b74",
  accent: "#52c97a",
  accentDark: "#2d8c4e",
  accentSurface: "rgba(82,201,122,0.10)",
  accentText: "#082318",
  danger: "#f07068",
  dangerSurface: "rgba(170,35,25,0.15)",
  warning: "#e09a3c",
  warningSurface: "rgba(130,70,10,0.18)",
  tertiary: "#5ec4d8",
  tertiarySurface: "rgba(0,180,210,0.10)",
};

const LightColors = {
  background: "#f1f5f9",
  surface: "#ffffff",
  surfaceElevated: "#f8fafc",
  surfaceHighest: "#e8eef5",
  border: "#cbd5e1",
  borderSubtle: "rgba(148,163,184,0.20)",
  textPrimary: "#0f172a",
  textSecondary: "#475569",
  textMuted: "#94a3b8",
  accent: "#16a34a",
  accentDark: "#15803d",
  accentSurface: "rgba(22,163,74,0.10)",
  accentText: "#ffffff",
  danger: "#dc2626",
  dangerSurface: "#fecaca",
  warning: "#d97706",
  warningSurface: "rgba(217,119,6,0.10)",
  tertiary: "#0284c7",
  tertiarySurface: "rgba(2,132,199,0.10)",
};

export const Colors = DarkColors;

export const ThemeColors: Record<ColorScheme, typeof DarkColors> = {
  dark: DarkColors,
  light: LightColors,
};

export const Gradients = {
  vitality: ["#52c97a", "#2d8c4e"] as [string, string],
};

export const GradientsByTheme: Record<ColorScheme, { vitality: [string, string] }> = {
  dark: { vitality: ["#52c97a", "#2d8c4e"] },
  light: { vitality: ["#16a34a", "#15803d"] },
};

export type MaturityStage = "early" | "developing" | "peak" | "mature" | "late";

export const MaturityColors: Record<MaturityStage, { background: string; text: string; label: string }> = {
  early: { background: "#162a45", text: "#78b4f8", label: "Early" },
  developing: { background: "#23104e", text: "#b29af8", label: "Developing" },
  peak: { background: "#0d2a17", text: "#52c97a", label: "Peak" },
  mature: { background: "#331508", text: "#f5c04a", label: "Mature" },
  late: { background: "#35100c", text: "#f07d75", label: "Late" },
};

export const MaturityColorsByTheme: Record<ColorScheme, Record<MaturityStage, { background: string; text: string; label: string }>> = {
  dark: MaturityColors,
  light: {
    early: { background: "#dbeafe", text: "#1d4ed8", label: "Early" },
    developing: { background: "#ede9fe", text: "#6d28d9", label: "Developing" },
    peak: { background: "#dcfce7", text: "#15803d", label: "Peak" },
    mature: { background: "#fef3c7", text: "#b45309", label: "Mature" },
    late: { background: "#fee2e2", text: "#b91c1c", label: "Late" },
  },
};

export function getMaturityColors(stage: string): { background: string; text: string; label: string } {
  return MaturityColors[stage as MaturityStage] ?? MaturityColors.early;
}
