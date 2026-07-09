import { View, Text, StyleSheet, ScrollView } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useRouter } from "expo-router";
import { useTheme } from "../contexts/ThemeContext";
import { ScreenHeader } from "../components/ScreenHeader";

const STEPS = [
  {
    number: "01",
    title: "Capture",
    description:
      "Take a close-up macro photo of your flower's trichomes. Hold your phone 2–3 cm away for sharp, usable detail.",
  },
  {
    number: "02",
    title: "Detect",
    description:
      "Our AI locates and identifies individual trichomes in the image, classifying each as clear, cloudy, or amber.",
  },
  {
    number: "03",
    title: "Analyze",
    description:
      "Trichome distribution ratios are measured alongside pistil (stigma) color to build a complete maturity profile.",
  },
  {
    number: "04",
    title: "Result",
    description:
      "You receive a maturity stage — Early, Developing, Peak, Mature, or Late — with a clear maturity estimate.",
  },
];

const TIPS = [
  "Use a clip-on macro lens or the built-in macro mode on newer iPhones for sharper trichome detail.",
  "Photograph under consistent, diffuse lighting. Avoid direct flash or strong shadows.",
  "Keep the phone steady. Even minor blur reduces detection accuracy significantly.",
  "Capture multiple images from different angles for a more representative result.",
];

const STAGES = [
  { label: "Early", color: "#60a5fa", desc: "Trichomes mostly clear. Early maturity." },
  { label: "Developing", color: "#a78bfa", desc: "Cloudy trichomes forming. Continue waiting." },
  { label: "Peak", color: "#4ade80", desc: "Mostly cloudy. Peak maturity." },
  { label: "Mature", color: "#fbbf24", desc: "Amber developing. More relaxing, sedative effect." },
  { label: "Late", color: "#f87171", desc: "Mostly amber. High CBN content, strong sedative." },
];

export default function HowItWorksScreen() {
  const router = useRouter();
  const { Colors } = useTheme();
  const styles = createStyles(Colors);

  return (
    <SafeAreaView style={styles.safe}>
      <ScreenHeader title="How It Works" onBack={() => router.back()} />

      <ScrollView
        contentContainerStyle={styles.scroll}
        showsVerticalScrollIndicator={false}
      >
        <Text style={styles.intro}>
          LoupeLab uses computer vision trained on trichome and pistil imagery
          to assess flower maturity in seconds.
        </Text>

        <Text style={styles.sectionTitle}>The Process</Text>
        {STEPS.map((step) => (
          <View key={step.number} style={styles.stepCard}>
            <View style={styles.stepNumberBox}>
              <Text style={styles.stepNumber}>{step.number}</Text>
            </View>
            <View style={styles.stepContent}>
              <Text style={styles.stepTitle}>{step.title}</Text>
              <Text style={styles.stepDescription}>{step.description}</Text>
            </View>
          </View>
        ))}

        <Text style={[styles.sectionTitle, styles.sectionTitleGap]}>
          Tips for Best Results
        </Text>
        {TIPS.map((tip, index) => (
          <View key={index} style={styles.tipRow}>
            <View style={styles.tipBullet} />
            <Text style={styles.tipText}>{tip}</Text>
          </View>
        ))}

        <Text style={[styles.sectionTitle, styles.sectionTitleGap]}>
          Maturity Stages
        </Text>
        <View style={styles.stagesCard}>
          {STAGES.map((stage, index) => (
            <View key={stage.label}>
              <View style={styles.stageRow}>
                <View
                  style={[styles.stageDot, { backgroundColor: stage.color }]}
                />
                <View style={styles.stageInfo}>
                  <Text style={[styles.stageLabel, { color: stage.color }]}>
                    {stage.label}
                  </Text>
                  <Text style={styles.stageDesc}>{stage.desc}</Text>
                </View>
              </View>
              {index < STAGES.length - 1 && (
                <View style={styles.stageDivider} />
              )}
            </View>
          ))}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

function createStyles(Colors: ReturnType<typeof useTheme>["Colors"]) { return StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  scroll: {
    paddingHorizontal: 20,
    paddingTop: 20,
    paddingBottom: 48,
  },
  intro: {
    fontSize: 14,
    color: Colors.textSecondary,
    lineHeight: 22,
    marginBottom: 28,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: "700",
    color: Colors.textPrimary,
    marginBottom: 12,
  },
  sectionTitleGap: {
    marginTop: 28,
  },
  stepCard: {
    flexDirection: "row",
    backgroundColor: Colors.surface,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: Colors.border,
    padding: 16,
    marginBottom: 10,
    gap: 16,
    alignItems: "flex-start",
  },
  stepNumberBox: {
    width: 36,
    height: 36,
    borderRadius: 10,
    backgroundColor: Colors.accentSurface,
    borderWidth: 1,
    borderColor: Colors.accentDark,
    alignItems: "center",
    justifyContent: "center",
    flexShrink: 0,
  },
  stepNumber: {
    fontSize: 11,
    fontWeight: "800",
    color: Colors.accent,
    letterSpacing: 0.5,
  },
  stepContent: {
    flex: 1,
    gap: 4,
  },
  stepTitle: {
    fontSize: 14,
    fontWeight: "700",
    color: Colors.textPrimary,
  },
  stepDescription: {
    fontSize: 13,
    color: Colors.textSecondary,
    lineHeight: 20,
  },
  tipRow: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: 12,
    backgroundColor: Colors.surface,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: Colors.border,
    padding: 14,
    marginBottom: 8,
  },
  tipBullet: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: Colors.accent,
    marginTop: 7,
    flexShrink: 0,
  },
  tipText: {
    flex: 1,
    fontSize: 13,
    color: Colors.textSecondary,
    lineHeight: 20,
  },
  stagesCard: {
    backgroundColor: Colors.surface,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: Colors.border,
    overflow: "hidden",
  },
  stageRow: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: 14,
    padding: 16,
  },
  stageDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    marginTop: 3,
    flexShrink: 0,
  },
  stageInfo: {
    flex: 1,
  },
  stageLabel: {
    fontSize: 13,
    fontWeight: "700",
    marginBottom: 2,
  },
  stageDesc: {
    fontSize: 12,
    color: Colors.textMuted,
    lineHeight: 18,
  },
  stageDivider: {
    height: 1,
    backgroundColor: Colors.border,
    marginLeft: 40,
  },
}); }
