import { useState, useCallback } from "react";
import { View, Text, Pressable, StyleSheet, ScrollView, Image } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useRouter, useLocalSearchParams } from "expo-router";
import { useFocusEffect } from "@react-navigation/native";
import { Ionicons } from "@expo/vector-icons";
import { LinearGradient } from "expo-linear-gradient";
import { useTheme } from "../contexts/ThemeContext";
import { DonutChart } from "../components/DonutChart";
import { ZoomableImage } from "../components/ZoomableImage";
import { AnalyzeResponse, TrichomeType, MaturityStage } from "../api/types";
import { AnalysisResultStore } from "../store/analysisResult";
import { ApiClient } from "../api/client";

const MATURITY_PREDICTION: Record<string, { heading: string; subtitle: string }> = {
  early: {
    heading: "3–4 weeks to peak",
    subtitle: "Plant is in early flowering. Significant development ahead.",
  },
  developing: {
    heading: "1–2 weeks to peak",
    subtitle: "Trichomes are developing. Continue monitoring daily.",
  },
  peak: {
    heading: "Harvest now",
    subtitle: "Optimal THC concentration reached. Act within 24–48 hours.",
  },
  mature: {
    heading: "1–3 days from peak",
    subtitle: "Review each section below to validate the current harvest window.",
  },
  late: {
    heading: "Past peak",
    subtitle: "Trichome degradation in progress. Harvest immediately.",
  },
};

const DOMINANT_PHASE: Record<TrichomeType, { title: string; desc: string }> = {
  clear: {
    title: "Clear trichomes",
    desc: "Most heads currently cluster in the clear stage.",
  },
  cloudy: {
    title: "Milky trichomes",
    desc: "Peak THC production in progress. Harvest window approaching.",
  },
  amber: {
    title: "Amber trichomes",
    desc: "Degradation in progress. Harvest soon for sedative effect.",
  },
};

const TRICHOME_COLORS: Record<TrichomeType, string> = {
  clear: "#6f758b",
  cloudy: "#dfe4fe",
  amber: "#d97706",
};

const STIGMA_COLORS = {
  green: "#52c97a",
  orange: "#d97706",
};

export const MOCK_RESULT: AnalyzeResponse = {
  id: "mock-id",
  created_at: "2026-03-24T12:00:00Z",
  device_id: "mock-device",
  image_url: "",
  annotated_image_url: null,
  maturity_stage: "mature",
  recommendation:
    "Trichome analysis indicates optimal THC concentration. 73% of glandular heads have reached a milky/cloudy state with 15% turning amber. Harvest within the next 48 hours for maximum potency.",
  trichome_result: {
    detections: [],
    distribution: { clear: 15, cloudy: 36, amber: 8 },
    total_count: 36,
  },
  stigma_result: {
    detections: [],
    avg_green_ratio: 0.53,
    avg_orange_ratio: 0.47,
    total_count: 6,
  },
  trichome_crops_b64: null,
  stigma_crops_b64: null,
};

function getDominantType(distribution: Record<TrichomeType, number>): TrichomeType {
  const entries = Object.entries(distribution) as [TrichomeType, number][];
  return entries.sort(([, a], [, b]) => b - a)[0][0];
}

function pct(value: number, total: number): number {
  return total === 0 ? 0 : Math.round((value / total) * 100);
}

type SessionSummary = {
  trichome_distribution: Record<TrichomeType, number>;
  total_trichome_count: number;
  maturity_stage: MaturityStage;
  avg_green_ratio: number;
  avg_orange_ratio: number;
  recommendation: string;
};

function computeSessionSummary(session: AnalyzeResponse[]): SessionSummary {
  const totalDist = { clear: 0, cloudy: 0, amber: 0 };
  let totalTrichomeCount = 0;
  let totalGreen = 0;
  let totalOrange = 0;

  session.forEach((r) => {
    totalDist.clear += r.trichome_result.distribution.clear;
    totalDist.cloudy += r.trichome_result.distribution.cloudy;
    totalDist.amber += r.trichome_result.distribution.amber;
    totalTrichomeCount += r.trichome_result.total_count;
    totalGreen += r.stigma_result.avg_green_ratio;
    totalOrange += r.stigma_result.avg_orange_ratio;
  });

  const stageCounts: Partial<Record<MaturityStage, number>> = {};
  session.forEach((r) => {
    stageCounts[r.maturity_stage] = (stageCounts[r.maturity_stage] ?? 0) + 1;
  });
  const dominantStage = (Object.entries(stageCounts) as [MaturityStage, number][]).sort(
    ([, a], [, b]) => b - a
  )[0][0];

  const cloudyPct = pct(totalDist.cloudy, totalTrichomeCount);
  const amberPct = pct(totalDist.amber, totalTrichomeCount);
  const prediction = MATURITY_PREDICTION[dominantStage] ?? MATURITY_PREDICTION.peak;
  const recommendation = `Combined analysis of ${session.length} captures: ${cloudyPct}% cloudy and ${amberPct}% amber trichomes on average. ${prediction.subtitle}`;

  return {
    trichome_distribution: totalDist,
    total_trichome_count: totalTrichomeCount,
    maturity_stage: dominantStage,
    avg_green_ratio: totalGreen / session.length,
    avg_orange_ratio: totalOrange / session.length,
    recommendation,
  };
}

function SingleCaptureDetail({
  result,
  title,
  onBack,
  showSaveButton,
  router,
  Colors,
  Gradients,
  getMaturityColors,
  styles,
}: {
  result: AnalyzeResponse;
  title: string;
  onBack: () => void;
  showSaveButton: boolean;
  router: ReturnType<typeof useRouter>;
  Colors: ReturnType<typeof useTheme>["Colors"];
  Gradients: ReturnType<typeof useTheme>["Gradients"];
  getMaturityColors: ReturnType<typeof useTheme>["getMaturityColors"];
  styles: ReturnType<typeof createStyles>;
}) {
  const stage = result.maturity_stage;
  const stageColors = getMaturityColors(stage);
  const prediction = MATURITY_PREDICTION[stage] ?? MATURITY_PREDICTION.peak;
  const totalTrichomes = result.trichome_result.total_count;
  const dist = result.trichome_result.distribution;
  const clearPct = pct(dist.clear, totalTrichomes);
  const cloudyPct = pct(dist.cloudy, totalTrichomes);
  const amberPct = pct(dist.amber, totalTrichomes);
  const dominantType = getDominantType(dist);
  const dominantPhase = DOMINANT_PHASE[dominantType];
  const dominantPct = pct(dist[dominantType], totalTrichomes);
  const greenPct = Math.round(result.stigma_result.avg_green_ratio * 100);
  const orangePct = Math.round(result.stigma_result.avg_orange_ratio * 100);
  const dominantStigma = greenPct >= orangePct ? "green" : "orange";

  const trichomeTypes: { type: TrichomeType; label: string; value: number }[] = [
    { type: "clear", label: "Clear", value: clearPct },
    { type: "cloudy", label: "Cloudy", value: cloudyPct },
    { type: "amber", label: "Amber", value: amberPct },
  ];

  return (
    <>
      <View style={styles.header}>
        <Pressable style={styles.backButton} onPress={onBack}>
          <Ionicons name="arrow-back" size={20} color={Colors.textPrimary} />
          <Text style={styles.backLabel}>Back</Text>
        </Pressable>
        <Text style={styles.headerTitle}>{title}</Text>
        <Pressable style={styles.headerAction}>
          <Ionicons name="share-outline" size={20} color={Colors.textSecondary} />
        </Pressable>
      </View>

      <ScrollView contentContainerStyle={styles.scroll} showsVerticalScrollIndicator={false}>
        <View style={styles.heroCard}>
          <View style={styles.heroBadge}>
            <Ionicons name="sparkles" size={12} color={Colors.accent} />
            <Text style={styles.heroBadgeText}>Detection complete</Text>
          </View>
          <Text style={styles.heroTitle}>Fresh analysis</Text>
          <Text style={styles.heroSubtitle}>
            Built from 1 capture with trichome and stigma scoring.
          </Text>
        </View>

        <View style={[styles.predictionCard, { backgroundColor: stageColors.background }]}>
          <View style={styles.predictionTop}>
            <Text style={styles.predictionLabel}>MATURITY PREDICTION</Text>
            <View style={styles.stigmaBadge}>
              <Text style={styles.stigmaBadgeText}>{result.stigma_result.total_count} stigmas</Text>
            </View>
          </View>
          <Text style={[styles.predictionHeading, { color: stageColors.text }]}>
            {prediction.heading}
          </Text>
          <Text style={styles.predictionSubtitle}>{prediction.subtitle}</Text>
        </View>

        <View style={styles.statsRow}>
          <View style={styles.statCard}>
            <Text style={styles.statValue}>{cloudyPct}%</Text>
            <Text style={styles.statLabel}>Cloudy trichomes</Text>
          </View>
          <View style={styles.statCard}>
            <Text style={styles.statValue}>{orangePct}%</Text>
            <Text style={styles.statLabel}>Orange stigmas</Text>
          </View>
        </View>

        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>Trichome profile</Text>
          <Text style={styles.sectionSubtitle}>Tap any card to inspect detections</Text>
        </View>

        <View style={styles.donutRow}>
          <DonutChart
            size={170}
            strokeWidth={22}
            centerLabel={`${dominantPct}%`}
            centerSublabel={dominantType.charAt(0).toUpperCase() + dominantType.slice(1)}
            segments={[
              { value: clearPct, color: TRICHOME_COLORS.clear },
              { value: cloudyPct, color: TRICHOME_COLORS.cloudy },
              { value: amberPct, color: TRICHOME_COLORS.amber },
            ]}
          />
          <View style={styles.donutLegend}>
            {trichomeTypes.map((t) => (
              <View key={t.type} style={styles.legendRow}>
                <View style={[styles.legendDot, { backgroundColor: TRICHOME_COLORS[t.type] }]} />
                <Text style={styles.legendLabel}>{t.label}</Text>
                <Text style={[styles.legendPct, { color: TRICHOME_COLORS[t.type] }]}>{t.value}%</Text>
              </View>
            ))}
          </View>
        </View>

        <View style={styles.dominantCard}>
          <Text style={styles.dominantLabel}>DOMINANT PHASE</Text>
          <Text style={styles.dominantTitle}>{dominantPhase.title}</Text>
          <Text style={styles.dominantDesc}>{dominantPhase.desc}</Text>
        </View>

        {trichomeTypes.map((t) => (
          <Pressable
            key={t.type}
            style={styles.typeCard}
            onPress={() => router.push(`/trichome-samples?type=${t.type}` as never)}
          >
            <View style={styles.typeCardTop}>
              <View style={styles.typeCardLeft}>
                <View style={[styles.typeDot, { backgroundColor: TRICHOME_COLORS[t.type] }]} />
                <Text style={styles.typeLabel}>{t.label}</Text>
              </View>
              <Text style={[styles.typePct, { color: TRICHOME_COLORS[t.type] }]}>{t.value}%</Text>
            </View>
            <View style={styles.typeBarTrack}>
              <View
                style={[
                  styles.typeBarFill,
                  { width: `${t.value}%`, backgroundColor: TRICHOME_COLORS[t.type] },
                ]}
              />
            </View>
            <View style={styles.typeCardBottom}>
              <Text style={styles.typeCount}>{dist[t.type]} detections</Text>
              <View style={styles.openSamplesLink}>
                <Text style={styles.openSamplesText}>OPEN SAMPLES</Text>
                <Ionicons name="chevron-forward" size={13} color={Colors.accent} />
              </View>
            </View>
          </Pressable>
        ))}

        <View style={[styles.sectionHeader, { marginTop: 24 }]}>
          <Text style={styles.sectionTitle}>Stigma color read</Text>
          <Text style={styles.sectionSubtitle}>Open the gallery for individual stigma crops</Text>
        </View>

        <View style={styles.stigmaDistHeader}>
          <Text style={styles.stigmaDistTitle}>Stigma distribution</Text>
          <View style={[styles.stigmaLeadBadge, { backgroundColor: dominantStigma === "green" ? Colors.accentSurface : "rgba(217,119,6,0.15)" }]}>
            <Text style={[styles.stigmaLeadText, { color: STIGMA_COLORS[dominantStigma] }]}>
              {dominantStigma === "green" ? greenPct : orangePct}% lead
            </Text>
          </View>
        </View>

        <View style={styles.splitBarContainer}>
          <View style={[styles.splitBarGreen, { flex: greenPct }]}>
            <Text style={styles.splitBarLabel}>{greenPct}%</Text>
            <Text style={styles.splitBarSublabel}>Green</Text>
          </View>
          <View style={[styles.splitBarOrange, { flex: orangePct }]}>
            <Text style={styles.splitBarLabel}>{orangePct}%</Text>
            <Text style={styles.splitBarSublabel}>Orange</Text>
          </View>
        </View>

        <View style={styles.stigmaCardsRow}>
          <Pressable
            style={[styles.stigmaTypeCard, { borderColor: "rgba(107,255,143,0.2)" }]}
            onPress={() => router.push("/stigma-samples?type=green" as never)}
          >
            <View style={styles.stigmaCardTop}>
              <View style={[styles.stigmaColorDot, { backgroundColor: Colors.accent }]} />
              <Text style={styles.stigmaTypeName}>Green</Text>
            </View>
            <Text style={styles.stigmaPct}>{greenPct}%</Text>
            <Text style={styles.stigmaTypeDesc}>Early-stage stigmas</Text>
            <View style={styles.openSamplesLink}>
              <Text style={styles.openSamplesText}>OPEN SAMPLES</Text>
              <Ionicons name="chevron-forward" size={13} color={Colors.accent} />
            </View>
          </Pressable>

          <Pressable
            style={[styles.stigmaTypeCard, { borderColor: "rgba(217,119,6,0.2)" }]}
            onPress={() => router.push("/stigma-samples?type=orange" as never)}
          >
            <View style={styles.stigmaCardTop}>
              <View style={[styles.stigmaColorDot, { backgroundColor: "#d97706" }]} />
              <Text style={styles.stigmaTypeName}>Orange</Text>
            </View>
            <Text style={[styles.stigmaPct, { color: "#d97706" }]}>{orangePct}%</Text>
            <Text style={styles.stigmaTypeDesc}>Ripening stigmas</Text>
            <View style={styles.openSamplesLink}>
              <Text style={[styles.openSamplesText, { color: "#d97706" }]}>OPEN SAMPLES</Text>
              <Ionicons name="chevron-forward" size={13} color="#d97706" />
            </View>
          </Pressable>
        </View>

        <View style={[styles.sectionHeader, { marginTop: 24 }]}>
          <Text style={styles.sectionTitle}>Source captures</Text>
          <Text style={styles.sectionSubtitle}>1 review frame</Text>
        </View>

        {result.annotated_image_url ? (
          <View style={styles.captureCard}>
            <ZoomableImage
              uri={result.annotated_image_url}
              style={styles.captureImage}
              imageStyle={styles.captureImage}
              resizeMode="cover"
            />
            <View style={styles.captureLabel}>
              <Text style={styles.captureLabelText}>Capture 1</Text>
              <Ionicons name="arrow-forward" size={14} color={Colors.textSecondary} />
            </View>
          </View>
        ) : (
          <View style={styles.capturePlaceholder}>
            <Ionicons name="image-outline" size={28} color={Colors.textMuted} />
            <Text style={styles.capturePlaceholderText}>Capture 1</Text>
            <Ionicons name="arrow-forward" size={14} color={Colors.textMuted} />
          </View>
        )}

        <View style={styles.recommendationCard}>
          <View style={styles.recRow}>
            <Ionicons name="bulb-outline" size={16} color={Colors.accent} />
            <Text style={styles.recLabel}>HARVEST RECOMMENDATION</Text>
          </View>
          <Text style={styles.recText}>{result.recommendation}</Text>
        </View>

        {showSaveButton && (
          <Pressable onPress={() => router.push("/save-flower" as never)}>
            {({ pressed }) => (
              <LinearGradient
                colors={Gradients.vitality}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
                style={[styles.saveButton, pressed && { opacity: 0.88 }]}
              >
                <Ionicons name="bookmark-outline" size={18} color={Colors.accentText} />
                <Text style={styles.saveButtonText}>Save flower</Text>
              </LinearGradient>
            )}
          </Pressable>
        )}

        <Pressable style={styles.addMoreButton} onPress={() => router.push("/camera" as never)}>
          <Ionicons name="add" size={18} color={Colors.textPrimary} />
          <Text style={styles.addMoreText}>Add more images</Text>
        </Pressable>
      </ScrollView>
    </>
  );
}

function MultiSessionResultsView({
  session,
  router,
  Colors,
  Gradients,
  getMaturityColors,
  styles,
}: {
  session: AnalyzeResponse[];
  router: ReturnType<typeof useRouter>;
  Colors: ReturnType<typeof useTheme>["Colors"];
  Gradients: ReturnType<typeof useTheme>["Gradients"];
  getMaturityColors: ReturnType<typeof useTheme>["getMaturityColors"];
  styles: ReturnType<typeof createStyles>;
}) {
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const summary = computeSessionSummary(session);
  const stageColors = getMaturityColors(summary.maturity_stage);
  const prediction = MATURITY_PREDICTION[summary.maturity_stage] ?? MATURITY_PREDICTION.peak;

  const totalTrichomes = summary.total_trichome_count;
  const clearPct = pct(summary.trichome_distribution.clear, totalTrichomes);
  const cloudyPct = pct(summary.trichome_distribution.cloudy, totalTrichomes);
  const amberPct = pct(summary.trichome_distribution.amber, totalTrichomes);
  const dominantType = getDominantType(summary.trichome_distribution);
  const dominantPct = pct(summary.trichome_distribution[dominantType], totalTrichomes);
  const orangePct = Math.round(summary.avg_orange_ratio * 100);

  const openCapture = (idx: number) => {
    AnalysisResultStore.set(session[idx]);
    setSelectedIndex(idx);
  };

  if (selectedIndex !== null) {
    return (
      <SafeAreaView style={styles.safe} edges={["top"]}>
        <View style={styles.content}>
          <SingleCaptureDetail
            result={session[selectedIndex]}
            title={`Capture ${selectedIndex + 1} of ${session.length}`}
            onBack={() => setSelectedIndex(null)}
            showSaveButton={false}
            router={router}
            Colors={Colors}
            Gradients={Gradients}
            getMaturityColors={getMaturityColors}
            styles={styles}
          />
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.safe} edges={["top"]}>
      <View style={styles.content}>
        <View style={styles.header}>
          <Pressable style={styles.backButton} onPress={() => router.back()}>
            <Ionicons name="arrow-back" size={20} color={Colors.textPrimary} />
            <Text style={styles.backLabel}>Back</Text>
          </Pressable>
          <Text style={styles.headerTitle}>Flower Analysis</Text>
          <View style={{ width: 28 }} />
        </View>

        <ScrollView contentContainerStyle={styles.scroll} showsVerticalScrollIndicator={false}>
          {/* Hero */}
          <View style={styles.heroCard}>
            <View style={styles.heroBadge}>
              <Ionicons name="sparkles" size={12} color={Colors.accent} />
              <Text style={styles.heroBadgeText}>Combined analysis</Text>
            </View>
            <Text style={styles.heroTitle}>{session.length} captures</Text>
            <Text style={styles.heroSubtitle}>
              Averaged across {session.length} images with trichome and stigma scoring.
            </Text>
          </View>

          {/* Maturity prediction */}
          <View style={[styles.predictionCard, { backgroundColor: stageColors.background }]}>
            <View style={styles.predictionTop}>
              <Text style={styles.predictionLabel}>COMBINED MATURITY</Text>
              <View style={styles.stigmaBadge}>
                <Text style={styles.stigmaBadgeText}>{session.length} photos</Text>
              </View>
            </View>
            <Text style={[styles.predictionHeading, { color: stageColors.text }]}>
              {prediction.heading}
            </Text>
            <Text style={styles.predictionSubtitle}>{prediction.subtitle}</Text>
          </View>

          {/* Key stats */}
          <View style={styles.statsRow}>
            <View style={styles.statCard}>
              <Text style={styles.statValue}>{cloudyPct}%</Text>
              <Text style={styles.statLabel}>Avg cloudy</Text>
            </View>
            <View style={styles.statCard}>
              <Text style={styles.statValue}>{orangePct}%</Text>
              <Text style={styles.statLabel}>Avg orange stigmas</Text>
            </View>
          </View>

          {/* Trichome donut */}
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>Average trichome profile</Text>
            <Text style={styles.sectionSubtitle}>Aggregated across all captures</Text>
          </View>
          <View style={styles.donutRow}>
            <DonutChart
              size={170}
              strokeWidth={22}
              centerLabel={`${dominantPct}%`}
              centerSublabel={dominantType.charAt(0).toUpperCase() + dominantType.slice(1)}
              segments={[
                { value: clearPct, color: TRICHOME_COLORS.clear },
                { value: cloudyPct, color: TRICHOME_COLORS.cloudy },
                { value: amberPct, color: TRICHOME_COLORS.amber },
              ]}
            />
            <View style={styles.donutLegend}>
              {([["clear", clearPct], ["cloudy", cloudyPct], ["amber", amberPct]] as [TrichomeType, number][]).map(
                ([t, v]) => (
                  <View key={t} style={styles.legendRow}>
                    <View style={[styles.legendDot, { backgroundColor: TRICHOME_COLORS[t] }]} />
                    <Text style={styles.legendLabel}>{t.charAt(0).toUpperCase() + t.slice(1)}</Text>
                    <Text style={[styles.legendPct, { color: TRICHOME_COLORS[t] }]}>{v}%</Text>
                  </View>
                )
              )}
            </View>
          </View>

          {/* Combined recommendation */}
          <View style={styles.recommendationCard}>
            <View style={styles.recRow}>
              <Ionicons name="bulb-outline" size={16} color={Colors.accent} />
              <Text style={styles.recLabel}>COMBINED RECOMMENDATION</Text>
            </View>
            <Text style={styles.recText}>{summary.recommendation}</Text>
          </View>

          {/* Individual captures */}
          <View style={[styles.sectionHeader, { marginTop: 8 }]}>
            <Text style={styles.sectionTitle}>Individual captures</Text>
            <Text style={styles.sectionSubtitle}>Tap any card to see full analysis</Text>
          </View>

          {session.map((capture, idx) => {
            const captureDist = capture.trichome_result.distribution;
            const captureTotal = capture.trichome_result.total_count;
            const captureMaturity = getMaturityColors(capture.maturity_stage);
            const captureClearPct = pct(captureDist.clear, captureTotal);
            const captureCloudyPct = pct(captureDist.cloudy, captureTotal);
            const captureAmberPct = pct(captureDist.amber, captureTotal);
            const captureGreenPct = Math.round(capture.stigma_result.avg_green_ratio * 100);
            const captureOrangePct = Math.round(capture.stigma_result.avg_orange_ratio * 100);

            return (
              <Pressable key={idx} style={styles.sessionCaptureCard} onPress={() => openCapture(idx)}>
                <View style={styles.sessionCaptureCardTop}>
                  <View style={styles.sessionCaptureThumbContainer}>
                    {capture.annotated_image_url ? (
                      <Image
                        source={{ uri: capture.annotated_image_url }}
                        style={styles.sessionCaptureThumb}
                        resizeMode="cover"
                      />
                    ) : (
                      <View style={styles.sessionCaptureThumbPlaceholder}>
                        <Ionicons name="image-outline" size={22} color={Colors.textMuted} />
                      </View>
                    )}
                  </View>
                  <View style={styles.sessionCaptureInfo}>
                    <View style={styles.sessionCaptureInfoTop}>
                      <Text style={styles.sessionCaptureTitle}>Capture {idx + 1}</Text>
                      <View style={[styles.sessionCaptureMaturityBadge, { backgroundColor: captureMaturity.background }]}>
                        <Text style={[styles.sessionCaptureMaturityText, { color: captureMaturity.text }]}>
                          {captureMaturity.label}
                        </Text>
                      </View>
                    </View>
                    <View style={styles.sessionCaptureTrichomeBar}>
                      <View style={[styles.sessionCaptureTrichomeSegment, { flex: captureClearPct || 0.001, backgroundColor: TRICHOME_COLORS.clear }]} />
                      <View style={[styles.sessionCaptureTrichomeSegment, { flex: captureCloudyPct || 0.001, backgroundColor: TRICHOME_COLORS.cloudy }]} />
                      <View style={[styles.sessionCaptureTrichomeSegment, { flex: captureAmberPct || 0.001, backgroundColor: TRICHOME_COLORS.amber }]} />
                    </View>
                    <Text style={styles.sessionCaptureStatLine}>
                      {captureCloudyPct}% cloudy · {captureAmberPct}% amber · {captureGreenPct}% green stigmas
                    </Text>
                  </View>
                  <Ionicons name="chevron-forward" size={16} color={Colors.textMuted} />
                </View>
                {captureOrangePct > 0 && (
                  <View style={styles.sessionCaptureStigmaBar}>
                    <View style={[styles.sessionCaptureStigmaSegment, { flex: captureGreenPct, backgroundColor: Colors.accentDark }]} />
                    <View style={[styles.sessionCaptureStigmaSegment, { flex: captureOrangePct, backgroundColor: "#b45309" }]} />
                  </View>
                )}
              </Pressable>
            );
          })}

          {/* Save all */}
          <Pressable onPress={() => router.push("/save-flower" as never)}>
            {({ pressed }) => (
              <LinearGradient
                colors={Gradients.vitality}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
                style={[styles.saveButton, pressed && { opacity: 0.88 }]}
              >
                <Ionicons name="bookmark-outline" size={18} color={Colors.accentText} />
                <Text style={styles.saveButtonText}>Save all to a flower</Text>
              </LinearGradient>
            )}
          </Pressable>

          <Pressable style={styles.addMoreButton} onPress={() => router.push("/camera" as never)}>
            <Ionicons name="add" size={18} color={Colors.textPrimary} />
            <Text style={styles.addMoreText}>Add more images</Text>
          </Pressable>
        </ScrollView>
      </View>
    </SafeAreaView>
  );
}

export default function ResultsScreen() {
  const router = useRouter();
  const { id, index } = useLocalSearchParams<{ id?: string; index?: string }>();
  const { Colors, Gradients, getMaturityColors } = useTheme();
  const styles = createStyles(Colors);
  const [result, setResult] = useState<AnalyzeResponse>(AnalysisResultStore.get() ?? MOCK_RESULT);
  const [session, setSession] = useState<AnalyzeResponse[]>(AnalysisResultStore.getSession());

  const captureIndex = index !== undefined ? parseInt(index, 10) : null;
  const isMultiSession = session.length > 1 && captureIndex === null && !id;

  useFocusEffect(
    useCallback(() => {
      const currentSession = AnalysisResultStore.getSession();
      setSession(currentSession);

      if (id) {
        ApiClient.getAnalysis(id).then(setResult).catch(() => {
          setResult(AnalysisResultStore.get() ?? MOCK_RESULT);
        });
      } else if (captureIndex !== null && currentSession[captureIndex]) {
        setResult(currentSession[captureIndex]);
        AnalysisResultStore.set(currentSession[captureIndex]);
      } else {
        setResult(AnalysisResultStore.get() ?? MOCK_RESULT);
      }
    }, [id, captureIndex])
  );

  if (isMultiSession) {
    return (
      <MultiSessionResultsView
        session={session}
        router={router}
        Colors={Colors}
        Gradients={Gradients}
        getMaturityColors={getMaturityColors}
        styles={styles}
      />
    );
  }

  return (
    <SafeAreaView style={styles.safe} edges={["top"]}>
      <View style={styles.content}>
        <SingleCaptureDetail
          result={result}
          title={captureIndex !== null ? `Capture ${captureIndex + 1}` : "Flower Analysis"}
          onBack={() => router.back()}
          showSaveButton={true}
          router={router}
          Colors={Colors}
          Gradients={Gradients}
          getMaturityColors={getMaturityColors}
          styles={styles}
        />
      </View>
    </SafeAreaView>
  );
}

function createStyles(Colors: ReturnType<typeof useTheme>["Colors"]) { return StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  content: {
    flex: 1,
  },
  header: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: 20,
    paddingTop: 8,
    paddingBottom: 12,
    backgroundColor: Colors.surface,
    borderBottomWidth: 1,
    borderBottomColor: Colors.borderSubtle,
  },
  backButton: {
    flexDirection: "row",
    alignItems: "center",
    gap: 4,
  },
  backLabel: {
    fontSize: 15,
    fontWeight: "600",
    color: Colors.textPrimary,
  },
  headerTitle: {
    fontSize: 16,
    fontWeight: "700",
    color: Colors.textPrimary,
  },
  headerAction: {
    padding: 4,
  },
  scroll: {
    paddingHorizontal: 16,
    paddingBottom: 32,
    paddingTop: 16,
    gap: 12,
  },
  heroCard: {
    backgroundColor: Colors.surfaceElevated,
    borderRadius: 20,
    padding: 20,
    gap: 8,
  },
  heroBadge: {
    flexDirection: "row",
    alignItems: "center",
    gap: 5,
    backgroundColor: Colors.accentSurface,
    alignSelf: "flex-start",
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 999,
  },
  heroBadgeText: {
    fontSize: 11,
    fontWeight: "700",
    color: Colors.accent,
    letterSpacing: 0.3,
  },
  heroTitle: {
    fontSize: 26,
    fontWeight: "800",
    color: Colors.textPrimary,
    letterSpacing: -0.5,
  },
  heroSubtitle: {
    fontSize: 13,
    color: Colors.textSecondary,
    lineHeight: 19,
  },
  predictionCard: {
    borderRadius: 16,
    padding: 18,
    gap: 6,
  },
  predictionTop: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  predictionLabel: {
    fontSize: 10,
    fontWeight: "800",
    color: Colors.textSecondary,
    letterSpacing: 1.5,
  },
  stigmaBadge: {
    backgroundColor: "rgba(255,255,255,0.12)",
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 999,
  },
  stigmaBadgeText: {
    fontSize: 11,
    fontWeight: "600",
    color: Colors.textPrimary,
  },
  predictionHeading: {
    fontSize: 26,
    fontWeight: "800",
    letterSpacing: -0.5,
  },
  predictionSubtitle: {
    fontSize: 13,
    color: Colors.textSecondary,
    lineHeight: 19,
  },
  statsRow: {
    flexDirection: "row",
    gap: 10,
  },
  statCard: {
    flex: 1,
    backgroundColor: Colors.surface,
    borderRadius: 14,
    padding: 16,
    gap: 4,
  },
  statValue: {
    fontSize: 28,
    fontWeight: "800",
    color: Colors.textPrimary,
    letterSpacing: -0.5,
  },
  statLabel: {
    fontSize: 12,
    color: Colors.textSecondary,
    fontWeight: "500",
  },
  sectionHeader: {
    gap: 2,
    marginBottom: 4,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: "700",
    color: Colors.textPrimary,
  },
  sectionSubtitle: {
    fontSize: 12,
    color: Colors.textMuted,
  },
  donutRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    backgroundColor: Colors.surface,
    borderRadius: 20,
    padding: 20,
    gap: 16,
  },
  donutLegend: {
    flex: 1,
    gap: 10,
  },
  legendRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  legendDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
  },
  legendLabel: {
    flex: 1,
    fontSize: 13,
    fontWeight: "600",
    color: Colors.textSecondary,
  },
  legendPct: {
    fontSize: 14,
    fontWeight: "800",
  },
  dominantCard: {
    backgroundColor: Colors.surface,
    borderRadius: 14,
    padding: 16,
    gap: 4,
  },
  dominantLabel: {
    fontSize: 10,
    fontWeight: "800",
    color: Colors.textMuted,
    letterSpacing: 1.5,
  },
  dominantTitle: {
    fontSize: 18,
    fontWeight: "700",
    color: Colors.textPrimary,
  },
  dominantDesc: {
    fontSize: 13,
    color: Colors.textSecondary,
    lineHeight: 18,
  },
  typeCard: {
    backgroundColor: Colors.surface,
    borderRadius: 14,
    padding: 14,
    gap: 10,
  },
  typeCardTop: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  typeCardLeft: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  typeDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
  },
  typeLabel: {
    fontSize: 15,
    fontWeight: "700",
    color: Colors.textPrimary,
  },
  typePct: {
    fontSize: 17,
    fontWeight: "800",
  },
  typeBarTrack: {
    height: 8,
    borderRadius: 4,
    backgroundColor: Colors.surfaceHighest,
    overflow: "hidden",
  },
  typeBarFill: {
    height: "100%",
    borderRadius: 4,
  },
  typeCardBottom: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  typeCount: {
    fontSize: 12,
    color: Colors.textMuted,
    fontWeight: "500",
  },
  openSamplesLink: {
    flexDirection: "row",
    alignItems: "center",
    gap: 3,
  },
  openSamplesText: {
    fontSize: 11,
    fontWeight: "800",
    color: Colors.accent,
    letterSpacing: 0.8,
  },
  stigmaDistHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: 4,
  },
  stigmaDistTitle: {
    fontSize: 15,
    fontWeight: "700",
    color: Colors.textPrimary,
  },
  stigmaLeadBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 999,
  },
  stigmaLeadText: {
    fontSize: 11,
    fontWeight: "700",
  },
  splitBarContainer: {
    flexDirection: "row",
    height: 64,
    borderRadius: 14,
    overflow: "hidden",
  },
  splitBarGreen: {
    backgroundColor: Colors.accentDark,
    alignItems: "center",
    justifyContent: "center",
    gap: 1,
  },
  splitBarOrange: {
    backgroundColor: "#b45309",
    alignItems: "center",
    justifyContent: "center",
    gap: 1,
  },
  splitBarLabel: {
    fontSize: 18,
    fontWeight: "800",
    color: "#fff",
  },
  splitBarSublabel: {
    fontSize: 10,
    fontWeight: "600",
    color: "rgba(255,255,255,0.7)",
    textTransform: "uppercase",
    letterSpacing: 0.5,
  },
  stigmaCardsRow: {
    flexDirection: "row",
    gap: 10,
  },
  stigmaTypeCard: {
    flex: 1,
    backgroundColor: Colors.surface,
    borderRadius: 14,
    padding: 14,
    borderWidth: 1,
    gap: 6,
  },
  stigmaCardTop: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
  },
  stigmaColorDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
  },
  stigmaTypeName: {
    fontSize: 13,
    fontWeight: "700",
    color: Colors.textPrimary,
  },
  stigmaPct: {
    fontSize: 26,
    fontWeight: "800",
    color: Colors.accent,
    letterSpacing: -0.5,
  },
  stigmaTypeDesc: {
    fontSize: 11,
    color: Colors.textMuted,
    marginBottom: 4,
  },
  captureCard: {
    backgroundColor: Colors.surface,
    borderRadius: 14,
    overflow: "hidden",
    width: 160,
  },
  captureImage: {
    width: "100%",
    height: 120,
    backgroundColor: Colors.surfaceHighest,
  },
  captureLabel: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    padding: 10,
  },
  captureLabelText: {
    fontSize: 12,
    fontWeight: "600",
    color: Colors.textSecondary,
  },
  capturePlaceholder: {
    width: 160,
    height: 120,
    backgroundColor: Colors.surface,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: Colors.border,
    borderStyle: "dashed",
    alignItems: "center",
    justifyContent: "center",
    flexDirection: "row",
    gap: 8,
  },
  capturePlaceholderText: {
    fontSize: 13,
    fontWeight: "600",
    color: Colors.textMuted,
  },
  recommendationCard: {
    backgroundColor: Colors.surfaceElevated,
    borderRadius: 14,
    padding: 16,
    gap: 8,
  },
  recRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
  },
  recLabel: {
    fontSize: 10,
    fontWeight: "800",
    color: Colors.textSecondary,
    letterSpacing: 1.5,
  },
  recText: {
    fontSize: 13,
    color: Colors.textPrimary,
    lineHeight: 20,
  },
  saveButton: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 16,
    borderRadius: 999,
    gap: 8,
  },
  saveButtonText: {
    fontSize: 15,
    fontWeight: "700",
    color: Colors.accentText,
  },
  addMoreButton: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 15,
    borderRadius: 999,
    borderWidth: 1.5,
    borderColor: Colors.border,
    gap: 6,
  },
  addMoreText: {
    fontSize: 15,
    fontWeight: "600",
    color: Colors.textPrimary,
  },
  sessionCaptureCard: {
    backgroundColor: Colors.surface,
    borderRadius: 14,
    padding: 12,
    gap: 8,
  },
  sessionCaptureCardTop: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
  },
  sessionCaptureThumbContainer: {
    width: 64,
    height: 64,
    borderRadius: 10,
    overflow: "hidden",
    backgroundColor: Colors.surfaceElevated,
  },
  sessionCaptureThumb: {
    width: "100%",
    height: "100%",
  },
  sessionCaptureThumbPlaceholder: {
    width: "100%",
    height: "100%",
    alignItems: "center",
    justifyContent: "center",
  },
  sessionCaptureInfo: {
    flex: 1,
    gap: 6,
  },
  sessionCaptureInfoTop: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  sessionCaptureTitle: {
    fontSize: 14,
    fontWeight: "700",
    color: Colors.textPrimary,
  },
  sessionCaptureMaturityBadge: {
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 6,
  },
  sessionCaptureMaturityText: {
    fontSize: 10,
    fontWeight: "700",
    textTransform: "uppercase",
    letterSpacing: 0.4,
  },
  sessionCaptureTrichomeBar: {
    height: 5,
    borderRadius: 3,
    flexDirection: "row",
    overflow: "hidden",
    backgroundColor: Colors.surfaceHighest,
  },
  sessionCaptureTrichomeSegment: {
    height: "100%",
  },
  sessionCaptureStatLine: {
    fontSize: 11,
    color: Colors.textMuted,
  },
  sessionCaptureStigmaBar: {
    height: 4,
    borderRadius: 2,
    flexDirection: "row",
    overflow: "hidden",
    marginTop: -4,
  },
  sessionCaptureStigmaSegment: {
    height: "100%",
  },
}); }
