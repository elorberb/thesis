import { useState } from "react";
import { View, Text, Pressable, StyleSheet, ScrollView } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useRouter, useLocalSearchParams } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { AnalysisResultStore } from "../store/analysisResult";
import { useTheme } from "../contexts/ThemeContext";
import { MOCK_RESULT } from "./results";
import { ZoomableImage } from "../components/ZoomableImage";

type StigmaColor = "green" | "orange";
type StigmaFilter = StigmaColor | "all";

const STIGMA_OVERLAY: Record<StigmaColor, string> = {
  green: "rgba(107,255,143,0.18)",
  orange: "rgba(217,119,6,0.22)",
};

export default function StigmaSamplesScreen() {
  const router = useRouter();
  const { Colors } = useTheme();
  const styles = createStyles(Colors);
  const { type } = useLocalSearchParams<{ type: StigmaFilter }>();
  const result = AnalysisResultStore.get() ?? MOCK_RESULT;
  const filter: StigmaFilter = (type as StigmaFilter) ?? "all";

  const allDetections = result.stigma_result.detections;
  const crops = result.stigma_crops_b64;

  const samples = allDetections.map((det, idx) => ({
    ...det,
    index: idx,
    cropB64: crops?.[idx] ?? null,
    originalColor: (det.green_ratio >= det.orange_ratio ? "green" : "orange") as StigmaColor,
  }));

  const [overrides, setOverrides] = useState<Record<number, StigmaColor>>({});

  const toggleColor = (originalIndex: number, currentColor: StigmaColor) => {
    const next: StigmaColor = currentColor === "green" ? "orange" : "green";
    setOverrides((prev) => ({ ...prev, [originalIndex]: next }));
  };

  const effectiveSamples = samples.map((s) => ({
    ...s,
    effectiveColor: overrides[s.index] ?? s.originalColor,
  }));

  const total = effectiveSamples.length;
  const greenCount = effectiveSamples.filter((s) => s.effectiveColor === "green").length;
  const orangeCount = total - greenCount;
  const liveGreenPct = total > 0 ? Math.round((greenCount / total) * 100) : 0;
  const liveOrangePct = total > 0 ? Math.round((orangeCount / total) * 100) : 0;

  const hasOverrides = Object.keys(overrides).length > 0;

  const displayedSamples =
    filter === "all"
      ? effectiveSamples
      : effectiveSamples.filter((s) => s.effectiveColor === filter);

  return (
    <SafeAreaView style={styles.safe} edges={["top"]}>
      <View style={styles.header}>
        <Pressable style={styles.backButton} onPress={() => router.back()}>
          <Ionicons name="arrow-back" size={20} color={Colors.textPrimary} />
          <Text style={styles.backLabel}>Back</Text>
        </Pressable>
        <Text style={styles.headerTitle}>Stigma Analysis</Text>
        {hasOverrides ? (
          <Pressable onPress={() => setOverrides({})}>
            <Text style={styles.resetText}>Reset</Text>
          </Pressable>
        ) : (
          <View style={{ width: 44 }} />
        )}
      </View>

      <ScrollView contentContainerStyle={styles.scroll} showsVerticalScrollIndicator={false}>
        {/* Live ratio card */}
        <View style={styles.liveCard}>
          <View style={styles.liveCardHeader}>
            <Text style={styles.liveCardTitle}>Average color ratios</Text>
            {hasOverrides && (
              <View style={styles.editedBadge}>
                <Ionicons name="pencil" size={10} color={Colors.warning} />
                <Text style={styles.editedBadgeText}>EDITED</Text>
              </View>
            )}
          </View>

          <Text style={styles.liveCardSubtitle}>Green to orange shift across this image</Text>

          <View style={styles.ratioBar}>
            <View style={[styles.ratioBarGreen, { flex: greenCount || 0.001 }]} />
            <View style={[styles.ratioBarOrange, { flex: orangeCount || 0.001 }]} />
          </View>

          <View style={styles.ratioPills}>
            <View style={styles.ratioPillGreen}>
              <View style={[styles.ratioDot, { backgroundColor: Colors.accent }]} />
              <Text style={styles.ratioPillGreenText}>Green {liveGreenPct}%</Text>
            </View>
            <View style={styles.ratioPillOrange}>
              <View style={[styles.ratioDot, { backgroundColor: "#d97706" }]} />
              <Text style={styles.ratioPillOrangeText}>Orange {liveOrangePct}%</Text>
            </View>
          </View>

          <View style={styles.totalRow}>
            <Text style={styles.totalText}>{total} total detections</Text>
            {hasOverrides && (
              <Text style={styles.overriddenText}>
                {Object.keys(overrides).length} reclassified
              </Text>
            )}
          </View>
        </View>

        {/* Hint */}
        <View style={styles.hintRow}>
          <Ionicons name="finger-print-outline" size={14} color={Colors.textMuted} />
          <Text style={styles.hintText}>
            Tap any sample to toggle between green and orange classification
          </Text>
        </View>

        {/* Crops grid */}
        {displayedSamples.length === 0 ? (
          <View style={styles.emptyState}>
            <Ionicons name="images-outline" size={40} color={Colors.textMuted} />
            <Text style={styles.emptyTitle}>No samples available</Text>
            <Text style={styles.emptyDesc}>
              {crops === null
                ? "Crop images are only available with cloud inference (Modal)."
                : "No detections found."}
            </Text>
          </View>
        ) : (
          <View style={styles.cropGrid}>
            {displayedSamples.map((sample, i) => {
              const color = sample.effectiveColor;
              const accentColor = color === "green" ? Colors.accent : "#d97706";
              const overlayColor = STIGMA_OVERLAY[color];
              const isOverridden = overrides[sample.index] !== undefined;
              const greenPct = Math.round(sample.green_ratio * 100);
              const orangePct = Math.round(sample.orange_ratio * 100);

              return (
                <View
                  key={sample.index}
                  style={[styles.cropCard, { borderColor: accentColor }]}
                >
                  <View style={styles.cropImageContainer}>
                    {sample.cropB64 ? (
                      <ZoomableImage
                        uri={`data:image/jpeg;base64,${sample.cropB64}`}
                        style={StyleSheet.absoluteFill}
                        imageStyle={styles.cropImage}
                        resizeMode="cover"
                        onSingleTap={() => toggleColor(sample.index, color)}
                      >
                        <View style={[styles.colorOverlay, { backgroundColor: overlayColor }]} />
                        <View style={[styles.imageLabel, { backgroundColor: accentColor + "dd" }]}>
                          <Text style={styles.imageLabelText}>{color.toUpperCase()}</Text>
                        </View>
                      </ZoomableImage>
                    ) : (
                      <Pressable style={styles.cropImagePlaceholder} onPress={() => toggleColor(sample.index, color)}>
                        <Ionicons name="flower-outline" size={28} color={accentColor} />
                      </Pressable>
                    )}
                  </View>

                  {isOverridden && (
                    <View style={styles.overriddenBadge}>
                      <Ionicons name="pencil" size={10} color="#fff" />
                    </View>
                  )}

                  <Pressable
                    style={styles.cropInfo}
                    onPress={() => toggleColor(sample.index, color)}
                  >
                    <View style={styles.cropInfoRow}>
                      <Text style={styles.cropSampleLabel}>Sample {i + 1}</Text>
                      <View style={[styles.colorDot, { backgroundColor: accentColor }]} />
                    </View>
                    <Text style={styles.cropRatios}>
                      Green {greenPct}% · Orange {orangePct}%
                    </Text>
                  </Pressable>
                </View>
              );
            })}
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

function createStyles(Colors: ReturnType<typeof useTheme>["Colors"]) { return StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  header: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: 16,
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
  resetText: {
    fontSize: 14,
    fontWeight: "700",
    color: Colors.danger,
  },
  scroll: {
    padding: 16,
    gap: 12,
    paddingBottom: 32,
  },
  liveCard: {
    backgroundColor: Colors.surface,
    borderRadius: 16,
    padding: 16,
    gap: 10,
  },
  liveCardHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  liveCardTitle: {
    fontSize: 15,
    fontWeight: "700",
    color: Colors.textPrimary,
  },
  editedBadge: {
    flexDirection: "row",
    alignItems: "center",
    gap: 4,
    backgroundColor: Colors.warningSurface,
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 999,
  },
  editedBadgeText: {
    fontSize: 9,
    fontWeight: "800",
    color: Colors.warning,
    letterSpacing: 1,
  },
  liveCardSubtitle: {
    fontSize: 12,
    color: Colors.textMuted,
    marginTop: -4,
  },
  ratioBar: {
    flexDirection: "row",
    height: 14,
    borderRadius: 7,
    overflow: "hidden",
    gap: 1,
  },
  ratioBarGreen: {
    backgroundColor: Colors.accentDark,
  },
  ratioBarOrange: {
    backgroundColor: "#b45309",
  },
  ratioPills: {
    flexDirection: "row",
    gap: 8,
  },
  ratioDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  ratioPillGreen: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    backgroundColor: Colors.accentSurface,
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 999,
  },
  ratioPillGreenText: {
    fontSize: 12,
    fontWeight: "700",
    color: Colors.accent,
  },
  ratioPillOrange: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    backgroundColor: "rgba(217,119,6,0.12)",
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 999,
  },
  ratioPillOrangeText: {
    fontSize: 12,
    fontWeight: "700",
    color: "#d97706",
  },
  totalRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    borderTopWidth: 1,
    borderTopColor: Colors.borderSubtle,
    paddingTop: 8,
  },
  totalText: {
    fontSize: 11,
    color: Colors.textMuted,
  },
  overriddenText: {
    fontSize: 11,
    color: Colors.warning,
    fontWeight: "600",
  },
  hintRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    paddingHorizontal: 4,
  },
  hintText: {
    fontSize: 12,
    color: Colors.textMuted,
    flex: 1,
  },
  emptyState: {
    alignItems: "center",
    paddingVertical: 48,
    gap: 8,
  },
  emptyTitle: {
    fontSize: 16,
    fontWeight: "700",
    color: Colors.textSecondary,
  },
  emptyDesc: {
    fontSize: 13,
    color: Colors.textMuted,
    textAlign: "center",
    paddingHorizontal: 24,
    lineHeight: 18,
  },
  cropGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
  },
  cropCard: {
    width: "47.5%",
    backgroundColor: Colors.surfaceElevated,
    borderRadius: 14,
    overflow: "hidden",
    borderWidth: 3,
    position: "relative",
  },
  cropImageContainer: {
    position: "relative",
    width: "100%",
    height: 120,
  },
  cropImage: {
    width: "100%",
    height: "100%",
  },
  cropImagePlaceholder: {
    width: "100%",
    height: "100%",
    backgroundColor: Colors.surfaceHighest,
    alignItems: "center",
    justifyContent: "center",
  },
  colorOverlay: {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
  },
  imageLabel: {
    position: "absolute",
    bottom: 6,
    left: 6,
    paddingHorizontal: 7,
    paddingVertical: 3,
    borderRadius: 6,
  },
  imageLabelText: {
    fontSize: 9,
    fontWeight: "800",
    color: "#fff",
    letterSpacing: 1,
  },
  overriddenBadge: {
    position: "absolute",
    top: 6,
    right: 6,
    width: 20,
    height: 20,
    borderRadius: 10,
    backgroundColor: Colors.warning,
    alignItems: "center",
    justifyContent: "center",
    zIndex: 10,
  },
  cropInfo: {
    padding: 8,
    gap: 3,
  },
  cropInfoRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  cropSampleLabel: {
    fontSize: 12,
    fontWeight: "700",
    color: Colors.textPrimary,
  },
  colorDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
  },
  cropRatios: {
    fontSize: 10,
    color: Colors.textSecondary,
  },
}); }
