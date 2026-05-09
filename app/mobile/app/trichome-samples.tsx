import { useState, useCallback } from "react";
import { View, Text, Pressable, StyleSheet, ScrollView } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useRouter, useLocalSearchParams } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { AnalysisResultStore } from "../store/analysisResult";
import { useTheme } from "../contexts/ThemeContext";
import { TrichomeType } from "../api/types";
import { MOCK_RESULT } from "./results";
import { ZoomableImage } from "../components/ZoomableImage";

type ExtendedTrichomeType = TrichomeType | "not_a_trichome";

const TYPE_COLORS: Record<ExtendedTrichomeType, string> = {
  clear: "#6f758b",
  cloudy: "#dfe4fe",
  amber: "#d97706",
  not_a_trichome: "#4b5563",
};

const TYPE_LABELS: Record<ExtendedTrichomeType, string> = {
  clear: "Clear",
  cloudy: "Cloudy",
  amber: "Amber",
  not_a_trichome: "Not a trichome",
};

const CYCLE_ORDER: ExtendedTrichomeType[] = ["clear", "cloudy", "amber", "not_a_trichome"];

export default function TrichomeSamplesScreen() {
  const router = useRouter();
  const { Colors } = useTheme();
  const { type } = useLocalSearchParams<{ type: TrichomeType }>();
  const result = AnalysisResultStore.get() ?? MOCK_RESULT;
  const focusedType: TrichomeType = (type as TrichomeType) ?? "cloudy";

  const allDetections = result.trichome_result.detections;
  const crops = result.trichome_crops_b64;

  const samples = allDetections.map((det, idx) => ({
    ...det,
    index: idx,
    cropB64: crops?.[idx] ?? null,
  }));

  const [overrides, setOverrides] = useState<Record<number, ExtendedTrichomeType>>({});

  const cycleType = (originalIndex: number, currentType: ExtendedTrichomeType) => {
    const nextType = CYCLE_ORDER[(CYCLE_ORDER.indexOf(currentType) + 1) % CYCLE_ORDER.length];
    setOverrides((prev) => ({ ...prev, [originalIndex]: nextType }));
  };

  const effectiveSamples = samples
    .filter((s) => s.trichome_type === focusedType)
    .map((s) => ({
      ...s,
      effectiveType: (overrides[s.index] ?? s.trichome_type) as ExtendedTrichomeType,
    }));

  const total = effectiveSamples.length;
  const validSamples = effectiveSamples.filter((s) => s.effectiveType !== "not_a_trichome");
  const excludedCount = total - validSamples.length;
  const liveCounts = { clear: 0, cloudy: 0, amber: 0 };
  validSamples.forEach((s) => liveCounts[s.effectiveType as TrichomeType]++);
  const validTotal = validSamples.length;
  const livePct = (t: TrichomeType) =>
    validTotal > 0 ? Math.round((liveCounts[t] / validTotal) * 100) : 0;

  const hasOverrides = Object.keys(overrides).length > 0;
  const styles = createStyles(Colors);

  const handleBack = useCallback(() => {
    if (hasOverrides) {
      const stored = AnalysisResultStore.get();
      if (stored) {
        const updatedDetections = stored.trichome_result.detections
          .map((det, idx) => ({
            ...det,
            effectiveType: (overrides[idx] ?? det.trichome_type) as ExtendedTrichomeType,
          }))
          .filter((d) => d.effectiveType !== "not_a_trichome")
          .map(({ effectiveType, ...det }) => ({
            ...det,
            trichome_type: effectiveType as TrichomeType,
          }));
        const newDist = { clear: 0, cloudy: 0, amber: 0 };
        updatedDetections.forEach((d) => newDist[d.trichome_type]++);
        AnalysisResultStore.set({
          ...stored,
          trichome_result: {
            ...stored.trichome_result,
            detections: updatedDetections,
            distribution: newDist,
            total_count: updatedDetections.length,
          },
        });
      }
    }
    router.back();
  }, [hasOverrides, overrides, router]);

  return (
    <SafeAreaView style={styles.safe} edges={["top"]}>
      <View style={styles.header}>
        <Pressable style={styles.backButton} onPress={handleBack}>
          <Ionicons name="arrow-back" size={20} color={Colors.textPrimary} />
          <Text style={styles.backLabel}>Back</Text>
        </Pressable>
        <Text style={styles.headerTitle}>{TYPE_LABELS[focusedType]} Trichomes</Text>
        {hasOverrides ? (
          <Pressable onPress={() => setOverrides({})}>
            <Text style={styles.resetText}>Reset</Text>
          </Pressable>
        ) : (
          <View style={{ width: 44 }} />
        )}
      </View>

      <ScrollView contentContainerStyle={styles.scroll} showsVerticalScrollIndicator={false}>
        {/* Live distribution bar */}
        <View style={styles.liveCard}>
          <View style={styles.liveCardHeader}>
            <Text style={styles.liveCardTitle}>
              {total} {TYPE_LABELS[focusedType]} samples
            </Text>
            {hasOverrides && (
              <View style={styles.editedBadge}>
                <Ionicons name="pencil" size={10} color={Colors.warning} />
                <Text style={styles.editedBadgeText}>EDITED</Text>
              </View>
            )}
          </View>

          <View style={styles.distributionBar}>
            {(["clear", "cloudy", "amber"] as TrichomeType[]).map((t) => (
              <View
                key={t}
                style={[
                  styles.distributionSegment,
                  { flex: liveCounts[t] || 0.001, backgroundColor: TYPE_COLORS[t] },
                ]}
              />
            ))}
          </View>

          <View style={styles.liveStats}>
            {(["clear", "cloudy", "amber"] as TrichomeType[]).map((t) => (
              <View key={t} style={styles.liveStat}>
                <View style={[styles.liveStatDot, { backgroundColor: TYPE_COLORS[t] }]} />
                <Text style={styles.liveStatLabel}>{TYPE_LABELS[t]}</Text>
                <Text style={[styles.liveStatPct, { color: TYPE_COLORS[t] }]}>
                  {livePct(t)}%
                </Text>
              </View>
            ))}
          </View>

          <View style={styles.totalRow}>
            <Text style={styles.totalText}>{validTotal} valid · {total} shown</Text>
            {excludedCount > 0 && (
              <Text style={styles.excludedText}>
                {excludedCount} excluded
              </Text>
            )}
            {hasOverrides && excludedCount === 0 && (
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
            Tap any sample to cycle: clear → cloudy → amber → not a trichome → clear
          </Text>
        </View>

        {/* Crops grid */}
        {effectiveSamples.length === 0 ? (
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
            {effectiveSamples.map((sample, i) => {
              const isExcluded = sample.effectiveType === "not_a_trichome";
              const borderColor = TYPE_COLORS[sample.effectiveType];
              const isOverridden = overrides[sample.index] !== undefined;
              const conf = Math.round(sample.confidence * 100);
              const isFocused = sample.effectiveType === focusedType;

              return (
                <View
                  key={i}
                  style={[
                    styles.cropCard,
                    { borderColor },
                    isFocused && styles.cropCardFocused,
                    isExcluded && styles.cropCardExcluded,
                  ]}
                >
                  {sample.cropB64 ? (
                    <ZoomableImage
                      uri={`data:image/jpeg;base64,${sample.cropB64}`}
                      style={[styles.cropImage, isExcluded && styles.cropImageExcluded]}
                      imageStyle={[styles.cropImage, isExcluded && styles.cropImageExcluded]}
                      resizeMode="cover"
                      onSingleTap={() => cycleType(sample.index, sample.effectiveType)}
                    />
                  ) : (
                    <Pressable
                      style={[styles.cropImagePlaceholder, { borderBottomColor: borderColor }]}
                      onPress={() => cycleType(sample.index, sample.effectiveType)}
                    >
                      <Ionicons name={isExcluded ? "close-circle-outline" : "leaf-outline"} size={28} color={borderColor} />
                    </Pressable>
                  )}

                  {isOverridden && (
                    <View style={styles.overriddenBadge}>
                      <Ionicons name="pencil" size={10} color="#fff" />
                    </View>
                  )}

                  <Pressable
                    style={styles.cropInfo}
                    onPress={() => cycleType(sample.index, sample.effectiveType)}
                  >
                    <View style={styles.cropInfoRow}>
                      <Text style={styles.cropSampleLabel}>#{i + 1}</Text>
                      <View
                        style={[styles.cropTypePill, { backgroundColor: borderColor + "33" }]}
                      >
                        <View style={[styles.cropTypeDot, { backgroundColor: borderColor }]} />
                        <Text style={[styles.cropTypeText, { color: borderColor }]}>
                          {TYPE_LABELS[sample.effectiveType]}
                        </Text>
                      </View>
                    </View>
                    {!isExcluded && <Text style={styles.cropConf}>{conf}% confidence</Text>}
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
    gap: 12,
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
  distributionBar: {
    flexDirection: "row",
    height: 14,
    borderRadius: 7,
    overflow: "hidden",
    gap: 1,
  },
  distributionSegment: {
    height: "100%",
  },
  liveStats: {
    flexDirection: "row",
    gap: 0,
  },
  liveStat: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    gap: 5,
  },
  liveStatDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  liveStatLabel: {
    fontSize: 12,
    color: Colors.textSecondary,
    fontWeight: "500",
    flex: 1,
  },
  liveStatPct: {
    fontSize: 13,
    fontWeight: "800",
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
  excludedText: {
    fontSize: 11,
    color: Colors.danger,
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
  cropCardFocused: {
    borderWidth: 3,
  },
  cropCardExcluded: {
    opacity: 0.45,
  },
  cropImageExcluded: {
    opacity: 0.5,
  },
  cropImage: {
    width: "100%",
    height: 120,
  },
  cropImagePlaceholder: {
    width: "100%",
    height: 120,
    backgroundColor: Colors.surfaceHighest,
    alignItems: "center",
    justifyContent: "center",
    borderBottomWidth: 2,
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
    color: Colors.textMuted,
  },
  cropTypePill: {
    flexDirection: "row",
    alignItems: "center",
    gap: 4,
    paddingHorizontal: 7,
    paddingVertical: 3,
    borderRadius: 999,
  },
  cropTypeDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
  },
  cropTypeText: {
    fontSize: 10,
    fontWeight: "800",
    letterSpacing: 0.3,
  },
  cropConf: {
    fontSize: 10,
    color: Colors.textMuted,
  },
}); }
