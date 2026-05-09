import { useState, useCallback, useRef, useMemo } from "react";
import {
  View,
  Text,
  FlatList,
  Pressable,
  StyleSheet,
  ActivityIndicator,
  Alert,
  TextInput,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useRouter, useLocalSearchParams } from "expo-router";
import { useFocusEffect } from "@react-navigation/native";
import { Ionicons } from "@expo/vector-icons";
import { ApiClient } from "../api/client";
import { useTheme } from "../contexts/ThemeContext";
import { PlantAnalysisItem, MaturityStage, TrichomeType, AnalysisPatch } from "../api/types";
import { AnalysisResultStore } from "../store/analysisResult";
import { ZoomableImage } from "../components/ZoomableImage";

type ImageView = "original" | "detected";

type EditState = {
  maturity_stage: MaturityStage;
  recommendation: string;
  trichome_distribution: Record<TrichomeType, number> | null;
  stigma_green: string;
  stigma_orange: string;
};

const MATURITY_STAGES: MaturityStage[] = ["early", "developing", "peak", "mature", "late"];

type TimelineItem =
  | { kind: "single"; analysis: PlantAnalysisItem }
  | { kind: "session"; analyses: PlantAnalysisItem[] };

function groupBySession(items: PlantAnalysisItem[]): TimelineItem[] {
  if (items.length === 0) return [];
  const sorted = [...items].sort(
    (a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
  );
  const groups: PlantAnalysisItem[][] = [];
  for (const item of sorted) {
    const lastGroup = groups[groups.length - 1];
    if (lastGroup) {
      const lastTime = new Date(lastGroup[lastGroup.length - 1].created_at).getTime();
      const itemTime = new Date(item.created_at).getTime();
      if (itemTime - lastTime < 5 * 60 * 1000) {
        lastGroup.push(item);
        continue;
      }
    }
    groups.push([item]);
  }
  return groups
    .map((g) =>
      g.length === 1
        ? ({ kind: "single", analysis: g[0] } as TimelineItem)
        : ({ kind: "session", analyses: g } as TimelineItem)
    )
    .reverse();
}

function cycleMaturity(current: MaturityStage): MaturityStage {
  const idx = MATURITY_STAGES.indexOf(current);
  return MATURITY_STAGES[(idx + 1) % MATURITY_STAGES.length];
}

function AnalysisImageBlock({
  imageUrl,
  annotatedUrl,
  blockStyles,
}: {
  imageUrl: string;
  annotatedUrl: string | null;
  blockStyles: ReturnType<typeof createStyles>;
}) {
  const { Colors } = useTheme();
  const hasDetected = !!annotatedUrl;
  const [view, setView] = useState<ImageView>(hasDetected ? "detected" : "original");
  const activeUrl = view === "detected" && annotatedUrl ? annotatedUrl : imageUrl;

  return (
    <View>
      <ZoomableImage
        uri={activeUrl}
        style={blockStyles.analysisImage}
        imageStyle={blockStyles.analysisImage}
        resizeMode="cover"
      />
      {hasDetected && (
        <View style={blockStyles.imageToggleRow}>
          <Pressable
            style={[blockStyles.imageToggleBtn, view === "original" && blockStyles.imageToggleBtnActive]}
            onPress={() => setView("original")}
          >
            <Ionicons name="image-outline" size={12} color={view === "original" ? Colors.accentText : Colors.textMuted} />
            <Text style={[blockStyles.imageToggleText, view === "original" && blockStyles.imageToggleTextActive]}>
              Original
            </Text>
          </Pressable>
          <Pressable
            style={[blockStyles.imageToggleBtn, view === "detected" && blockStyles.imageToggleBtnActive]}
            onPress={() => setView("detected")}
          >
            <Ionicons name="scan-outline" size={12} color={view === "detected" ? Colors.accentText : Colors.textMuted} />
            <Text style={[blockStyles.imageToggleText, view === "detected" && blockStyles.imageToggleTextActive]}>
              Detected
            </Text>
          </Pressable>
        </View>
      )}
    </View>
  );
}

export default function PlantDetailScreen() {
  const router = useRouter();
  const { Colors, getMaturityColors } = useTheme();
  const { plantId, plantName } = useLocalSearchParams<{ plantId: string; plantName: string }>();
  const [analyses, setAnalyses] = useState<PlantAnalysisItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editState, setEditState] = useState<EditState | null>(null);
  const [saving, setSaving] = useState(false);
  const [loadingTrichomeEdit, setLoadingTrichomeEdit] = useState(false);
  const [loadingSessionKey, setLoadingSessionKey] = useState<string | null>(null);
  const fromTrichomeSamplesRef = useRef(false);
  const editingIdRef = useRef<string | null>(null);

  const fetchAnalyses = useCallback(async () => {
    if (!plantId) return;
    setLoading(true);
    setError(null);
    try {
      const history = await ApiClient.listPlantAnalyses(plantId);
      setAnalyses(history.items);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load analyses");
    } finally {
      setLoading(false);
    }
  }, [plantId]);

  useFocusEffect(
    useCallback(() => {
      if (fromTrichomeSamplesRef.current) {
        fromTrichomeSamplesRef.current = false;
        const stored = AnalysisResultStore.get();
        if (stored && editingIdRef.current) {
          setEditState((prev) =>
            prev ? { ...prev, trichome_distribution: stored.trichome_result.distribution } : prev
          );
        }
        return;
      }
      fetchAnalyses();
    }, [fetchAnalyses])
  );

  const styles = createStyles(Colors);

  const startEdit = (item: PlantAnalysisItem) => {
    setEditingId(item.id);
    editingIdRef.current = item.id;
    setEditState({
      maturity_stage: item.maturity_stage,
      recommendation: item.recommendation,
      trichome_distribution: item.trichome_distribution ?? null,
      stigma_green: item.stigma_ratios ? String(Math.round(item.stigma_ratios.green * 100)) : "50",
      stigma_orange: item.stigma_ratios ? String(Math.round(item.stigma_ratios.orange * 100)) : "50",
    });
  };

  const cancelEdit = () => {
    setEditingId(null);
    editingIdRef.current = null;
    setEditState(null);
  };

  const saveEdit = async () => {
    if (!editingId || !editState) return;
    setSaving(true);
    try {
      const patch: AnalysisPatch = {
        maturity_stage: editState.maturity_stage,
        recommendation: editState.recommendation,
      };
      if (editState.trichome_distribution) {
        patch.trichome_distribution = editState.trichome_distribution;
      }
      const greenVal = parseFloat(editState.stigma_green) / 100;
      const orangeVal = parseFloat(editState.stigma_orange) / 100;
      if (!isNaN(greenVal) && !isNaN(orangeVal)) {
        patch.stigma_ratios = { green: greenVal, orange: orangeVal };
      }
      await ApiClient.patchAnalysis(editingId, patch);
      setAnalyses((prev) =>
        prev.map((a) => {
          if (a.id !== editingId) return a;
          return {
            ...a,
            maturity_stage: editState.maturity_stage,
            recommendation: editState.recommendation,
            trichome_distribution: editState.trichome_distribution,
            stigma_ratios: patch.stigma_ratios ?? a.stigma_ratios,
          };
        })
      );
      setEditingId(null);
      editingIdRef.current = null;
      setEditState(null);
    } catch {
      Alert.alert("Error", "Failed to save changes.");
    } finally {
      setSaving(false);
    }
  };

  const openTrichomeSamples = async (item: PlantAnalysisItem) => {
    setLoadingTrichomeEdit(true);
    try {
      const full = await ApiClient.getAnalysis(item.id);
      if (editState?.trichome_distribution) {
        full.trichome_result.distribution = editState.trichome_distribution;
      }
      AnalysisResultStore.set(full);
      fromTrichomeSamplesRef.current = true;
      editingIdRef.current = item.id;
      router.push("/trichome-samples?type=cloudy" as never);
    } catch {
      Alert.alert("Error", "Failed to load analysis data.");
    } finally {
      setLoadingTrichomeEdit(false);
    }
  };

  const handleDeleteAnalysis = (analysisId: string) => {
    Alert.alert("Delete this scan?", "This cannot be undone.", [
      { text: "Cancel", style: "cancel" },
      {
        text: "Delete",
        style: "destructive",
        onPress: async () => {
          try {
            await ApiClient.deleteAnalysis(analysisId);
            setAnalyses((prev) => prev.filter((a) => a.id !== analysisId));
          } catch {
            Alert.alert("Error", "Failed to delete scan.");
          }
        },
      },
    ]);
  };

  const handleDeleteSession = (ids: string[]) => {
    Alert.alert(`Delete ${ids.length} scans?`, "This session will be removed. This cannot be undone.", [
      { text: "Cancel", style: "cancel" },
      {
        text: "Delete all",
        style: "destructive",
        onPress: async () => {
          try {
            await Promise.all(ids.map((id) => ApiClient.deleteAnalysis(id)));
            setAnalyses((prev) => prev.filter((a) => !ids.includes(a.id)));
          } catch {
            Alert.alert("Error", "Failed to delete scans.");
          }
        },
      },
    ]);
  };

  const openSession = async (sessionAnalyses: PlantAnalysisItem[]) => {
    const key = sessionAnalyses.map((a) => a.id).join("-");
    setLoadingSessionKey(key);
    try {
      const full = await Promise.all(sessionAnalyses.map((a) => ApiClient.getAnalysis(a.id)));
      AnalysisResultStore.setSession(full);
      router.push("/results" as never);
    } catch {
      Alert.alert("Error", "Failed to load session data.");
    } finally {
      setLoadingSessionKey(null);
    }
  };

  const timelineItems = useMemo(() => groupBySession(analyses), [analyses]);

  const renderSessionCard = ({ analyses: sessionAnalyses, isLast }: { analyses: PlantAnalysisItem[]; isLast: boolean }) => {
    const combined = { clear: 0, cloudy: 0, amber: 0 };
    sessionAnalyses.forEach((a) => {
      if (a.trichome_distribution) {
        combined.clear += a.trichome_distribution.clear;
        combined.cloudy += a.trichome_distribution.cloudy;
        combined.amber += a.trichome_distribution.amber;
      }
    });
    const combinedTotal = combined.clear + combined.cloudy + combined.amber;

    const stageCounts: Partial<Record<MaturityStage, number>> = {};
    sessionAnalyses.forEach((a) => {
      stageCounts[a.maturity_stage] = (stageCounts[a.maturity_stage] ?? 0) + 1;
    });
    const dominantStage = (Object.entries(stageCounts) as [MaturityStage, number][]).sort(
      ([, a], [, b]) => b - a
    )[0][0];
    const maturity = getMaturityColors(dominantStage);

    const firstDate = new Date(sessionAnalyses[0].created_at);
    const dateLabel = firstDate.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
    const timeLabel = firstDate.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" });

    const cloudyPct = combinedTotal > 0 ? Math.round((combined.cloudy / combinedTotal) * 100) : 0;
    const amberPct = combinedTotal > 0 ? Math.round((combined.amber / combinedTotal) * 100) : 0;
    const clearPct = combinedTotal > 0 ? Math.round((combined.clear / combinedTotal) * 100) : 0;

    const combinedRec = sessionAnalyses[0].recommendation;

    return (
      <View style={styles.timelineItem}>
        <View style={styles.timelineConnector}>
          <View style={[styles.timelineDot, { backgroundColor: maturity.text }]} />
          {!isLast && <View style={styles.timelineLine} />}
        </View>

        <View style={styles.analysisCard}>
          <View style={styles.cardTopRow}>
            <View style={styles.dateBlock}>
              <Text style={styles.dateText}>{dateLabel}</Text>
              <Text style={styles.timeText}>{timeLabel}</Text>
            </View>
            <View style={styles.cardTopRight}>
              <View style={[styles.maturityBadge, { backgroundColor: maturity.background }]}>
                <Ionicons name="images-outline" size={11} color={maturity.text} style={{ marginRight: 3 }} />
                <Text style={[styles.maturityText, { color: maturity.text }]}>
                  {sessionAnalyses.length} captures
                </Text>
              </View>
              <Pressable onPress={() => handleDeleteSession(sessionAnalyses.map((a) => a.id))} hitSlop={8}>
                <Ionicons name="trash-outline" size={15} color={Colors.danger} />
              </Pressable>
            </View>
          </View>

          {combinedTotal > 0 && (
            <View style={styles.trichomeBlock}>
              <View style={styles.trichomeBar}>
                <View style={[styles.trichomeSegment, { flex: combined.clear || 0.001, backgroundColor: "#9ca3af" }]} />
                <View style={[styles.trichomeSegment, { flex: combined.cloudy || 0.001, backgroundColor: "#a78bfa" }]} />
                <View style={[styles.trichomeSegment, { flex: combined.amber || 0.001, backgroundColor: "#f59e0b" }]} />
              </View>
              <View style={styles.statsRow}>
                <View style={styles.statChip}>
                  <View style={[styles.statDot, { backgroundColor: "#9ca3af" }]} />
                  <Text style={styles.statValue}>{clearPct}%</Text>
                  <Text style={styles.statLabel}>clear</Text>
                </View>
                <View style={styles.statChip}>
                  <View style={[styles.statDot, { backgroundColor: "#a78bfa" }]} />
                  <Text style={styles.statValue}>{cloudyPct}%</Text>
                  <Text style={styles.statLabel}>cloudy</Text>
                </View>
                <View style={styles.statChip}>
                  <View style={[styles.statDot, { backgroundColor: "#f59e0b" }]} />
                  <Text style={styles.statValue}>{amberPct}%</Text>
                  <Text style={styles.statLabel}>amber</Text>
                </View>
              </View>
            </View>
          )}

          <View style={styles.insightBlock}>
            <Ionicons name="bulb-outline" size={13} color={Colors.textMuted} style={{ marginTop: 1 }} />
            <Text style={styles.recommendation} numberOfLines={3}>{combinedRec}</Text>
          </View>

          {(() => {
            const sessionKey = sessionAnalyses.map((a) => a.id).join("-");
            const isLoadingThis = loadingSessionKey === sessionKey;
            return (
              <Pressable
                style={styles.viewDetailsRow}
                onPress={() => openSession(sessionAnalyses)}
                disabled={isLoadingThis}
              >
                {isLoadingThis ? (
                  <ActivityIndicator size="small" color={Colors.accent} />
                ) : (
                  <>
                    <Text style={styles.viewDetailsText}>View session results</Text>
                    <Ionicons name="chevron-forward" size={13} color={Colors.accent} />
                  </>
                )}
              </Pressable>
            );
          })()}
        </View>
      </View>
    );
  };

  const renderTimelineItem = ({ item, index }: { item: TimelineItem; index: number }) => {
    const isLast = index === timelineItems.length - 1;
    if (item.kind === "single") {
      return renderAnalysisItem({ item: item.analysis, isLast });
    }
    return renderSessionCard({ analyses: item.analyses, isLast });
  };

  const renderAnalysisItem = ({ item, isLast }: { item: PlantAnalysisItem; isLast: boolean }) => {
    const isEditing = editingId === item.id;
    const currentEdit = isEditing ? editState : null;
    const displayStage = currentEdit?.maturity_stage ?? item.maturity_stage;
    const maturity = getMaturityColors(displayStage);
    const date = new Date(item.created_at);
    const dateLabel = date.toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
    const timeLabel = date.toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
    });

    const trichomeDist = currentEdit?.trichome_distribution ?? item.trichome_distribution;
    const trichomeTotal = trichomeDist
      ? Object.values(trichomeDist).reduce((a, b) => a + b, 0)
      : 0;
    const cloudy = trichomeDist?.cloudy ?? 0;
    const amber = trichomeDist?.amber ?? 0;
    const clear = trichomeDist?.clear ?? 0;

    const stigmaRatios = item.stigma_ratios;

    return (
      <View style={styles.timelineItem}>
        <View style={styles.timelineConnector}>
          <View style={[styles.timelineDot, { backgroundColor: maturity.text }]} />
          {!isLast && <View style={styles.timelineLine} />}
        </View>

        <View style={styles.analysisCard}>
          {/* Card header */}
          <View style={styles.cardTopRow}>
            <View style={styles.dateBlock}>
              <Text style={styles.dateText}>{dateLabel}</Text>
              <Text style={styles.timeText}>{timeLabel}</Text>
            </View>
            <View style={styles.cardTopRight}>
              {isEditing ? (
                <Pressable
                  style={[styles.maturityBadge, { backgroundColor: maturity.background }]}
                  onPress={() =>
                    setEditState((prev) =>
                      prev ? { ...prev, maturity_stage: cycleMaturity(prev.maturity_stage) } : prev
                    )
                  }
                >
                  <Ionicons name="chevron-forward" size={10} color={maturity.text} style={{ marginRight: 2 }} />
                  <Text style={[styles.maturityText, { color: maturity.text }]}>
                    {maturity.label}
                  </Text>
                </Pressable>
              ) : (
                <View style={[styles.maturityBadge, { backgroundColor: maturity.background }]}>
                  <Text style={[styles.maturityText, { color: maturity.text }]}>
                    {maturity.label}
                  </Text>
                </View>
              )}
              {!isEditing && (
                <Pressable onPress={() => startEdit(item)} hitSlop={8}>
                  <Ionicons name="pencil-outline" size={15} color={Colors.textMuted} />
                </Pressable>
              )}
              {!isEditing && (
                <Pressable onPress={() => handleDeleteAnalysis(item.id)} hitSlop={8}>
                  <Ionicons name="trash-outline" size={15} color={Colors.danger} />
                </Pressable>
              )}
            </View>
          </View>

          {item.image_url ? (
            <AnalysisImageBlock imageUrl={item.image_url} annotatedUrl={item.annotated_image_url} blockStyles={styles} />
          ) : item.annotated_image_url ? (
            <ZoomableImage
              uri={item.annotated_image_url}
              style={styles.analysisImage}
              imageStyle={styles.analysisImage}
              resizeMode="cover"
            />
          ) : null}

          {trichomeDist && trichomeTotal > 0 && (
            <Pressable
              style={styles.trichomeBlock}
              onPress={isEditing ? () => openTrichomeSamples(item) : undefined}
              disabled={!isEditing || loadingTrichomeEdit}
            >
              <View style={styles.trichomeBar}>
                <View style={[styles.trichomeSegment, { flex: clear, backgroundColor: "#9ca3af" }]} />
                <View style={[styles.trichomeSegment, { flex: cloudy, backgroundColor: "#a78bfa" }]} />
                <View style={[styles.trichomeSegment, { flex: amber, backgroundColor: "#f59e0b" }]} />
              </View>
              <View style={styles.statsRow}>
                <View style={styles.statChip}>
                  <View style={[styles.statDot, { backgroundColor: "#9ca3af" }]} />
                  <Text style={styles.statValue}>{Math.round((clear / trichomeTotal) * 100)}%</Text>
                  <Text style={styles.statLabel}>clear</Text>
                </View>
                <View style={styles.statChip}>
                  <View style={[styles.statDot, { backgroundColor: "#a78bfa" }]} />
                  <Text style={styles.statValue}>{Math.round((cloudy / trichomeTotal) * 100)}%</Text>
                  <Text style={styles.statLabel}>cloudy</Text>
                </View>
                <View style={styles.statChip}>
                  <View style={[styles.statDot, { backgroundColor: "#f59e0b" }]} />
                  <Text style={styles.statValue}>{Math.round((amber / trichomeTotal) * 100)}%</Text>
                  <Text style={styles.statLabel}>amber</Text>
                </View>
              </View>
              {isEditing && (
                <View style={styles.editTrichomeHint}>
                  {loadingTrichomeEdit ? (
                    <ActivityIndicator size="small" color={Colors.accent} />
                  ) : (
                    <>
                      <Ionicons name="open-outline" size={12} color={Colors.accent} />
                      <Text style={styles.editTrichomeHintText}>Tap to relabel detections</Text>
                    </>
                  )}
                </View>
              )}
            </Pressable>
          )}

          {stigmaRatios && !isEditing && (
            <View style={styles.stigmaBlock}>
              <View style={styles.stigmaLabelRow}>
                <Text style={styles.stigmaLabel}>Stigma color</Text>
                <Text style={styles.stigmaRatioText}>
                  {Math.round(stigmaRatios.orange * 100)}% orange · {Math.round(stigmaRatios.green * 100)}% green
                </Text>
              </View>
              <View style={styles.stigmaBar}>
                <View style={[styles.stigmaSegment, { flex: stigmaRatios.orange, backgroundColor: "#f97316" }]} />
                <View style={[styles.stigmaSegment, { flex: stigmaRatios.green, backgroundColor: Colors.accent }]} />
              </View>
            </View>
          )}

          {isEditing && stigmaRatios && (
            <View style={styles.stigmaEditBlock}>
              <Text style={styles.stigmaLabel}>Stigma ratios (%)</Text>
              <View style={styles.stigmaEditRow}>
                <View style={styles.stigmaEditField}>
                  <View style={[styles.stigmaColorDot, { backgroundColor: Colors.accent }]} />
                  <Text style={styles.stigmaEditLabel}>Green</Text>
                  <TextInput
                    style={styles.stigmaInput}
                    value={currentEdit?.stigma_green ?? ""}
                    onChangeText={(v) =>
                      setEditState((prev) => (prev ? { ...prev, stigma_green: v } : prev))
                    }
                    keyboardType="numeric"
                    maxLength={3}
                    placeholderTextColor={Colors.textMuted}
                  />
                  <Text style={styles.stigmaEditUnit}>%</Text>
                </View>
                <View style={styles.stigmaEditField}>
                  <View style={[styles.stigmaColorDot, { backgroundColor: "#f97316" }]} />
                  <Text style={styles.stigmaEditLabel}>Orange</Text>
                  <TextInput
                    style={styles.stigmaInput}
                    value={currentEdit?.stigma_orange ?? ""}
                    onChangeText={(v) =>
                      setEditState((prev) => (prev ? { ...prev, stigma_orange: v } : prev))
                    }
                    keyboardType="numeric"
                    maxLength={3}
                    placeholderTextColor={Colors.textMuted}
                  />
                  <Text style={styles.stigmaEditUnit}>%</Text>
                </View>
              </View>
            </View>
          )}

          {/* Recommendation */}
          {isEditing ? (
            <View style={styles.recommendationEditBlock}>
              <View style={styles.insightLabelRow}>
                <Ionicons name="bulb-outline" size={13} color={Colors.accent} style={{ marginTop: 1 }} />
                <Text style={styles.insightEditLabel}>Recommendation</Text>
              </View>
              <TextInput
                style={styles.recommendationInput}
                value={currentEdit?.recommendation ?? ""}
                onChangeText={(v) =>
                  setEditState((prev) => (prev ? { ...prev, recommendation: v } : prev))
                }
                multiline
                numberOfLines={4}
                placeholderTextColor={Colors.textMuted}
                placeholder="Enter recommendation…"
              />
            </View>
          ) : (
            <View style={styles.insightBlock}>
              <Ionicons name="bulb-outline" size={13} color={Colors.textMuted} style={{ marginTop: 1 }} />
              <Text style={styles.recommendation} numberOfLines={3}>
                {item.recommendation}
              </Text>
            </View>
          )}

          {!isEditing && (
            <Pressable
              style={styles.viewDetailsRow}
              onPress={() => router.push(`/results?id=${item.id}` as never)}
            >
              <Text style={styles.viewDetailsText}>View full analysis</Text>
              <Ionicons name="chevron-forward" size={13} color={Colors.accent} />
            </Pressable>
          )}

          {/* Edit action buttons */}
          {isEditing && (
            <View style={styles.editActions}>
              <Pressable style={styles.cancelButton} onPress={cancelEdit} disabled={saving}>
                <Text style={styles.cancelButtonText}>Cancel</Text>
              </Pressable>
              <Pressable style={styles.saveButton} onPress={saveEdit} disabled={saving}>
                {saving ? (
                  <ActivityIndicator size="small" color={Colors.accentText} />
                ) : (
                  <Text style={styles.saveButtonText}>Save</Text>
                )}
              </Pressable>
            </View>
          )}
        </View>
      </View>
    );
  };

  return (
    <SafeAreaView style={styles.safe} edges={["top"]}>
      <View style={styles.header}>
        <Pressable style={styles.backButton} onPress={() => router.back()}>
          <Ionicons name="arrow-back" size={20} color={Colors.textPrimary} />
        </Pressable>
        <View style={styles.headerCenter}>
          <Text style={styles.headerTitle} numberOfLines={1}>{plantName ?? "Plant"}</Text>
          <Text style={styles.headerSub}>{analyses.length} scan{analyses.length !== 1 ? "s" : ""}</Text>
        </View>
        <View style={{ width: 36 }} />
      </View>

      {loading && (
        <View style={styles.centeredState}>
          <ActivityIndicator size="large" color={Colors.accent} />
          <Text style={styles.stateText}>Loading history...</Text>
        </View>
      )}

      {!loading && error && (
        <View style={styles.centeredState}>
          <Ionicons name="warning-outline" size={40} color={Colors.danger} />
          <Text style={styles.errorText}>{error}</Text>
          <Pressable style={styles.retryButton} onPress={fetchAnalyses}>
            <Text style={styles.retryText}>Retry</Text>
          </Pressable>
        </View>
      )}

      {!loading && !error && analyses.length === 0 && (
        <View style={styles.centeredState}>
          <View style={styles.emptyIconBox}>
            <Ionicons name="scan-outline" size={30} color={Colors.accent} />
          </View>
          <Text style={styles.emptyTitle}>No scans yet</Text>
          <Text style={styles.emptySub}>
            Analyze this plant to start building its maturity history.
          </Text>
          <Pressable style={styles.ctaButton} onPress={() => router.push("/camera")}>
            <Ionicons name="camera-outline" size={15} color={Colors.accentText} />
            <Text style={styles.ctaButtonText}>Scan Now</Text>
          </Pressable>
        </View>
      )}

      {!loading && !error && analyses.length > 0 && (
        <FlatList
          data={timelineItems}
          keyExtractor={(item) =>
            item.kind === "single" ? item.analysis.id : item.analyses.map((a) => a.id).join("-")
          }
          renderItem={renderTimelineItem}
          contentContainerStyle={styles.listContent}
          showsVerticalScrollIndicator={false}
        />
      )}
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
    paddingHorizontal: 16,
    paddingTop: 8,
    paddingBottom: 12,
    borderBottomWidth: 1,
    borderBottomColor: Colors.borderSubtle,
    gap: 12,
  },
  backButton: {
    width: 36,
    height: 36,
    borderRadius: 10,
    backgroundColor: Colors.surfaceElevated,
    alignItems: "center",
    justifyContent: "center",
  },
  headerCenter: {
    flex: 1,
    gap: 2,
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: "700",
    color: Colors.textPrimary,
  },
  headerSub: {
    fontSize: 12,
    color: Colors.textMuted,
  },
  centeredState: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 40,
    gap: 12,
  },
  stateText: {
    fontSize: 14,
    color: Colors.textMuted,
    marginTop: 8,
  },
  errorText: {
    fontSize: 14,
    color: Colors.danger,
    textAlign: "center",
  },
  retryButton: {
    backgroundColor: Colors.surfaceElevated,
    paddingHorizontal: 24,
    paddingVertical: 10,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  retryText: {
    fontSize: 14,
    fontWeight: "600",
    color: Colors.textPrimary,
  },
  emptyIconBox: {
    width: 72,
    height: 72,
    borderRadius: 22,
    backgroundColor: Colors.accentSurface,
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 8,
  },
  emptyTitle: {
    fontSize: 18,
    fontWeight: "700",
    color: Colors.textPrimary,
    textAlign: "center",
  },
  emptySub: {
    fontSize: 14,
    color: Colors.textMuted,
    textAlign: "center",
    lineHeight: 22,
  },
  ctaButton: {
    marginTop: 12,
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    backgroundColor: Colors.accent,
    paddingHorizontal: 24,
    paddingVertical: 11,
    borderRadius: 12,
  },
  ctaButtonText: {
    fontSize: 14,
    fontWeight: "700",
    color: Colors.accentText,
  },
  listContent: {
    paddingHorizontal: 16,
    paddingTop: 12,
    paddingBottom: 40,
  },
  timelineItem: {
    flexDirection: "row",
    gap: 12,
  },
  timelineConnector: {
    alignItems: "center",
    width: 20,
    paddingTop: 6,
  },
  timelineDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    borderWidth: 2,
    borderColor: Colors.surface,
  },
  timelineLine: {
    flex: 1,
    width: 1,
    backgroundColor: Colors.border,
    marginTop: 4,
    marginBottom: -4,
    opacity: 0.6,
  },
  analysisCard: {
    flex: 1,
    backgroundColor: Colors.surface,
    borderRadius: 16,
    padding: 14,
    marginBottom: 16,
    gap: 12,
  },
  cardTopRow: {
    flexDirection: "row",
    alignItems: "flex-start",
    justifyContent: "space-between",
  },
  cardTopRight: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
  },
  dateBlock: {
    gap: 2,
  },
  dateText: {
    fontSize: 14,
    fontWeight: "700",
    color: Colors.textPrimary,
  },
  timeText: {
    fontSize: 12,
    color: Colors.textMuted,
  },
  maturityBadge: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 8,
  },
  maturityText: {
    fontSize: 11,
    fontWeight: "700",
    textTransform: "uppercase",
    letterSpacing: 0.5,
  },
  analysisImage: {
    width: "100%",
    height: 220,
    borderRadius: 12,
    backgroundColor: Colors.surfaceElevated,
  },
  imageToggleRow: {
    flexDirection: "row",
    gap: 6,
    marginTop: 8,
  },
  imageToggleBtn: {
    flexDirection: "row",
    alignItems: "center",
    gap: 4,
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 999,
    backgroundColor: Colors.surfaceHighest,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  imageToggleBtnActive: {
    backgroundColor: Colors.accent,
    borderColor: Colors.accent,
  },
  imageToggleText: {
    fontSize: 11,
    fontWeight: "600",
    color: Colors.textMuted,
    letterSpacing: 0.3,
  },
  imageToggleTextActive: {
    color: Colors.accentText,
  },
  trichomeBlock: {
    gap: 8,
  },
  trichomeBar: {
    height: 6,
    borderRadius: 3,
    flexDirection: "row",
    overflow: "hidden",
    backgroundColor: Colors.surfaceElevated,
  },
  trichomeSegment: {
    height: "100%",
  },
  statsRow: {
    flexDirection: "row",
    gap: 6,
  },
  statChip: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: Colors.surfaceElevated,
    borderRadius: 8,
    paddingHorizontal: 7,
    paddingVertical: 6,
    gap: 4,
  },
  statDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
  },
  statValue: {
    fontSize: 12,
    fontWeight: "700",
    color: Colors.textPrimary,
  },
  statLabel: {
    fontSize: 10,
    color: Colors.textMuted,
  },
  editTrichomeHint: {
    flexDirection: "row",
    alignItems: "center",
    gap: 5,
    paddingTop: 2,
  },
  editTrichomeHintText: {
    fontSize: 11,
    color: Colors.accent,
    fontWeight: "600",
  },
  stigmaBlock: {
    gap: 6,
  },
  stigmaLabelRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  stigmaLabel: {
    fontSize: 11,
    fontWeight: "600",
    color: Colors.textMuted,
    textTransform: "uppercase",
    letterSpacing: 0.5,
  },
  stigmaBar: {
    height: 8,
    borderRadius: 4,
    flexDirection: "row",
    overflow: "hidden",
    backgroundColor: Colors.surfaceElevated,
  },
  stigmaSegment: {
    height: "100%",
  },
  stigmaRatioText: {
    fontSize: 11,
    color: Colors.textMuted,
  },
  stigmaEditBlock: {
    gap: 8,
  },
  stigmaEditRow: {
    flexDirection: "row",
    gap: 10,
  },
  stigmaEditField: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    gap: 5,
    backgroundColor: Colors.surfaceElevated,
    borderRadius: 8,
    paddingHorizontal: 8,
    paddingVertical: 6,
  },
  stigmaColorDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  stigmaEditLabel: {
    fontSize: 11,
    color: Colors.textMuted,
    flex: 1,
  },
  stigmaInput: {
    fontSize: 13,
    fontWeight: "700",
    color: Colors.textPrimary,
    width: 36,
    textAlign: "right",
  },
  stigmaEditUnit: {
    fontSize: 11,
    color: Colors.textMuted,
  },
  insightBlock: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: 7,
    backgroundColor: Colors.surfaceElevated,
    borderRadius: 10,
    padding: 10,
  },
  recommendation: {
    flex: 1,
    fontSize: 12,
    color: Colors.textSecondary,
    lineHeight: 18,
  },
  recommendationEditBlock: {
    gap: 6,
    backgroundColor: Colors.surfaceElevated,
    borderRadius: 10,
    padding: 10,
  },
  insightLabelRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 5,
    marginBottom: 4,
  },
  insightEditLabel: {
    fontSize: 11,
    fontWeight: "700",
    color: Colors.accent,
    letterSpacing: 0.5,
    textTransform: "uppercase",
  },
  recommendationInput: {
    fontSize: 12,
    color: Colors.textPrimary,
    lineHeight: 18,
    textAlignVertical: "top",
    minHeight: 72,
    borderWidth: 1,
    borderColor: Colors.border,
    borderRadius: 8,
    padding: 8,
    backgroundColor: Colors.surface,
  },
  editActions: {
    flexDirection: "row",
    gap: 8,
  },
  cancelButton: {
    flex: 1,
    paddingVertical: 10,
    borderRadius: 10,
    borderWidth: 1.5,
    borderColor: Colors.border,
    alignItems: "center",
    justifyContent: "center",
  },
  cancelButtonText: {
    fontSize: 13,
    fontWeight: "600",
    color: Colors.textSecondary,
  },
  saveButton: {
    flex: 2,
    paddingVertical: 10,
    borderRadius: 10,
    backgroundColor: Colors.accent,
    alignItems: "center",
    justifyContent: "center",
  },
  saveButtonText: {
    fontSize: 13,
    fontWeight: "700",
    color: Colors.accentText,
  },
  viewDetailsRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "flex-end",
    gap: 3,
    paddingTop: 8,
    borderTopWidth: 1,
    borderTopColor: Colors.borderSubtle,
    marginTop: 4,
    minHeight: 28,
  },
  viewDetailsText: {
    fontSize: 11,
    fontWeight: "700",
    color: Colors.accent,
    letterSpacing: 0.4,
    textTransform: "uppercase",
  },
}); }
