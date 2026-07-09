import { useState, useCallback } from "react";
import { View, Text, FlatList, StyleSheet, Alert } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useRouter } from "expo-router";
import { useFocusEffect } from "@react-navigation/native";
import { ApiClient } from "../api/client";
import { AnalysisListItem } from "../api/types";
import { useTheme } from "../contexts/ThemeContext";
import { AnalysisCard } from "../components/AnalysisCard";
import { EmptyState } from "../components/EmptyState";
import { ErrorState } from "../components/ErrorState";
import { LoadingState } from "../components/LoadingState";

export default function HistoryScreen() {
  const router = useRouter();
  const { Colors } = useTheme();
  const styles = createStyles(Colors);
  const [analyses, setAnalyses] = useState<AnalysisListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchAnalyses = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await ApiClient.listAnalyses(50);
      setAnalyses(response.items);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load analyses");
    } finally {
      setLoading(false);
    }
  }, []);

  useFocusEffect(
    useCallback(() => {
      fetchAnalyses();
    }, [fetchAnalyses])
  );

  const handleDelete = (analysis: AnalysisListItem) => {
    Alert.alert("Delete this analysis?", "This cannot be undone.", [
      { text: "Cancel", style: "cancel" },
      {
        text: "Delete",
        style: "destructive",
        onPress: async () => {
          try {
            await ApiClient.deleteAnalysis(analysis.id);
            setAnalyses((prev) => prev.filter((a) => a.id !== analysis.id));
          } catch {
            Alert.alert("Error", "Failed to delete analysis.");
          }
        },
      },
    ]);
  };

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>History</Text>
        <Text style={styles.headerSub}>Your most recent analyses</Text>
      </View>

      {loading && <LoadingState message="Loading history…" />}

      {!loading && error && <ErrorState message={error} onRetry={fetchAnalyses} />}

      {!loading && !error && analyses.length === 0 && (
        <EmptyState
          icon="time-outline"
          title="No analyses yet"
          message="Start your first scan to see your analysis history here."
          actionLabel="Start First Scan"
          actionIcon="camera-outline"
          onAction={() => router.push("/camera")}
        />
      )}

      {!loading && !error && analyses.length > 0 && (
        <FlatList
          data={analyses}
          keyExtractor={(item) => item.id}
          renderItem={({ item }) => (
            <AnalysisCard
              analysis={item}
              showTime
              onPress={() => router.push({ pathname: "/results", params: { id: item.id } })}
              onDelete={() => handleDelete(item)}
            />
          )}
          contentContainerStyle={styles.listContent}
          showsVerticalScrollIndicator={false}
          ListHeaderComponent={
            <Text style={styles.listHeader}>
              {analyses.length} analys{analyses.length !== 1 ? "es" : "is"}
            </Text>
          }
        />
      )}
    </SafeAreaView>
  );
}

function createStyles(Colors: ReturnType<typeof useTheme>["Colors"]) {
  return StyleSheet.create({
    safe: {
      flex: 1,
      backgroundColor: Colors.background,
    },
    header: {
      paddingHorizontal: 20,
      paddingTop: 10,
      paddingBottom: 16,
      gap: 2,
    },
    headerTitle: {
      fontSize: 26,
      fontWeight: "800",
      color: Colors.textPrimary,
      letterSpacing: -0.5,
    },
    headerSub: {
      fontSize: 13,
      color: Colors.textMuted,
    },
    listContent: {
      paddingHorizontal: 16,
      paddingTop: 4,
      paddingBottom: 16,
    },
    listHeader: {
      fontSize: 12,
      fontWeight: "600",
      color: Colors.textMuted,
      marginBottom: 12,
      textTransform: "uppercase",
      letterSpacing: 0.8,
    },
  });
}
