import { useState, useCallback } from "react";
import {
  View,
  Text,
  FlatList,
  Pressable,
  StyleSheet,
  ActivityIndicator,
  Alert,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useRouter } from "expo-router";
import { useFocusEffect } from "@react-navigation/native";
import { Ionicons } from "@expo/vector-icons";
import { ApiClient } from "../api/client";
import { PlantResponse } from "../api/types";
import { useTheme } from "../contexts/ThemeContext";

export default function HistoryScreen() {
  const router = useRouter();
  const { Colors } = useTheme();
  const styles = createStyles(Colors);
  const [plants, setPlants] = useState<PlantResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchPlants = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await ApiClient.listPlants();
      setPlants(response.items);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load plants");
    } finally {
      setLoading(false);
    }
  }, []);

  useFocusEffect(
    useCallback(() => {
      fetchPlants();
    }, [fetchPlants])
  );

  const handleDeletePlant = (plant: PlantResponse) => {
    Alert.alert(
      `Delete "${plant.name}"?`,
      "This will permanently delete this plant and all its linked data.",
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Delete",
          style: "destructive",
          onPress: async () => {
            try {
              await ApiClient.deletePlant(plant.id);
              setPlants((prev) => prev.filter((p) => p.id !== plant.id));
            } catch {
              Alert.alert("Error", "Failed to delete plant.");
            }
          },
        },
      ]
    );
  };

  const renderPlantCard = ({ item }: { item: PlantResponse }) => {
    const createdDate = new Date(item.created_at).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    });
    const metaCount = Object.keys(item.metadata || {}).length;
    const isHarvested = item.status === "harvested";

    return (
      <Pressable
        style={({ pressed }) => [styles.plantCard, pressed && styles.plantCardPressed]}
        onPress={() => router.push({ pathname: "/plant-detail", params: { plantId: item.id, plantName: item.name } })}
      >
        <View style={[styles.cardAccent, isHarvested && styles.cardAccentHarvested]} />
        <View style={styles.plantIconContainer}>
          <Ionicons name="leaf" size={18} color={isHarvested ? Colors.warning : Colors.accent} />
        </View>
        <View style={styles.plantCardContent}>
          <Text style={styles.plantName} numberOfLines={1}>{item.name}</Text>
          <View style={styles.plantMeta}>
            <Text style={styles.plantDate}>{createdDate}</Text>
            {metaCount > 0 && (
              <>
                <View style={styles.metaDot} />
                <Text style={styles.plantDate}>{metaCount} tag{metaCount !== 1 ? "s" : ""}</Text>
              </>
            )}
            {isHarvested && (
              <>
                <View style={styles.metaDot} />
                <Text style={[styles.plantDate, { color: Colors.warning }]}>harvested</Text>
              </>
            )}
          </View>
        </View>
        <View style={styles.plantCardActions}>
          <Pressable style={styles.deleteButton} onPress={() => handleDeletePlant(item)} hitSlop={12}>
            <Ionicons name="trash-outline" size={14} color={Colors.textMuted} />
          </Pressable>
          <Ionicons name="chevron-forward" size={16} color={Colors.textMuted} />
        </View>
      </Pressable>
    );
  };

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>My Plants</Text>
      </View>

      {loading && (
        <View style={styles.centeredState}>
          <ActivityIndicator size="large" color={Colors.accent} />
          <Text style={styles.stateText}>Loading plants...</Text>
        </View>
      )}

      {!loading && error && (
        <View style={styles.centeredState}>
          <Ionicons name="warning-outline" size={40} color={Colors.danger} />
          <Text style={styles.errorText}>{error}</Text>
          <Pressable style={styles.retryButton} onPress={fetchPlants}>
            <Text style={styles.retryText}>Retry</Text>
          </Pressable>
        </View>
      )}

      {!loading && !error && plants.length === 0 && (
        <View style={styles.centeredState}>
          <View style={styles.emptyIconContainer}>
            <View style={styles.emptyIconInner}>
              <Ionicons name="leaf-outline" size={32} color={Colors.accent} />
            </View>
          </View>
          <Text style={styles.emptyTitle}>No plants yet</Text>
          <Text style={styles.emptySub}>
            Complete a scan and save it to a plant ID to start tracking its maturity over time.
          </Text>
          <Pressable style={styles.ctaButton} onPress={() => router.push("/camera")}>
            <Ionicons name="camera-outline" size={16} color={Colors.accentText} />
            <Text style={styles.ctaButtonText}>Start First Scan</Text>
          </Pressable>
        </View>
      )}

      {!loading && !error && plants.length > 0 && (
        <FlatList
          data={plants}
          keyExtractor={(item) => item.id}
          renderItem={renderPlantCard}
          contentContainerStyle={styles.listContent}
          showsVerticalScrollIndicator={false}
          ListHeaderComponent={
            <Text style={styles.listHeader}>{plants.length} plant{plants.length !== 1 ? "s" : ""}</Text>
          }
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
    paddingHorizontal: 20,
    paddingTop: 10,
    paddingBottom: 16,
  },
  headerTitle: {
    fontSize: 26,
    fontWeight: "800",
    color: Colors.textPrimary,
    letterSpacing: -0.5,
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
    lineHeight: 20,
  },
  retryButton: {
    marginTop: 8,
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
  emptyIconContainer: {
    marginBottom: 8,
  },
  emptyIconInner: {
    width: 80,
    height: 80,
    borderRadius: 24,
    backgroundColor: Colors.accentSurface,
    alignItems: "center",
    justifyContent: "center",
  },
  emptyTitle: {
    fontSize: 19,
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
    paddingVertical: 12,
    borderRadius: 12,
  },
  ctaButtonText: {
    fontSize: 14,
    fontWeight: "700",
    color: Colors.accentText,
    letterSpacing: 0.2,
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
  plantCard: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: Colors.surface,
    borderRadius: 14,
    marginBottom: 8,
    overflow: "hidden",
  },
  plantCardPressed: {
    opacity: 0.75,
  },
  cardAccent: {
    width: 3,
    alignSelf: "stretch",
    backgroundColor: Colors.accent,
  },
  cardAccentHarvested: {
    backgroundColor: Colors.warning,
  },
  plantIconContainer: {
    width: 36,
    height: 36,
    borderRadius: 10,
    backgroundColor: Colors.accentSurface,
    alignItems: "center",
    justifyContent: "center",
    marginLeft: 12,
  },
  plantCardContent: {
    flex: 1,
    paddingVertical: 14,
    paddingLeft: 12,
    gap: 4,
  },
  plantName: {
    fontSize: 15,
    fontWeight: "700",
    color: Colors.textPrimary,
  },
  plantMeta: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
  },
  plantDate: {
    fontSize: 12,
    color: Colors.textMuted,
  },
  metaDot: {
    width: 3,
    height: 3,
    borderRadius: 2,
    backgroundColor: Colors.textMuted,
    opacity: 0.5,
  },
  plantCardActions: {
    flexDirection: "row",
    alignItems: "center",
    gap: 4,
    paddingRight: 12,
  },
  deleteButton: {
    padding: 6,
  },
}); }
