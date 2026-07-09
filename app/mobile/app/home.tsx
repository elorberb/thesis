import { useState, useCallback } from "react";
import { View, Text, Pressable, StyleSheet, ScrollView, ActivityIndicator } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useRouter } from "expo-router";
import { useFocusEffect } from "@react-navigation/native";
import { Ionicons } from "@expo/vector-icons";
import { LinearGradient } from "expo-linear-gradient";
import { ApiClient } from "../api/client";
import { AnalysisListItem } from "../api/types";
import { useTheme } from "../contexts/ThemeContext";
import { useAuth } from "../contexts/AuthContext";
import { AnalysisCard } from "../components/AnalysisCard";
import { AppButton } from "../components/AppButton";

export default function HomeScreen() {
  const router = useRouter();
  const { Colors, Gradients } = useTheme();
  const { user } = useAuth();
  const [recentAnalyses, setRecentAnalyses] = useState<AnalysisListItem[]>([]);
  const [loadingRecent, setLoadingRecent] = useState(true);

  useFocusEffect(
    useCallback(() => {
      setLoadingRecent(true);
      ApiClient.listAnalyses(2)
        .then((res) => setRecentAnalyses(res.items))
        .catch(() => setRecentAnalyses([]))
        .finally(() => setLoadingRecent(false));
    }, [])
  );

  const styles = createStyles(Colors);

  return (
    <SafeAreaView style={styles.safe} edges={["top"]}>
      <View style={styles.content}>
        <ScrollView
          contentContainerStyle={styles.scroll}
          showsVerticalScrollIndicator={false}
        >
          <View style={styles.header}>
            <View style={styles.brandGroup}>
              <LinearGradient
                colors={Gradients.vitality}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
                style={styles.brandIcon}
              >
                <Ionicons name="search" size={16} color={Colors.accentText} />
              </LinearGradient>
              <View style={styles.brandTextGroup}>
                <Text style={styles.brandName}>LOUPELAB</Text>
                <Text style={styles.brandTagline}>Maturity Intelligence</Text>
              </View>
            </View>
            <Pressable
              style={styles.profileButton}
              onPress={() => router.push("/profile")}
              accessibilityRole="button"
              accessibilityLabel="Open profile"
            >
              <Text style={styles.profileInitial}>
                {(user?.user_metadata?.full_name as string | undefined)?.[0]?.toUpperCase() ??
                  user?.email?.[0]?.toUpperCase() ?? "?"}
              </Text>
            </Pressable>
          </View>

          <Pressable style={styles.heroCard} onPress={() => router.push("/camera")}>
            <View style={styles.heroContent}>
              <View style={styles.heroBadge}>
                <Text style={styles.heroBadgeText}>NEW ANALYSIS</Text>
              </View>
              <Text style={styles.heroTitle}>Analyze Your Flower</Text>
              <Text style={styles.heroSubtitle}>
                Get instant insights on flower maturity and suggested review timing.
              </Text>
            </View>
            <View style={styles.heroIconArea}>
              <Text style={styles.heroIconLabel}>Start Scan</Text>
              <LinearGradient
                colors={Gradients.vitality}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
                style={styles.heroIconCircle}
              >
                <Ionicons name="camera" size={28} color={Colors.accentText} />
              </LinearGradient>
            </View>
          </Pressable>

          <View style={styles.navCard}>
            <Pressable style={styles.navRow} onPress={() => router.push("/my-plants")}>
              <View style={styles.navRowLeft}>
                <View style={[styles.navIconBox, { backgroundColor: Colors.accentSurface }]}>
                  <Ionicons name="folder" size={20} color={Colors.accent} />
                </View>
                <Text style={styles.navRowTitle}>My Plants</Text>
              </View>
              <Ionicons name="chevron-forward" size={18} color={Colors.textMuted} />
            </Pressable>

            <View style={styles.navDivider} />

            <Pressable style={styles.navRow} onPress={() => router.push("/how-it-works")}>
              <View style={styles.navRowLeft}>
                <View style={[styles.navIconBox, { backgroundColor: Colors.tertiarySurface }]}>
                  <Ionicons name="information-circle" size={20} color={Colors.tertiary} />
                </View>
                <Text style={styles.navRowTitle}>How It Works</Text>
              </View>
              <Ionicons name="chevron-forward" size={18} color={Colors.textMuted} />
            </Pressable>
          </View>

          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>Recent Analyses</Text>
            <Pressable onPress={() => router.push("/history")}>
              <Text style={styles.seeAll}>See all</Text>
            </Pressable>
          </View>

          {loadingRecent ? (
            <View style={styles.emptyCard}>
              <ActivityIndicator color={Colors.accent} />
            </View>
          ) : recentAnalyses.length === 0 ? (
            <View style={styles.emptyCard}>
              <View style={styles.emptyIconBox}>
                <Ionicons name="camera-outline" size={28} color={Colors.textMuted} />
              </View>
              <Text style={styles.emptyTitle}>No analyses yet</Text>
              <Text style={styles.emptySub}>Start your first scan to see results here.</Text>
              <View style={styles.emptyAction}>
                <AppButton
                  label="Start First Scan"
                  icon="camera-outline"
                  onPress={() => router.push("/camera")}
                  fullWidth={false}
                />
              </View>
            </View>
          ) : (
            recentAnalyses.map((item) => (
              <AnalysisCard
                key={item.id}
                analysis={item}
                onPress={() => router.push({ pathname: "/results", params: { id: item.id } })}
              />
            ))
          )}
        </ScrollView>

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
  scroll: {
    paddingHorizontal: 20,
    paddingBottom: 16,
  },
  header: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingTop: 14,
    marginBottom: 24,
  },
  brandGroup: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
  },
  brandIcon: {
    width: 36,
    height: 36,
    borderRadius: 10,
    alignItems: "center",
    justifyContent: "center",
  },
  brandTextGroup: {
    gap: 1,
  },
  brandName: {
    fontSize: 17,
    fontWeight: "800",
    color: Colors.accent,
    letterSpacing: 2.5,
  },
  brandTagline: {
    fontSize: 9,
    fontWeight: "600",
    color: Colors.textMuted,
    letterSpacing: 1.5,
    textTransform: "uppercase",
  },
  profileButton: {
    width: 38,
    height: 38,
    borderRadius: 19,
    backgroundColor: Colors.accentSurface,
    borderWidth: 1.5,
    borderColor: Colors.accent,
    alignItems: "center",
    justifyContent: "center",
  },
  profileInitial: {
    fontSize: 15,
    fontWeight: "700",
    color: Colors.accent,
  },
  heroCard: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: Colors.surface,
    borderRadius: 24,
    borderWidth: 1,
    borderColor: "rgba(107,255,143,0.2)",
    overflow: "hidden",
    marginBottom: 16,
    padding: 22,
  },
  heroContent: {
    flex: 1,
    gap: 10,
  },
  heroBadge: {
    alignSelf: "flex-start",
    backgroundColor: Colors.accentSurface,
    borderRadius: 999,
    paddingHorizontal: 10,
    paddingVertical: 4,
  },
  heroBadgeText: {
    fontSize: 9,
    fontWeight: "800",
    color: Colors.accent,
    letterSpacing: 1.5,
    textTransform: "uppercase",
  },
  heroTitle: {
    fontSize: 20,
    fontWeight: "700",
    color: Colors.textPrimary,
    letterSpacing: -0.3,
  },
  heroSubtitle: {
    fontSize: 12,
    color: Colors.textSecondary,
    lineHeight: 18,
    maxWidth: "80%",
  },
  pressed: {
    opacity: 0.88,
  },
  heroIconArea: {
    paddingLeft: 12,
    alignItems: "center",
    gap: 8,
  },
  heroIconLabel: {
    fontSize: 11,
    fontWeight: "700",
    color: Colors.accent,
    letterSpacing: 0.5,
    textTransform: "uppercase",
  },
  heroIconCircle: {
    width: 72,
    height: 72,
    borderRadius: 36,
    alignItems: "center",
    justifyContent: "center",
  },
  navCard: {
    backgroundColor: Colors.surface,
    borderRadius: 20,
    marginBottom: 28,
    overflow: "hidden",
  },
  navRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingHorizontal: 18,
    paddingVertical: 16,
  },
  navRowLeft: {
    flexDirection: "row",
    alignItems: "center",
    gap: 14,
  },
  navIconBox: {
    width: 44,
    height: 44,
    borderRadius: 12,
    alignItems: "center",
    justifyContent: "center",
  },
  navRowTitle: {
    fontSize: 15,
    fontWeight: "600",
    color: Colors.textPrimary,
  },
  navDivider: {
    height: 1,
    backgroundColor: Colors.border,
    opacity: 0.2,
    marginLeft: 76,
  },
  sectionHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: 12,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: "700",
    color: Colors.textPrimary,
  },
  seeAll: {
    fontSize: 11,
    color: Colors.accent,
    fontWeight: "700",
    letterSpacing: 1,
    textTransform: "uppercase",
  },
  emptyCard: {
    backgroundColor: Colors.surface,
    borderRadius: 20,
    padding: 32,
    alignItems: "center",
    gap: 8,
  },
  emptyIconBox: {
    width: 56,
    height: 56,
    borderRadius: 14,
    backgroundColor: Colors.surfaceElevated,
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 4,
  },
  emptyTitle: {
    fontSize: 15,
    fontWeight: "700",
    color: Colors.textPrimary,
  },
  emptySub: {
    fontSize: 13,
    color: Colors.textMuted,
    textAlign: "center",
    lineHeight: 19,
  },
  emptyAction: {
    marginTop: 8,
  },
}); }
