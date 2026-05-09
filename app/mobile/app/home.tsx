import { useState, useCallback } from "react";
import { View, Text, Pressable, StyleSheet, ScrollView, Image, ActivityIndicator } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useRouter } from "expo-router";
import { useFocusEffect } from "@react-navigation/native";
import { Ionicons } from "@expo/vector-icons";
import { LinearGradient } from "expo-linear-gradient";
import { MaturityBadge } from "../components/MaturityBadge";
import { ApiClient } from "../api/client";
import { AnalysisListItem } from "../api/types";
import { useTheme } from "../contexts/ThemeContext";
import { useAuth } from "../contexts/AuthContext";

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
              <Ionicons name="leaf" size={18} color={Colors.accent} />
              <Text style={styles.brandName}>AGRIVISION</Text>
            </View>
            <Pressable style={styles.profileButton} onPress={() => router.push("/profile")}>
              <Text style={styles.profileInitial}>
                {(user?.user_metadata?.full_name as string | undefined)?.[0]?.toUpperCase() ??
                  user?.email?.[0]?.toUpperCase() ?? "?"}
              </Text>
            </Pressable>
          </View>

          <Pressable style={styles.heroCard} onPress={() => router.push("/camera")}>
            <View style={styles.heroBlobBg} />
            <View style={styles.heroContent}>
              <View style={styles.heroBadge}>
                <Text style={styles.heroBadgeText}>NEW ANALYSIS</Text>
              </View>
              <Text style={styles.heroTitle}>Analyze Your Flower</Text>
              <Text style={styles.heroSubtitle}>
                Get instant insights on cannabinoid maturity and harvest timing.
              </Text>
              <Pressable onPress={() => router.push("/camera")}>
                {({ pressed }) => (
                  <LinearGradient
                    colors={Gradients.vitality}
                    start={{ x: 0, y: 0 }}
                    end={{ x: 1, y: 1 }}
                    style={[styles.heroButton, pressed && styles.pressed]}
                  >
                    <Ionicons name="camera" size={16} color={Colors.accentText} />
                    <Text style={styles.heroButtonText}>Start Scan</Text>
                  </LinearGradient>
                )}
              </Pressable>
            </View>
            <View style={styles.heroIconArea}>
              <View style={styles.heroIconRing}>
                <LinearGradient
                  colors={Gradients.vitality}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 1 }}
                  style={styles.heroIconCircle}
                >
                  <Ionicons name="camera" size={28} color={Colors.accentText} />
                </LinearGradient>
              </View>
            </View>
          </Pressable>

          <View style={styles.navCard}>
            <Pressable style={styles.navRow} onPress={() => router.push("/history")}>
              <View style={styles.navRowLeft}>
                <View style={[styles.navIconBox, { backgroundColor: "rgba(122,251,183,0.12)" }]}>
                  <Ionicons name="folder" size={20} color={Colors.accent} />
                </View>
                <Text style={styles.navRowTitle}>My Plants</Text>
              </View>
              <Ionicons name="chevron-forward" size={18} color={Colors.textMuted} />
            </Pressable>

            <View style={styles.navDivider} />

            <Pressable style={styles.navRow} onPress={() => router.push("/how-it-works")}>
              <View style={styles.navRowLeft}>
                <View style={[styles.navIconBox, { backgroundColor: "rgba(125,233,255,0.10)" }]}>
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
              <Pressable onPress={() => router.push("/camera")}>
                {({ pressed }) => (
                  <LinearGradient
                    colors={Gradients.vitality}
                    start={{ x: 0, y: 0 }}
                    end={{ x: 1, y: 1 }}
                    style={[styles.emptyAction, pressed && styles.pressed]}
                  >
                    <Text style={styles.emptyActionText}>Start First Scan</Text>
                  </LinearGradient>
                )}
              </Pressable>
            </View>
          ) : (
            recentAnalyses.map((item) => {
              const date = new Date(item.created_at).toLocaleDateString("en-US", {
                month: "short",
                day: "numeric",
              });
              return (
                <Pressable
                  key={item.id}
                  style={styles.recentCard}
                  onPress={() => router.push({ pathname: "/results", params: { id: item.id } })}
                >
                  {item.annotated_image_url ? (
                    <Image source={{ uri: item.annotated_image_url }} style={styles.recentThumbnail} resizeMode="cover" />
                  ) : (
                    <View style={styles.recentThumbnailPlaceholder}>
                      <Ionicons name="leaf-outline" size={22} color={Colors.textMuted} />
                    </View>
                  )}
                  <View style={styles.recentContent}>
                    <View style={styles.recentTopRow}>
                      <Text style={styles.recentDate}>{date}</Text>
                      <MaturityBadge stage={item.maturity_stage} size="sm" />
                    </View>
                    <Text style={styles.recentRec} numberOfLines={2}>
                      {item.recommendation}
                    </Text>
                  </View>
                  <Ionicons name="chevron-forward" size={16} color={Colors.textMuted} />
                </Pressable>
              );
            })
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
    gap: 8,
  },
  brandName: {
    fontSize: 18,
    fontWeight: "800",
    color: Colors.accent,
    letterSpacing: 3,
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
  heroBlobBg: {
    position: "absolute",
    right: -30,
    top: -30,
    width: 160,
    height: 160,
    borderRadius: 80,
    backgroundColor: Colors.accent,
    opacity: 0.04,
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
  heroButton: {
    flexDirection: "row",
    alignItems: "center",
    alignSelf: "flex-start",
    paddingHorizontal: 16,
    paddingVertical: 9,
    borderRadius: 999,
    gap: 6,
  },
  pressed: {
    opacity: 0.88,
  },
  heroButtonText: {
    fontSize: 13,
    fontWeight: "700",
    color: Colors.accentText,
    letterSpacing: 0.3,
  },
  heroIconArea: {
    paddingLeft: 12,
  },
  heroIconRing: {
    width: 76,
    height: 76,
    borderRadius: 38,
    borderWidth: 1,
    borderColor: "rgba(107,255,143,0.2)",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(23,31,54,0.5)",
  },
  heroIconCircle: {
    width: 58,
    height: 58,
    borderRadius: 29,
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
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 999,
  },
  emptyActionText: {
    fontSize: 13,
    fontWeight: "700",
    color: Colors.accentText,
  },
  recentCard: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: Colors.surface,
    borderRadius: 16,
    padding: 14,
    marginBottom: 10,
    gap: 14,
  },
  recentThumbnail: {
    width: 52,
    height: 52,
    borderRadius: 12,
    backgroundColor: Colors.surfaceHighest,
  },
  recentThumbnailPlaceholder: {
    width: 52,
    height: 52,
    borderRadius: 12,
    backgroundColor: Colors.surfaceHighest,
    alignItems: "center",
    justifyContent: "center",
  },
  recentContent: {
    flex: 1,
    gap: 6,
  },
  recentTopRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  recentDate: {
    fontSize: 10,
    color: Colors.textMuted,
    fontWeight: "700",
    letterSpacing: 0.5,
    textTransform: "uppercase",
  },
  recentRec: {
    fontSize: 12,
    color: Colors.textSecondary,
    lineHeight: 17,
  },
}); }
