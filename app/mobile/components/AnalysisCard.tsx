import { View, Text, Image, Pressable, StyleSheet } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { MaturityBadge } from "./MaturityBadge";
import { useTheme } from "../contexts/ThemeContext";
import { AnalysisListItem } from "../api/types";

type Props = {
  analysis: AnalysisListItem;
  onPress: () => void;
  onDelete?: () => void;
  showTime?: boolean;
};

export function AnalysisCard({ analysis, onPress, onDelete, showTime = false }: Props) {
  const { Colors } = useTheme();
  const styles = createStyles(Colors);

  const created = new Date(analysis.created_at);
  const dateLabel = created.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
  const timeLabel = created.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" });
  const thumbnail = analysis.annotated_image_url ?? analysis.image_url;

  return (
    <Pressable
      style={({ pressed }) => [styles.card, pressed && styles.cardPressed]}
      onPress={onPress}
      accessibilityRole="button"
      accessibilityLabel={`Analysis from ${dateLabel}`}
    >
      {thumbnail ? (
        <Image source={{ uri: thumbnail }} style={styles.thumbnail} resizeMode="cover" />
      ) : (
        <View style={styles.thumbnailPlaceholder}>
          <Ionicons name="leaf-outline" size={22} color={Colors.textMuted} />
        </View>
      )}

      <View style={styles.content}>
        <View style={styles.topRow}>
          <Text style={styles.date}>{dateLabel}</Text>
          <MaturityBadge stage={analysis.maturity_stage} size="sm" />
        </View>
        {showTime ? <Text style={styles.time}>{timeLabel}</Text> : null}
        <Text style={styles.recommendation} numberOfLines={2}>
          {analysis.recommendation}
        </Text>
      </View>

      {onDelete ? (
        <Pressable
          style={styles.deleteButton}
          onPress={onDelete}
          hitSlop={12}
          accessibilityRole="button"
          accessibilityLabel="Delete analysis"
        >
          <Ionicons name="trash-outline" size={18} color={Colors.textMuted} />
        </Pressable>
      ) : (
        <Ionicons name="chevron-forward" size={18} color={Colors.textMuted} style={styles.chevron} />
      )}
    </Pressable>
  );
}

function createStyles(Colors: ReturnType<typeof useTheme>["Colors"]) {
  return StyleSheet.create({
    card: {
      flexDirection: "row",
      alignItems: "center",
      backgroundColor: Colors.surface,
      borderRadius: 16,
      padding: 12,
      marginBottom: 10,
      gap: 12,
    },
    cardPressed: {
      opacity: 0.75,
    },
    thumbnail: {
      width: 56,
      height: 56,
      borderRadius: 12,
      backgroundColor: Colors.surfaceElevated,
    },
    thumbnailPlaceholder: {
      width: 56,
      height: 56,
      borderRadius: 12,
      backgroundColor: Colors.surfaceElevated,
      alignItems: "center",
      justifyContent: "center",
    },
    content: {
      flex: 1,
      gap: 4,
    },
    topRow: {
      flexDirection: "row",
      alignItems: "center",
      justifyContent: "space-between",
    },
    date: {
      fontSize: 14,
      fontWeight: "700",
      color: Colors.textPrimary,
    },
    time: {
      fontSize: 11,
      color: Colors.textMuted,
    },
    recommendation: {
      fontSize: 12,
      color: Colors.textSecondary,
      lineHeight: 17,
    },
    deleteButton: {
      width: 44,
      height: 44,
      alignItems: "center",
      justifyContent: "center",
    },
    chevron: {
      marginLeft: 4,
    },
  });
}
