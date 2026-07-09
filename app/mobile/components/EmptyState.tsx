import { View, Text, StyleSheet } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { useTheme } from "../contexts/ThemeContext";
import { AppButton } from "./AppButton";

type Props = {
  icon: keyof typeof Ionicons.glyphMap;
  title: string;
  message?: string;
  actionLabel?: string;
  actionIcon?: keyof typeof Ionicons.glyphMap;
  onAction?: () => void;
};

export function EmptyState({ icon, title, message, actionLabel, actionIcon, onAction }: Props) {
  const { Colors } = useTheme();
  const styles = createStyles(Colors);

  return (
    <View style={styles.container}>
      <View style={styles.iconCircle}>
        <Ionicons name={icon} size={32} color={Colors.accent} />
      </View>
      <Text style={styles.title}>{title}</Text>
      {message ? <Text style={styles.message}>{message}</Text> : null}
      {actionLabel && onAction ? (
        <View style={styles.action}>
          <AppButton label={actionLabel} icon={actionIcon} onPress={onAction} fullWidth={false} />
        </View>
      ) : null}
    </View>
  );
}

function createStyles(Colors: ReturnType<typeof useTheme>["Colors"]) {
  return StyleSheet.create({
    container: {
      flex: 1,
      alignItems: "center",
      justifyContent: "center",
      paddingHorizontal: 40,
      gap: 10,
    },
    iconCircle: {
      width: 80,
      height: 80,
      borderRadius: 24,
      backgroundColor: Colors.accentSurface,
      alignItems: "center",
      justifyContent: "center",
      marginBottom: 6,
    },
    title: {
      fontSize: 19,
      fontWeight: "700",
      color: Colors.textPrimary,
      textAlign: "center",
    },
    message: {
      fontSize: 14,
      color: Colors.textMuted,
      textAlign: "center",
      lineHeight: 22,
    },
    action: {
      marginTop: 8,
    },
  });
}
