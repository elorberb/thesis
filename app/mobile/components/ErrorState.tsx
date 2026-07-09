import { View, Text, StyleSheet } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { useTheme } from "../contexts/ThemeContext";
import { AppButton } from "./AppButton";

type Props = {
  message: string;
  title?: string;
  onRetry?: () => void;
};

export function ErrorState({ message, title = "Something went wrong", onRetry }: Props) {
  const { Colors } = useTheme();
  const styles = createStyles(Colors);

  return (
    <View style={styles.container}>
      <Ionicons name="warning-outline" size={40} color={Colors.danger} />
      <Text style={styles.title}>{title}</Text>
      <Text style={styles.message}>{message}</Text>
      {onRetry ? (
        <View style={styles.action}>
          <AppButton label="Retry" icon="refresh" variant="secondary" onPress={onRetry} fullWidth={false} />
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
    title: {
      fontSize: 17,
      fontWeight: "700",
      color: Colors.textPrimary,
      textAlign: "center",
    },
    message: {
      fontSize: 14,
      color: Colors.textMuted,
      textAlign: "center",
      lineHeight: 20,
    },
    action: {
      marginTop: 8,
    },
  });
}
