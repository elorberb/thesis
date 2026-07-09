import { View, Text, ActivityIndicator, StyleSheet } from "react-native";
import { useTheme } from "../contexts/ThemeContext";

type Props = {
  message?: string;
};

export function LoadingState({ message = "Loading…" }: Props) {
  const { Colors } = useTheme();
  const styles = createStyles(Colors);

  return (
    <View style={styles.container}>
      <ActivityIndicator size="large" color={Colors.accent} />
      <Text style={styles.message}>{message}</Text>
    </View>
  );
}

function createStyles(Colors: ReturnType<typeof useTheme>["Colors"]) {
  return StyleSheet.create({
    container: {
      flex: 1,
      alignItems: "center",
      justifyContent: "center",
      gap: 12,
    },
    message: {
      fontSize: 14,
      color: Colors.textMuted,
    },
  });
}
