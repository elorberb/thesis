import { ReactNode } from "react";
import { View, Text, Pressable, StyleSheet } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { useTheme } from "../contexts/ThemeContext";

type Props = {
  title: string;
  subtitle?: string;
  onBack?: () => void;
  right?: ReactNode;
};

export function ScreenHeader({ title, subtitle, onBack, right }: Props) {
  const { Colors } = useTheme();
  const styles = createStyles(Colors);

  return (
    <View style={styles.header}>
      <View style={styles.side}>
        {onBack && (
          <Pressable
            style={styles.backButton}
            onPress={onBack}
            accessibilityRole="button"
            accessibilityLabel="Go back"
            hitSlop={10}
          >
            <Ionicons name="arrow-back" size={22} color={Colors.textPrimary} />
          </Pressable>
        )}
      </View>
      <View style={styles.titleWrap}>
        <Text style={styles.title} numberOfLines={1}>
          {title}
        </Text>
        {subtitle ? (
          <Text style={styles.subtitle} numberOfLines={1}>
            {subtitle}
          </Text>
        ) : null}
      </View>
      <View style={[styles.side, styles.sideRight]}>{right}</View>
    </View>
  );
}

function createStyles(Colors: ReturnType<typeof useTheme>["Colors"]) {
  return StyleSheet.create({
    header: {
      flexDirection: "row",
      alignItems: "center",
      paddingHorizontal: 16,
      paddingTop: 8,
      paddingBottom: 12,
    },
    side: {
      width: 44,
      justifyContent: "center",
    },
    sideRight: {
      alignItems: "flex-end",
    },
    backButton: {
      width: 44,
      height: 44,
      justifyContent: "center",
    },
    titleWrap: {
      flex: 1,
      alignItems: "center",
    },
    title: {
      fontSize: 17,
      fontWeight: "700",
      color: Colors.textPrimary,
    },
    subtitle: {
      fontSize: 12,
      color: Colors.textMuted,
      marginTop: 1,
    },
  });
}
