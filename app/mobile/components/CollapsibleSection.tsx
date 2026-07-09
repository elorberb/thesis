import { useState, ReactNode } from "react";
import { View, Text, Pressable, StyleSheet, LayoutAnimation, Platform, UIManager } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { useTheme } from "../contexts/ThemeContext";

if (Platform.OS === "android" && UIManager.setLayoutAnimationEnabledExperimental) {
  UIManager.setLayoutAnimationEnabledExperimental(true);
}

type Props = {
  title: string;
  subtitle?: string;
  defaultOpen?: boolean;
  children: ReactNode;
};

export function CollapsibleSection({ title, subtitle, defaultOpen = false, children }: Props) {
  const { Colors } = useTheme();
  const styles = createStyles(Colors);
  const [open, setOpen] = useState(defaultOpen);

  const toggle = () => {
    LayoutAnimation.configureNext(LayoutAnimation.Presets.easeInEaseOut);
    setOpen((value) => !value);
  };

  return (
    <View style={styles.wrap}>
      <Pressable
        style={styles.header}
        onPress={toggle}
        accessibilityRole="button"
        accessibilityLabel={title}
        accessibilityState={{ expanded: open }}
        hitSlop={6}
      >
        <View style={styles.headerText}>
          <Text style={styles.title}>{title}</Text>
          {subtitle ? <Text style={styles.subtitle}>{subtitle}</Text> : null}
        </View>
        <Ionicons name={open ? "chevron-up" : "chevron-down"} size={18} color={Colors.textMuted} />
      </Pressable>
      {open ? <View style={styles.body}>{children}</View> : null}
    </View>
  );
}

function createStyles(Colors: ReturnType<typeof useTheme>["Colors"]) {
  return StyleSheet.create({
    wrap: {
      backgroundColor: Colors.surface,
      borderRadius: 16,
      borderWidth: 1,
      borderColor: Colors.border,
      overflow: "hidden",
    },
    header: {
      flexDirection: "row",
      alignItems: "center",
      justifyContent: "space-between",
      paddingHorizontal: 16,
      paddingVertical: 14,
      gap: 12,
    },
    headerText: {
      flex: 1,
      gap: 2,
    },
    title: {
      fontSize: 15,
      fontWeight: "700",
      color: Colors.textPrimary,
    },
    subtitle: {
      fontSize: 12,
      color: Colors.textMuted,
    },
    body: {
      paddingHorizontal: 12,
      paddingBottom: 12,
      paddingTop: 2,
      gap: 12,
    },
  });
}
