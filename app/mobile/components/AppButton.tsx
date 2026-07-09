import { Pressable, Text, View, StyleSheet, ViewStyle } from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import { Ionicons } from "@expo/vector-icons";
import { useTheme } from "../contexts/ThemeContext";

type Variant = "primary" | "secondary" | "ghost";

type Props = {
  label: string;
  onPress: () => void;
  icon?: keyof typeof Ionicons.glyphMap;
  variant?: Variant;
  fullWidth?: boolean;
  disabled?: boolean;
  style?: ViewStyle;
};

export function AppButton({
  label,
  onPress,
  icon,
  variant = "primary",
  fullWidth = true,
  disabled = false,
  style,
}: Props) {
  const { Colors, Gradients } = useTheme();
  const styles = createStyles(Colors);

  if (variant === "primary") {
    return (
      <Pressable
        onPress={onPress}
        disabled={disabled}
        accessibilityRole="button"
        accessibilityLabel={label}
        accessibilityState={{ disabled }}
        style={[fullWidth && styles.fullWidth, style]}
      >
        {({ pressed }) =>
          disabled ? (
            <View style={[styles.base, styles.disabled]}>
              {icon && <Ionicons name={icon} size={18} color={Colors.textMuted} />}
              <Text style={[styles.label, styles.labelDisabled]}>{label}</Text>
            </View>
          ) : (
            <LinearGradient
              colors={Gradients.vitality}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 1 }}
              style={[styles.base, pressed && styles.pressed]}
            >
              {icon && <Ionicons name={icon} size={18} color={Colors.accentText} />}
              <Text style={[styles.label, styles.labelPrimary]}>{label}</Text>
            </LinearGradient>
          )
        }
      </Pressable>
    );
  }

  const isSecondary = variant === "secondary";
  const foreground = disabled ? Colors.textMuted : isSecondary ? Colors.textPrimary : Colors.accent;

  return (
    <Pressable
      onPress={onPress}
      disabled={disabled}
      accessibilityRole="button"
      accessibilityLabel={label}
      accessibilityState={{ disabled }}
      style={({ pressed }) => [
        fullWidth && styles.fullWidth,
        styles.base,
        isSecondary ? styles.secondary : styles.ghost,
        pressed && styles.pressed,
        disabled && styles.disabled,
        style,
      ]}
    >
      {icon && <Ionicons name={icon} size={18} color={foreground} />}
      <Text style={[styles.label, { color: foreground }]}>{label}</Text>
    </Pressable>
  );
}

function createStyles(Colors: ReturnType<typeof useTheme>["Colors"]) {
  return StyleSheet.create({
    fullWidth: {
      width: "100%",
    },
    base: {
      flexDirection: "row",
      alignItems: "center",
      justifyContent: "center",
      paddingVertical: 15,
      paddingHorizontal: 24,
      borderRadius: 999,
      gap: 8,
    },
    pressed: {
      opacity: 0.88,
    },
    disabled: {
      backgroundColor: Colors.surfaceElevated,
    },
    secondary: {
      backgroundColor: Colors.surface,
      borderWidth: 1,
      borderColor: Colors.border,
    },
    ghost: {
      backgroundColor: "transparent",
    },
    label: {
      fontSize: 15,
      fontWeight: "700",
      letterSpacing: 0.3,
    },
    labelPrimary: {
      color: Colors.accentText,
    },
    labelDisabled: {
      color: Colors.textMuted,
    },
  });
}
