import { View, Text, Pressable, StyleSheet, Alert } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { useTheme } from "../contexts/ThemeContext";
import { useAuth } from "../contexts/AuthContext";
import { ScreenHeader } from "../components/ScreenHeader";

function getInitials(name: string | undefined, email: string | undefined): string {
  if (name) {
    const parts = name.trim().split(" ");
    return parts.length >= 2
      ? (parts[0][0] + parts[parts.length - 1][0]).toUpperCase()
      : parts[0].slice(0, 2).toUpperCase();
  }
  if (email) return email[0].toUpperCase();
  return "?";
}

export default function ProfileScreen() {
  const router = useRouter();
  const { Colors } = useTheme();
  const { user, signOut } = useAuth();
  const styles = createStyles(Colors);

  const displayName = user?.user_metadata?.full_name as string | undefined;
  const email = user?.email;
  const initials = getInitials(displayName, email);
  const joinedDate = user?.created_at
    ? new Date(user.created_at).toLocaleDateString("en-US", { month: "long", year: "numeric" })
    : null;

  const handleSignOut = () => {
    Alert.alert("Sign out", "Are you sure you want to sign out?", [
      { text: "Cancel", style: "cancel" },
      {
        text: "Sign out",
        style: "destructive",
        onPress: async () => {
          await signOut();
          router.replace("/");
        },
      },
    ]);
  };

  return (
    <SafeAreaView style={styles.safe} edges={["top"]}>
      <ScreenHeader title="Profile" onBack={() => router.back()} />

      <View style={styles.avatarSection}>
        <View style={styles.avatarCircle}>
          <Text style={styles.avatarText}>{initials}</Text>
        </View>
        {displayName ? (
          <Text style={styles.displayName}>{displayName}</Text>
        ) : null}
        <Text style={styles.emailText}>{email}</Text>
        {joinedDate ? (
          <Text style={styles.joinedText}>Member since {joinedDate}</Text>
        ) : null}
      </View>

      <View style={styles.section}>
        <Pressable style={styles.row} onPress={() => router.push("/settings")}>
          <View style={styles.rowLeft}>
            <View style={[styles.rowIcon, { backgroundColor: Colors.accentSurface }]}>
              <Ionicons name="settings-outline" size={18} color={Colors.accent} />
            </View>
            <Text style={styles.rowLabel}>Settings</Text>
          </View>
          <Ionicons name="chevron-forward" size={16} color={Colors.textMuted} />
        </Pressable>

        <View style={styles.rowDivider} />

        <Pressable style={styles.row} onPress={() => router.push("/my-plants")}>
          <View style={styles.rowLeft}>
            <View style={[styles.rowIcon, { backgroundColor: Colors.accentSurface }]}>
              <Ionicons name="leaf-outline" size={18} color={Colors.accent} />
            </View>
            <Text style={styles.rowLabel}>My Plants</Text>
          </View>
          <Ionicons name="chevron-forward" size={16} color={Colors.textMuted} />
        </Pressable>
      </View>

      <View style={[styles.section, { marginTop: 12 }]}>
        <Pressable style={styles.row} onPress={handleSignOut}>
          <View style={styles.rowLeft}>
            <View style={[styles.rowIcon, { backgroundColor: Colors.dangerSurface }]}>
              <Ionicons name="log-out-outline" size={18} color={Colors.danger} />
            </View>
            <Text style={[styles.rowLabel, { color: Colors.danger }]}>Sign Out</Text>
          </View>
        </Pressable>
      </View>
    </SafeAreaView>
  );
}

function createStyles(Colors: ReturnType<typeof useTheme>["Colors"]) {
  return StyleSheet.create({
    safe: {
      flex: 1,
      backgroundColor: Colors.background,
    },
    avatarSection: {
      alignItems: "center",
      paddingVertical: 32,
      gap: 6,
    },
    avatarCircle: {
      width: 80,
      height: 80,
      borderRadius: 40,
      backgroundColor: Colors.accentSurface,
      borderWidth: 2,
      borderColor: Colors.accent,
      alignItems: "center",
      justifyContent: "center",
      marginBottom: 8,
    },
    avatarText: {
      fontSize: 28,
      fontWeight: "700",
      color: Colors.accent,
    },
    displayName: {
      fontSize: 20,
      fontWeight: "700",
      color: Colors.textPrimary,
    },
    emailText: {
      fontSize: 14,
      color: Colors.textSecondary,
    },
    joinedText: {
      fontSize: 12,
      color: Colors.textMuted,
      marginTop: 2,
    },
    section: {
      marginHorizontal: 16,
      backgroundColor: Colors.surface,
      borderRadius: 16,
      borderWidth: 1,
      borderColor: Colors.border,
      overflow: "hidden",
    },
    row: {
      flexDirection: "row",
      alignItems: "center",
      justifyContent: "space-between",
      paddingHorizontal: 16,
      paddingVertical: 14,
    },
    rowLeft: {
      flexDirection: "row",
      alignItems: "center",
      gap: 12,
    },
    rowIcon: {
      width: 34,
      height: 34,
      borderRadius: 10,
      alignItems: "center",
      justifyContent: "center",
    },
    rowLabel: {
      fontSize: 15,
      fontWeight: "600",
      color: Colors.textPrimary,
    },
    rowDivider: {
      height: 1,
      backgroundColor: Colors.borderSubtle,
      marginLeft: 62,
    },
  });
}
