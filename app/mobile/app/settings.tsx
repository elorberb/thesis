import { View, Text, Pressable, StyleSheet, ScrollView, Alert, Switch } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useRouter } from "expo-router";
import { useTheme } from "../contexts/ThemeContext";
import { useAuth } from "../contexts/AuthContext";
import { ScreenHeader } from "../components/ScreenHeader";

function SectionLabel({ label, colors }: { label: string; colors: ReturnType<typeof useTheme>["Colors"] }) {
  return <Text style={{ fontSize: 11, fontWeight: "700", color: colors.textMuted, letterSpacing: 1, marginBottom: 8, marginTop: 4 }}>{label}</Text>;
}

export default function SettingsScreen() {
  const router = useRouter();
  const { Colors, scheme, toggleTheme } = useTheme();
  const { signOut } = useAuth();
  const styles = createStyles(Colors);

  const handleClearData = () => {
    Alert.alert(
      "Clear Local Data",
      "This will remove all locally cached data. This cannot be undone.",
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Clear",
          style: "destructive",
          onPress: () => Alert.alert("Done", "Local data cleared."),
        },
      ]
    );
  };

  const handleSignOut = () => {
    Alert.alert("Sign Out", "Are you sure you want to sign out?", [
      { text: "Cancel", style: "cancel" },
      {
        text: "Sign Out",
        style: "destructive",
        onPress: async () => {
          await signOut();
          router.replace("/");
        },
      },
    ]);
  };

  return (
    <SafeAreaView style={styles.safe}>
      <ScreenHeader title="Settings" onBack={() => router.back()} />

      <ScrollView contentContainerStyle={styles.scroll} showsVerticalScrollIndicator={false}>
        <SectionLabel label="APPEARANCE" colors={Colors} />
        <View style={styles.card}>
          <View style={styles.row}>
            <Text style={styles.rowLabel}>Light Mode</Text>
            <Switch
              value={scheme === "light"}
              onValueChange={toggleTheme}
              trackColor={{ false: Colors.border, true: Colors.accent }}
              thumbColor={Colors.surface}
            />
          </View>
        </View>

        <SectionLabel label="ACCOUNT" colors={Colors} />
        <View style={styles.card}>
          <Pressable style={styles.row} onPress={handleSignOut}>
            <Text style={[styles.rowLabel, styles.rowLabelDanger]}>Sign Out</Text>
            <Text style={styles.rowChevron}>›</Text>
          </Pressable>
        </View>

        <SectionLabel label="DATA" colors={Colors} />
        <View style={styles.card}>
          <Pressable style={styles.row} onPress={handleClearData}>
            <Text style={[styles.rowLabel, styles.rowLabelDanger]}>Clear Local Data</Text>
            <Text style={styles.rowChevron}>›</Text>
          </Pressable>
        </View>

        <Text style={styles.version}>LoupeLab · v0.1.0</Text>
      </ScrollView>
    </SafeAreaView>
  );
}

function createStyles(Colors: ReturnType<typeof useTheme>["Colors"]) {
  return StyleSheet.create({
    safe: { flex: 1, backgroundColor: Colors.background },
    scroll: { paddingHorizontal: 20, paddingTop: 24, paddingBottom: 48 },
    card: {
      backgroundColor: Colors.surface,
      borderRadius: 14,
      borderWidth: 1,
      borderColor: Colors.border,
      marginBottom: 24,
      overflow: "hidden",
    },
    row: {
      flexDirection: "row",
      alignItems: "center",
      justifyContent: "space-between",
      paddingHorizontal: 16,
      paddingVertical: 16,
    },
    rowLabel: { fontSize: 15, color: Colors.textPrimary, fontWeight: "500" },
    rowLabelDanger: { color: Colors.danger },
    rowChevron: { fontSize: 20, color: Colors.textMuted },
    version: { textAlign: "center", fontSize: 12, color: Colors.textMuted, marginTop: 8 },
  });
}
