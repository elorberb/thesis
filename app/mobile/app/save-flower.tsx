import { useState } from "react";
import {
  View,
  Text,
  TextInput,
  Pressable,
  StyleSheet,
  ScrollView,
  KeyboardAvoidingView,
  Platform,
  ActivityIndicator,
  Alert,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { LinearGradient } from "expo-linear-gradient";
import { ApiClient } from "../api/client";
import { useTheme } from "../contexts/ThemeContext";
import { AnalysisResultStore } from "../store/analysisResult";
import { ScreenHeader } from "../components/ScreenHeader";

type Tag = {
  id: string;
  key: string;
  value: string;
};

export default function SaveFlowerScreen() {
  const router = useRouter();
  const { Colors, Gradients } = useTheme();
  const styles = createStyles(Colors);
  const [plantId, setPlantId] = useState("");
  const [tags, setTags] = useState<Tag[]>([]);
  const [saving, setSaving] = useState(false);

  const addTag = () => {
    setTags((prev) => [...prev, { id: Date.now().toString(), key: "", value: "" }]);
  };

  const removeTag = (id: string) => {
    setTags((prev) => prev.filter((t) => t.id !== id));
  };

  const updateTag = (id: string, field: "key" | "value", text: string) => {
    setTags((prev) => prev.map((t) => (t.id === id ? { ...t, [field]: text } : t)));
  };

  const buildMetadata = (): Record<string, string> => {
    const metadata: Record<string, string> = {};
    for (const tag of tags) {
      if (tag.key.trim()) metadata[tag.key.trim()] = tag.value.trim();
    }
    return metadata;
  };

  const linkAndNavigate = async (existingPlantId: string) => {
    const session = AnalysisResultStore.getSession();
    if (session.length === 0) {
      Alert.alert("No scan found", "Please complete a scan before saving.");
      return;
    }
    await Promise.all(
      session.map((analysis) => ApiClient.linkAnalysisToPlant(analysis.id, existingPlantId))
    );
    router.replace("/home");
  };

  const handleSave = async () => {
    if (!plantId.trim()) {
      Alert.alert("Plant ID required", "Please enter a plant ID.");
      return;
    }

    setSaving(true);
    try {
      const matches = await ApiClient.listPlants(plantId.trim());
      const normalizedInput = plantId.trim().toLowerCase();
      const existing = matches.items.find((p) => p.name.toLowerCase() === normalizedInput);

      if (existing) {
        setSaving(false);
        Alert.alert(
          `"${existing.name}" already exists`,
          "Do you want to add this scan to its history?",
          [
            { text: "Cancel", style: "cancel" },
            {
              text: "Add to plant",
              onPress: async () => {
                setSaving(true);
                try {
                  await linkAndNavigate(existing.id);
                } finally {
                  setSaving(false);
                }
              },
            },
          ]
        );
        return;
      }

      const plant = await ApiClient.createPlant({
        name: plantId.trim(),
        metadata: buildMetadata(),
      });
      await linkAndNavigate(plant.id);
    } catch (error) {
      Alert.alert("Save failed", error instanceof Error ? error.message : "Something went wrong.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <SafeAreaView style={styles.safe} edges={["top"]}>
      <KeyboardAvoidingView
        style={styles.flex}
        behavior={Platform.OS === "ios" ? "padding" : "height"}
      >
        <ScreenHeader title="Save Flower" onBack={() => router.back()} />

        <ScrollView
          contentContainerStyle={styles.scroll}
          showsVerticalScrollIndicator={false}
          keyboardShouldPersistTaps="handled"
        >
          <Text style={styles.pageTitle}>Save Flower</Text>
          <Text style={styles.pageSubtitle}>
            Enter a plant ID to track {AnalysisResultStore.getSession().length > 1 ? `these ${AnalysisResultStore.getSession().length} scans` : "this scan"}. If the ID already exists, the scan is added to that plant's history.
          </Text>

          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Plant ID</Text>
            <TextInput
              style={styles.input}
              placeholder="e.g. A3, Tent-2-Row-1, OG Kush #4"
              placeholderTextColor={Colors.textMuted}
              value={plantId}
              onChangeText={setPlantId}
              autoCapitalize="none"
              autoFocus
            />
            <Text style={styles.fieldHint}>
              Used to match or create a plant. Case-insensitive.
            </Text>
          </View>

          <View style={styles.section}>
            <View style={styles.sectionHeader}>
              <Text style={styles.sectionTitle}>Metadata</Text>
              <Pressable style={styles.addTagButton} onPress={addTag}>
                <Ionicons name="add" size={16} color={Colors.accent} />
                <Text style={styles.addTagText}>Add field</Text>
              </Pressable>
            </View>

            {tags.length === 0 ? (
              <Text style={styles.emptyHint}>
                Optional — add grow week, room, nutrient schedule, or any note you want attached to this plant.
              </Text>
            ) : (
              <View style={styles.tagsList}>
                {tags.map((tag) => (
                  <View key={tag.id} style={styles.tagRow}>
                    <TextInput
                      style={[styles.tagInput, styles.tagKeyInput]}
                      placeholder="Field"
                      placeholderTextColor={Colors.textMuted}
                      value={tag.key}
                      onChangeText={(text) => updateTag(tag.id, "key", text)}
                    />
                    <Text style={styles.tagSeparator}>:</Text>
                    <TextInput
                      style={[styles.tagInput, styles.tagValueInput]}
                      placeholder="Value"
                      placeholderTextColor={Colors.textMuted}
                      value={tag.value}
                      onChangeText={(text) => updateTag(tag.id, "value", text)}
                    />
                    <Pressable style={styles.removeTagButton} onPress={() => removeTag(tag.id)}>
                      <Ionicons name="close" size={18} color={Colors.danger} />
                    </Pressable>
                  </View>
                ))}
              </View>
            )}
          </View>
        </ScrollView>

        <View style={styles.footer}>
          <Pressable onPress={handleSave} disabled={saving}>
            {({ pressed }) => (
              <LinearGradient
                colors={Gradients.vitality}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
                style={[styles.saveButton, (pressed || saving) && { opacity: 0.75 }]}
              >
                {saving ? (
                  <ActivityIndicator size="small" color={Colors.accentText} />
                ) : (
                  <Ionicons name="bookmark-outline" size={18} color={Colors.accentText} />
                )}
                <Text style={styles.saveButtonText}>
                  {saving ? "Saving..." : "Save to My Flowers"}
                </Text>
              </LinearGradient>
            )}
          </Pressable>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

function createStyles(Colors: ReturnType<typeof useTheme>["Colors"]) { return StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  flex: {
    flex: 1,
  },
  scroll: {
    padding: 24,
    gap: 28,
    flexGrow: 1,
  },
  pageTitle: {
    fontSize: 26,
    fontWeight: "800",
    color: Colors.textPrimary,
    letterSpacing: -0.5,
  },
  pageSubtitle: {
    fontSize: 14,
    color: Colors.textSecondary,
    lineHeight: 20,
    marginTop: 4,
  },
  section: {
    gap: 10,
  },
  sectionHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: "700",
    color: Colors.textPrimary,
  },
  input: {
    backgroundColor: Colors.surface,
    borderWidth: 1,
    borderColor: Colors.border,
    borderRadius: 12,
    paddingHorizontal: 14,
    paddingVertical: 14,
    fontSize: 15,
    color: Colors.textPrimary,
  },
  fieldHint: {
    fontSize: 12,
    color: Colors.textMuted,
    lineHeight: 17,
  },
  addTagButton: {
    flexDirection: "row",
    alignItems: "center",
    gap: 4,
  },
  addTagText: {
    fontSize: 14,
    fontWeight: "700",
    color: Colors.accent,
  },
  emptyHint: {
    fontSize: 13,
    color: Colors.textMuted,
    lineHeight: 19,
  },
  tagsList: {
    gap: 10,
  },
  tagRow: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: Colors.surface,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: Colors.border,
    paddingHorizontal: 12,
    gap: 8,
  },
  tagInput: {
    height: 48,
    fontSize: 14,
    color: Colors.textPrimary,
  },
  tagKeyInput: {
    flex: 1,
  },
  tagValueInput: {
    flex: 1.5,
  },
  tagSeparator: {
    fontSize: 16,
    fontWeight: "600",
    color: Colors.textMuted,
  },
  removeTagButton: {
    padding: 4,
  },
  footer: {
    padding: 16,
    paddingBottom: 24,
    borderTopWidth: 1,
    borderTopColor: Colors.borderSubtle,
    backgroundColor: Colors.background,
  },
  saveButton: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 16,
    borderRadius: 999,
    gap: 8,
  },
  saveButtonText: {
    fontSize: 15,
    fontWeight: "700",
    color: Colors.accentText,
  },
}); }
