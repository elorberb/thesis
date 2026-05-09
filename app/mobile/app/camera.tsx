import { useState } from "react";
import {
  View,
  Text,
  Pressable,
  Image,
  StyleSheet,
  Alert,
  ActivityIndicator,
  ScrollView,
  Dimensions,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useRouter } from "expo-router";
import * as ImagePicker from "expo-image-picker";
import * as ImageManipulator from "expo-image-manipulator";
import { Colors } from "../constants/theme";
import { ApiClient } from "../api/client";
import { AnalysisResultStore } from "../store/analysisResult";

const SCREEN_WIDTH = Dimensions.get("window").width;
const GRID_PADDING = 20;
const GRID_GAP = 8;
const CELL_SIZE = (SCREEN_WIDTH - GRID_PADDING * 2 - GRID_GAP * 2) / 3;

export default function CameraScreen() {
  const router = useRouter();
  const [images, setImages] = useState<string[]>([]);
  const [analyzing, setAnalyzing] = useState(false);
  const [progress, setProgress] = useState<{ current: number; total: number } | null>(null);
  const [currentAnalyzingUri, setCurrentAnalyzingUri] = useState<string | null>(null);

  const takePhoto = async () => {
    const permission = await ImagePicker.requestCameraPermissionsAsync();
    if (!permission.granted) {
      Alert.alert(
        "Permission required",
        "Camera access is needed to take photos."
      );
      return;
    }
    const result = await ImagePicker.launchCameraAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 1,
    });
    if (!result.canceled) {
      setImages((prev) => [...prev, result.assets[0].uri]);
    }
  };

  const chooseFromGallery = async () => {
    const permission =
      await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!permission.granted) {
      Alert.alert(
        "Permission required",
        "Gallery access is needed to pick photos."
      );
      return;
    }
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      quality: 1,
      allowsMultipleSelection: true,
    });
    if (!result.canceled) {
      setImages((prev) => [...prev, ...result.assets.map((a) => a.uri)]);
    }
  };

  const removeImage = (index: number) => {
    setImages((prev) => prev.filter((_, i) => i !== index));
  };

  const analyze = async () => {
    if (images.length === 0) return;
    setAnalyzing(true);
    const results = [];
    try {
      for (let i = 0; i < images.length; i++) {
        setCurrentAnalyzingUri(images[i]);
        setProgress({ current: i + 1, total: images.length });
        const resized = await ImageManipulator.manipulateAsync(
          images[i],
          [{ resize: { width: 1200 } }],
          { compress: 0.85, format: ImageManipulator.SaveFormat.JPEG }
        );
        const result = await ApiClient.analyzeImage(resized.uri, "local-dev");
        results.push(result);
      }
      AnalysisResultStore.setSession(results);
      router.push("/results");
    } catch (error) {
      Alert.alert(
        "Analysis failed",
        error instanceof Error ? error.message : "Could not reach the server. Check your connection."
      );
    } finally {
      setAnalyzing(false);
      setProgress(null);
      setCurrentAnalyzingUri(null);
    }
  };

  const analyzeLabel =
    images.length === 0
      ? "Add photos to analyze"
      : `Analyze ${images.length} Photo${images.length > 1 ? "s" : ""}`;

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.header}>
        <Pressable onPress={() => router.back()}>
          <Text style={styles.backText}>← Back</Text>
        </Pressable>
        <Text style={styles.headerTitle}>New Analysis</Text>
        <View style={styles.headerSpacer} />
      </View>

      <View style={styles.tipBanner}>
        <View style={styles.tipAccent} />
        <Text style={styles.tipText}>
          Hold your phone 2–3 cm from the trichomes for best results.
        </Text>
      </View>

      <ScrollView
        style={styles.imageArea}
        contentContainerStyle={styles.imageAreaContent}
        showsVerticalScrollIndicator={false}
      >
        {images.length === 0 ? (
          <View style={styles.emptyImageArea}>
            <View style={styles.emptyIconPlaceholder} />
            <Text style={styles.emptyTitle}>No photos added</Text>
            <Text style={styles.emptySubtitle}>
              Use the buttons below to capture or select macro images.
            </Text>
          </View>
        ) : (
          <View style={styles.imageGrid}>
            {images.map((uri, index) => (
              <View key={index} style={styles.imageCell}>
                <Image source={{ uri }} style={styles.thumbnail} />
                <Pressable
                  style={styles.removeButton}
                  onPress={() => removeImage(index)}
                >
                  <Text style={styles.removeButtonText}>×</Text>
                </Pressable>
              </View>
            ))}
          </View>
        )}
      </ScrollView>

      <View style={styles.bottomArea}>
        <View style={styles.captureRow}>
          <Pressable style={styles.captureButton} onPress={takePhoto}>
            <Text style={styles.captureLabel}>Take Photo</Text>
          </Pressable>
          <Pressable style={styles.captureButton} onPress={chooseFromGallery}>
            <Text style={styles.captureLabel}>Choose from Gallery</Text>
          </Pressable>
        </View>

        <Pressable
          style={[
            styles.analyzeButton,
            (images.length === 0 || analyzing) && styles.analyzeButtonDisabled,
          ]}
          disabled={images.length === 0 || analyzing}
          onPress={analyze}
        >
          <Text
            style={[
              styles.analyzeButtonText,
              (images.length === 0 || analyzing) && styles.analyzeButtonTextDisabled,
            ]}
          >
            {analyzing ? "Analyzing…" : analyzeLabel}
          </Text>
        </Pressable>
      </View>

      {analyzing && (
        <View style={styles.analyzingOverlay}>
          <View style={styles.analyzingCard}>
            {currentAnalyzingUri && (
              <Image
                source={{ uri: currentAnalyzingUri }}
                style={styles.analyzingThumb}
                resizeMode="cover"
              />
            )}
            <Text style={styles.analyzingCounter}>
              {progress ? `${progress.current} / ${progress.total}` : "…"}
            </Text>
            <Text style={styles.analyzingSubtitle}>
              {progress
                ? `Analyzing photo ${progress.current} of ${progress.total}`
                : "Preparing…"}
            </Text>
            {progress && (
              <View style={styles.analyzingTrack}>
                <View
                  style={[
                    styles.analyzingFill,
                    { width: `${Math.round((progress.current / progress.total) * 100)}%` },
                  ]}
                />
              </View>
            )}
            <Text style={styles.analyzingHint}>This may take a moment per photo</Text>
          </View>
        </View>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  header: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 20,
    paddingTop: 8,
    paddingBottom: 12,
  },
  backText: {
    fontSize: 15,
    color: Colors.accent,
    fontWeight: "500",
    width: 60,
  },
  headerTitle: {
    flex: 1,
    textAlign: "center",
    fontSize: 17,
    fontWeight: "700",
    color: Colors.textPrimary,
  },
  headerSpacer: {
    width: 60,
  },
  tipBanner: {
    flexDirection: "row",
    alignItems: "stretch",
    backgroundColor: Colors.surface,
    borderWidth: 1,
    borderColor: Colors.border,
    marginHorizontal: 20,
    borderRadius: 12,
    marginBottom: 16,
    overflow: "hidden",
  },
  tipAccent: {
    width: 3,
    backgroundColor: Colors.warning,
  },
  tipText: {
    flex: 1,
    fontSize: 13,
    color: Colors.textSecondary,
    lineHeight: 19,
    paddingVertical: 12,
    paddingHorizontal: 14,
  },
  imageArea: {
    flex: 1,
    paddingHorizontal: GRID_PADDING,
  },
  imageAreaContent: {
    flexGrow: 1,
  },
  emptyImageArea: {
    flex: 1,
    minHeight: 200,
    backgroundColor: Colors.surface,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: Colors.border,
    alignItems: "center",
    justifyContent: "center",
    padding: 32,
    gap: 8,
  },
  emptyIconPlaceholder: {
    width: 48,
    height: 48,
    borderRadius: 12,
    backgroundColor: Colors.surfaceElevated,
    borderWidth: 1,
    borderColor: Colors.border,
    marginBottom: 8,
  },
  emptyTitle: {
    fontSize: 15,
    fontWeight: "700",
    color: Colors.textSecondary,
  },
  emptySubtitle: {
    fontSize: 13,
    color: Colors.textMuted,
    textAlign: "center",
    lineHeight: 19,
  },
  imageGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: GRID_GAP,
  },
  imageCell: {
    width: CELL_SIZE,
    height: CELL_SIZE,
    borderRadius: 10,
    overflow: "visible",
  },
  thumbnail: {
    width: CELL_SIZE,
    height: CELL_SIZE,
    borderRadius: 10,
  },
  removeButton: {
    position: "absolute",
    top: -8,
    right: -8,
    width: 22,
    height: 22,
    borderRadius: 11,
    backgroundColor: Colors.danger,
    alignItems: "center",
    justifyContent: "center",
  },
  removeButtonText: {
    color: "#fff",
    fontSize: 14,
    lineHeight: 18,
    fontWeight: "700",
  },
  bottomArea: {
    paddingHorizontal: 20,
    paddingTop: 16,
    paddingBottom: 8,
    borderTopWidth: 1,
    borderTopColor: Colors.surfaceElevated,
    gap: 10,
  },
  captureRow: {
    flexDirection: "row",
    gap: 10,
  },
  captureButton: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: Colors.surface,
    borderWidth: 1,
    borderColor: Colors.border,
    borderRadius: 12,
    paddingVertical: 14,
  },
  captureLabel: {
    fontSize: 14,
    fontWeight: "600",
    color: Colors.textPrimary,
  },
  analyzeButton: {
    width: "100%",
    paddingVertical: 16,
    borderRadius: 12,
    backgroundColor: Colors.accent,
    alignItems: "center",
  },
  analyzeButtonDisabled: {
    backgroundColor: Colors.surfaceElevated,
  },
  analyzeButtonText: {
    fontSize: 15,
    fontWeight: "700",
    color: Colors.accentText,
    letterSpacing: 0.3,
  },
  analyzeButtonTextDisabled: {
    color: Colors.textMuted,
  },
  analyzingOverlay: {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: "rgba(0,0,0,0.72)",
    alignItems: "center",
    justifyContent: "center",
  },
  analyzingCard: {
    backgroundColor: Colors.surface,
    borderRadius: 20,
    padding: 28,
    width: "78%",
    alignItems: "center",
    gap: 12,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  analyzingThumb: {
    width: 110,
    height: 110,
    borderRadius: 12,
    marginBottom: 4,
  },
  analyzingCounter: {
    fontSize: 38,
    fontWeight: "800",
    color: Colors.textPrimary,
    letterSpacing: -0.5,
  },
  analyzingSubtitle: {
    fontSize: 14,
    fontWeight: "600",
    color: Colors.textSecondary,
    textAlign: "center",
  },
  analyzingTrack: {
    height: 6,
    width: "100%",
    backgroundColor: Colors.surfaceElevated,
    borderRadius: 3,
    overflow: "hidden",
    marginTop: 4,
  },
  analyzingFill: {
    height: 6,
    backgroundColor: Colors.accent,
    borderRadius: 3,
  },
  analyzingHint: {
    fontSize: 12,
    color: Colors.textMuted,
    textAlign: "center",
  },
});
