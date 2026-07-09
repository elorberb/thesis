import { useState, useEffect, useRef } from "react";
import {
  View,
  Text,
  Pressable,
  Image,
  StyleSheet,
  Alert,
  ScrollView,
  Dimensions,
  Animated,
  Easing,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useRouter, useLocalSearchParams } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import * as ImagePicker from "expo-image-picker";
import * as ImageManipulator from "expo-image-manipulator";
import { useTheme } from "../contexts/ThemeContext";
import { ApiClient } from "../api/client";
import { AnalysisResultStore } from "../store/analysisResult";
import { ScreenHeader } from "../components/ScreenHeader";
import { AppButton } from "../components/AppButton";

const SCREEN_WIDTH = Dimensions.get("window").width;
const GRID_PADDING = 20;
const GRID_GAP = 8;
const CELL_SIZE = (SCREEN_WIDTH - GRID_PADDING * 2 - GRID_GAP * 2) / 3;

const CARD_WIDTH = SCREEN_WIDTH * 0.78;
const TRACK_WIDTH = CARD_WIDTH - 56;
const SHIMMER_WIDTH = TRACK_WIDTH * 0.4;

const ANALYSIS_STAGES = [
  "Uploading image…",
  "Detecting trichomes…",
  "Classifying clear · cloudy · amber…",
  "Reading stigma colors…",
  "Assessing maturity…",
];

export default function CameraScreen() {
  const router = useRouter();
  const { plantId, plantName } = useLocalSearchParams<{ plantId?: string; plantName?: string }>();
  const { Colors } = useTheme();
  const styles = createStyles(Colors);
  const [images, setImages] = useState<string[]>([]);
  const [analyzing, setAnalyzing] = useState(false);
  const [progress, setProgress] = useState<{ current: number; total: number } | null>(null);
  const [currentAnalyzingUri, setCurrentAnalyzingUri] = useState<string | null>(null);
  const [stageIndex, setStageIndex] = useState(0);
  const shimmerAnim = useRef(new Animated.Value(0)).current;
  const pulseAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    if (!analyzing) return;
    setStageIndex(0);
    const interval = setInterval(() => {
      setStageIndex((prev) => Math.min(prev + 1, ANALYSIS_STAGES.length - 1));
    }, 1800);

    const shimmer = Animated.loop(
      Animated.timing(shimmerAnim, {
        toValue: 1,
        duration: 1200,
        easing: Easing.inOut(Easing.ease),
        useNativeDriver: true,
      })
    );
    const pulse = Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, { toValue: 1, duration: 900, easing: Easing.inOut(Easing.ease), useNativeDriver: true }),
        Animated.timing(pulseAnim, { toValue: 0, duration: 900, easing: Easing.inOut(Easing.ease), useNativeDriver: true }),
      ])
    );
    shimmer.start();
    pulse.start();

    return () => {
      clearInterval(interval);
      shimmer.stop();
      pulse.stop();
      shimmerAnim.setValue(0);
      pulseAnim.setValue(0);
    };
  }, [analyzing, shimmerAnim, pulseAnim]);

  const shimmerTranslate = shimmerAnim.interpolate({
    inputRange: [0, 1],
    outputRange: [-SHIMMER_WIDTH, TRACK_WIDTH],
  });
  const pulseScale = pulseAnim.interpolate({ inputRange: [0, 1], outputRange: [1, 1.05] });

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
        const result = await ApiClient.analyzeImage(resized.uri, plantId);
        results.push(result);
      }
      AnalysisResultStore.setSession(results);
      if (plantId) {
        router.replace({ pathname: "/results", params: { fromPlant: "1" } });
      } else {
        router.push("/results");
      }
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
      <ScreenHeader
        title={plantId ? `Add to ${plantName || "plant"}` : "New Analysis"}
        onBack={() => router.back()}
      />

      <View style={styles.tipBanner}>
        <View style={styles.tipIconWrap}>
          <Ionicons name="bulb-outline" size={16} color={Colors.accent} />
        </View>
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
            <Ionicons name="images-outline" size={44} color={Colors.textMuted} style={styles.emptyIcon} />
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
                  accessibilityRole="button"
                  accessibilityLabel="Remove photo"
                  hitSlop={12}
                >
                  <Ionicons name="close" size={16} color="#fff" />
                </Pressable>
              </View>
            ))}
          </View>
        )}
      </ScrollView>

      <View style={styles.bottomArea}>
        <View style={styles.captureRow}>
          <Pressable style={styles.captureButton} onPress={takePhoto}>
            <Ionicons name="camera-outline" size={18} color={Colors.textPrimary} />
            <Text style={styles.captureLabel}>Take Photo</Text>
          </Pressable>
          <Pressable style={styles.captureButton} onPress={chooseFromGallery}>
            <Ionicons name="images-outline" size={18} color={Colors.textPrimary} />
            <Text style={styles.captureLabel}>Choose from Gallery</Text>
          </Pressable>
        </View>

        <AppButton
          label={analyzing ? "Analyzing…" : analyzeLabel}
          icon="scan-outline"
          onPress={analyze}
          disabled={images.length === 0 || analyzing}
        />
      </View>

      {analyzing && (
        <View style={styles.analyzingOverlay}>
          <View style={styles.analyzingCard}>
            {currentAnalyzingUri && (
              <Animated.Image
                source={{ uri: currentAnalyzingUri }}
                style={[styles.analyzingThumb, { transform: [{ scale: pulseScale }] }]}
                resizeMode="cover"
              />
            )}

            {images.length > 1 && progress && (
              <Text style={styles.analyzingCounter}>
                {progress.current} / {progress.total}
              </Text>
            )}

            <Text style={styles.analyzingStage}>{ANALYSIS_STAGES[stageIndex]}</Text>

            {images.length > 1 && progress ? (
              <View style={styles.analyzingTrack}>
                <View
                  style={[
                    styles.analyzingFill,
                    { width: `${Math.round((progress.current / progress.total) * 100)}%` },
                  ]}
                />
              </View>
            ) : (
              <View style={styles.analyzingTrack}>
                <Animated.View
                  style={[styles.analyzingShimmer, { transform: [{ translateX: shimmerTranslate }] }]}
                />
              </View>
            )}

            <Text style={styles.analyzingHint}>
              {images.length > 1 ? "Analyzing your photos" : "This usually takes a few seconds"}
            </Text>
          </View>
        </View>
      )}
    </SafeAreaView>
  );
}

function createStyles(Colors: ReturnType<typeof useTheme>["Colors"]) { return StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  tipBanner: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: Colors.surface,
    borderWidth: 1,
    borderColor: Colors.border,
    marginHorizontal: 20,
    borderRadius: 12,
    marginBottom: 16,
    overflow: "hidden",
  },
  tipIconWrap: {
    paddingLeft: 14,
    justifyContent: "center",
  },
  tipText: {
    flex: 1,
    fontSize: 13,
    color: Colors.textSecondary,
    lineHeight: 19,
    paddingVertical: 12,
    paddingHorizontal: 12,
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
  emptyIcon: {
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
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    backgroundColor: Colors.surface,
    borderWidth: 1,
    borderColor: Colors.border,
    borderRadius: 14,
    paddingVertical: 14,
  },
  captureLabel: {
    fontSize: 14,
    fontWeight: "600",
    color: Colors.textPrimary,
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
  analyzingStage: {
    fontSize: 15,
    fontWeight: "600",
    color: Colors.textPrimary,
    textAlign: "center",
    minHeight: 20,
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
  analyzingShimmer: {
    width: SHIMMER_WIDTH,
    height: 6,
    backgroundColor: Colors.accent,
    borderRadius: 3,
  },
  analyzingHint: {
    fontSize: 12,
    color: Colors.textMuted,
    textAlign: "center",
  },
}); }
