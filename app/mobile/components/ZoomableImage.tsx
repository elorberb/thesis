import { useState, useRef } from "react";
import {
  View,
  Image,
  Modal,
  Pressable,
  StyleSheet,
  useWindowDimensions,
  StatusBar,
  StyleProp,
  ViewStyle,
  ImageStyle,
  ImageResizeMode,
} from "react-native";
import { Gesture, GestureDetector } from "react-native-gesture-handler";
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withSpring,
  runOnJS,
} from "react-native-reanimated";
import { Ionicons } from "@expo/vector-icons";

interface ZoomableImageProps {
  uri: string;
  style?: StyleProp<ViewStyle>;
  imageStyle?: StyleProp<ImageStyle>;
  resizeMode?: ImageResizeMode;
  onSingleTap?: () => void;
  children?: React.ReactNode;
}

function ImageViewer({ uri, onClose }: { uri: string; onClose: () => void }) {
  const { width, height } = useWindowDimensions();

  const scale = useSharedValue(1);
  const savedScale = useSharedValue(1);
  const translateX = useSharedValue(0);
  const translateY = useSharedValue(0);
  const savedTranslateX = useSharedValue(0);
  const savedTranslateY = useSharedValue(0);

  const resetZoom = () => {
    scale.value = withSpring(1, { damping: 20 });
    savedScale.value = 1;
    translateX.value = withSpring(0, { damping: 20 });
    translateY.value = withSpring(0, { damping: 20 });
    savedTranslateX.value = 0;
    savedTranslateY.value = 0;
  };

  const pinch = Gesture.Pinch()
    .onUpdate((e) => {
      scale.value = Math.max(1, Math.min(savedScale.value * e.scale, 6));
    })
    .onEnd(() => {
      if (scale.value < 1) {
        scale.value = withSpring(1);
        savedScale.value = 1;
      } else {
        savedScale.value = scale.value;
      }
    });

  const pan = Gesture.Pan()
    .averageTouches(true)
    .onUpdate((e) => {
      translateX.value = savedTranslateX.value + e.translationX;
      translateY.value = savedTranslateY.value + e.translationY;
    })
    .onEnd(() => {
      savedTranslateX.value = translateX.value;
      savedTranslateY.value = translateY.value;
    });

  const doubleTap = Gesture.Tap()
    .numberOfTaps(2)
    .maxDelay(250)
    .onEnd(() => {
      if (scale.value > 1) {
        runOnJS(resetZoom)();
      } else {
        scale.value = withSpring(2.5, { damping: 20 });
        savedScale.value = 2.5;
      }
    });

  const composed = Gesture.Simultaneous(
    Gesture.Simultaneous(pinch, pan),
    doubleTap
  );

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [
      { scale: scale.value },
      { translateX: translateX.value },
      { translateY: translateY.value },
    ],
  }));

  return (
    <View style={[styles.modalContainer, { width, height }]}>
      <StatusBar hidden />
      <GestureDetector gesture={composed}>
        <Animated.View style={[styles.imageWrapper, { width, height }]}>
          <Animated.Image
            source={{ uri }}
            style={[styles.fullImage, { width, height }, animatedStyle]}
            resizeMode="contain"
          />
        </Animated.View>
      </GestureDetector>

      <Pressable style={styles.closeButton} onPress={onClose} hitSlop={12}>
        <Ionicons name="close" size={22} color="#fff" />
      </Pressable>

      <View style={styles.zoomHint}>
        <Ionicons name="expand-outline" size={12} color="rgba(255,255,255,0.5)" />
      </View>
    </View>
  );
}

export function ZoomableImage({
  uri,
  style,
  imageStyle,
  resizeMode = "cover",
  onSingleTap,
  children,
}: ZoomableImageProps) {
  const [modalVisible, setModalVisible] = useState(false);
  const tapCount = useRef(0);
  const tapTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const openModal = () => setModalVisible(true);

  const handlePress = () => {
    tapCount.current += 1;
    if (tapTimer.current) clearTimeout(tapTimer.current);
    tapTimer.current = setTimeout(() => {
      if (tapCount.current >= 2) {
        openModal();
      } else if (onSingleTap) {
        onSingleTap();
      } else {
        openModal();
      }
      tapCount.current = 0;
    }, 260);
  };

  return (
    <>
      <Pressable style={style} onPress={handlePress}>
        <Image
          source={{ uri }}
          style={[StyleSheet.absoluteFill, imageStyle]}
          resizeMode={resizeMode}
        />
        {children}
      </Pressable>

      <Modal
        visible={modalVisible}
        transparent
        animationType="fade"
        statusBarTranslucent
        onRequestClose={() => setModalVisible(false)}
      >
        <ImageViewer uri={uri} onClose={() => setModalVisible(false)} />
      </Modal>
    </>
  );
}

const styles = StyleSheet.create({
  modalContainer: {
    backgroundColor: "#000",
    position: "relative",
    overflow: "hidden",
  },
  imageWrapper: {
    alignItems: "center",
    justifyContent: "center",
    overflow: "hidden",
  },
  fullImage: {
    flex: 1,
  },
  closeButton: {
    position: "absolute",
    top: 52,
    right: 20,
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: "rgba(0,0,0,0.6)",
    alignItems: "center",
    justifyContent: "center",
  },
  zoomHint: {
    position: "absolute",
    bottom: 40,
    alignSelf: "center",
    opacity: 0.5,
  },
});
