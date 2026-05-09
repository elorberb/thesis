import { useState } from "react";
import {
  View,
  Text,
  TextInput,
  Pressable,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  Alert,
  ActivityIndicator,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { LinearGradient } from "expo-linear-gradient";
import { Colors, Gradients } from "../constants/theme";
import { useAuth } from "../contexts/AuthContext";

export default function LoginScreen() {
  const router = useRouter();
  const { signIn, signInWithGoogle } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [passwordVisible, setPasswordVisible] = useState(false);
  const [emailFocused, setEmailFocused] = useState(false);
  const [passwordFocused, setPasswordFocused] = useState(false);
  const [loadingEmail, setLoadingEmail] = useState(false);
  const [loadingGoogle, setLoadingGoogle] = useState(false);

  const handleSignIn = async () => {
    if (!email.trim() || !password) {
      Alert.alert("Missing fields", "Please enter your email and password.");
      return;
    }
    setLoadingEmail(true);
    try {
      await signIn(email.trim(), password);
    } catch (err) {
      Alert.alert("Sign in failed", err instanceof Error ? err.message : "Please check your credentials.");
    } finally {
      setLoadingEmail(false);
    }
  };

  const handleGoogle = async () => {
    setLoadingGoogle(true);
    try {
      await signInWithGoogle();
    } catch (err) {
      Alert.alert("Google sign in failed", err instanceof Error ? err.message : "Please try again.");
    } finally {
      setLoadingGoogle(false);
    }
  };

  const isLoading = loadingEmail || loadingGoogle;

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.glowTopRight} />
      <View style={styles.glowBottomLeft} />

      <KeyboardAvoidingView
        style={styles.flex}
        behavior={Platform.OS === "ios" ? "padding" : "height"}
      >
        <ScrollView
          contentContainerStyle={styles.scroll}
          keyboardShouldPersistTaps="handled"
          showsVerticalScrollIndicator={false}
        >
          <View style={styles.logoArea}>
            <LinearGradient
              colors={Gradients.vitality}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 1 }}
              style={styles.logoIconWrapper}
            >
              <Ionicons name="leaf" size={30} color={Colors.accentText} />
            </LinearGradient>
            <View style={styles.logoTextGroup}>
              <Text style={styles.logoTextAgri}>AGRI</Text>
              <Text style={styles.logoTextVision}>VISION</Text>
            </View>
            <Text style={styles.tagline}>
              AI-powered cannabis maturity analysis
            </Text>
          </View>

          <View style={styles.formCard}>
            <View style={styles.formCardGlow} />
            <Text style={styles.formHeading}>Welcome back</Text>
            <Text style={styles.formSubheading}>Sign in to your account</Text>

            <View style={styles.fieldGroup}>
              <Text style={styles.label}>EMAIL ADDRESS</Text>
              <View style={[styles.inputWrapper, emailFocused && styles.inputWrapperFocused]}>
                <Ionicons name="mail-outline" size={18} color={Colors.accent} style={styles.inputIcon} />
                <TextInput
                  style={styles.input}
                  placeholder="farmer@agrivision.ai"
                  placeholderTextColor={Colors.textMuted}
                  value={email}
                  onChangeText={setEmail}
                  onFocus={() => setEmailFocused(true)}
                  onBlur={() => setEmailFocused(false)}
                  autoCapitalize="none"
                  keyboardType="email-address"
                  returnKeyType="next"
                  editable={!isLoading}
                />
              </View>
            </View>

            <View style={styles.fieldGroup}>
              <View style={styles.labelRow}>
                <Text style={styles.label}>PASSWORD</Text>
              </View>
              <View style={[styles.inputWrapper, passwordFocused && styles.inputWrapperFocused]}>
                <Ionicons name="lock-closed-outline" size={18} color={Colors.accent} style={styles.inputIcon} />
                <TextInput
                  style={styles.input}
                  placeholder="••••••••"
                  placeholderTextColor={Colors.textMuted}
                  value={password}
                  onChangeText={setPassword}
                  onFocus={() => setPasswordFocused(true)}
                  onBlur={() => setPasswordFocused(false)}
                  secureTextEntry={!passwordVisible}
                  returnKeyType="done"
                  onSubmitEditing={handleSignIn}
                  editable={!isLoading}
                />
                <Pressable onPress={() => setPasswordVisible(!passwordVisible)} style={styles.eyeButton}>
                  <Ionicons name={passwordVisible ? "eye-outline" : "eye-off-outline"} size={18} color={Colors.textMuted} />
                </Pressable>
              </View>
            </View>

            <Pressable
              style={[styles.primaryButtonWrapper, isLoading && styles.disabledButton]}
              onPress={handleSignIn}
              disabled={isLoading}
            >
              {({ pressed }) => (
                <LinearGradient
                  colors={Gradients.vitality}
                  start={{ x: 0, y: 0 }}
                  end={{ x: 1, y: 1 }}
                  style={[styles.primaryButton, pressed && styles.pressed]}
                >
                  {loadingEmail ? (
                    <ActivityIndicator color={Colors.accentText} />
                  ) : (
                    <>
                      <Text style={styles.primaryButtonText}>Sign In</Text>
                      <Ionicons name="arrow-forward" size={16} color={Colors.accentText} />
                    </>
                  )}
                </LinearGradient>
              )}
            </Pressable>

            <View style={styles.dividerRow}>
              <View style={styles.dividerLine} />
              <Text style={styles.dividerText}>or continue with</Text>
              <View style={styles.dividerLine} />
            </View>

            <Pressable
              style={[styles.googleButton, loadingGoogle && styles.disabledButton]}
              onPress={handleGoogle}
              disabled={isLoading}
            >
              {loadingGoogle ? (
                <ActivityIndicator color="#1f1f1f" />
              ) : (
                <>
                  <View style={styles.googleSvgBox}>
                    <Text style={styles.googleG}>G</Text>
                  </View>
                  <Text style={styles.googleText}>Continue with Google</Text>
                </>
              )}
            </Pressable>
          </View>

          <View style={styles.footer}>
            <Pressable onPress={() => router.push("/register")} disabled={isLoading}>
              <Text style={styles.footerLink}>
                Don't have an account?{" "}
                <Text style={styles.footerLinkAccent}>Create one</Text>
              </Text>
            </Pressable>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>

      <View style={styles.bottomGlow} />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  flex: {
    flex: 1,
  },
  glowTopRight: {
    position: "absolute",
    top: -80,
    right: -80,
    width: 300,
    height: 300,
    borderRadius: 150,
    backgroundColor: Colors.accent,
    opacity: 0.04,
  },
  glowBottomLeft: {
    position: "absolute",
    bottom: -60,
    left: -60,
    width: 240,
    height: 240,
    borderRadius: 120,
    backgroundColor: Colors.tertiary,
    opacity: 0.04,
  },
  bottomGlow: {
    position: "absolute",
    bottom: 0,
    left: 0,
    right: 0,
    height: 1,
    backgroundColor: Colors.accent,
    opacity: 0.25,
  },
  scroll: {
    flexGrow: 1,
    justifyContent: "center",
    paddingHorizontal: 24,
    paddingVertical: 32,
  },
  logoArea: {
    alignItems: "center",
    marginBottom: 36,
  },
  logoIconWrapper: {
    width: 64,
    height: 64,
    borderRadius: 18,
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 16,
  },
  logoTextGroup: {
    flexDirection: "row",
    marginBottom: 8,
  },
  logoTextAgri: {
    fontSize: 32,
    fontWeight: "800",
    color: Colors.accent,
    letterSpacing: 4,
  },
  logoTextVision: {
    fontSize: 32,
    fontWeight: "800",
    color: Colors.accentDark,
    letterSpacing: 4,
  },
  tagline: {
    fontSize: 13,
    color: Colors.textMuted,
    textAlign: "center",
    letterSpacing: 0.3,
  },
  formCard: {
    backgroundColor: "rgba(23,31,54,0.70)",
    borderRadius: 28,
    borderWidth: 1,
    borderColor: Colors.border,
    padding: 28,
    marginBottom: 20,
    overflow: "hidden",
  },
  formCardGlow: {
    position: "absolute",
    top: -40,
    right: -40,
    width: 120,
    height: 120,
    borderRadius: 60,
    backgroundColor: Colors.accent,
    opacity: 0.06,
  },
  formHeading: {
    fontSize: 24,
    fontWeight: "700",
    color: Colors.textPrimary,
    letterSpacing: -0.4,
    marginBottom: 4,
  },
  formSubheading: {
    fontSize: 14,
    color: Colors.textSecondary,
    marginBottom: 28,
    fontWeight: "500",
    letterSpacing: 0.3,
  },
  fieldGroup: {
    marginBottom: 18,
  },
  label: {
    fontSize: 10,
    fontWeight: "800",
    color: Colors.textSecondary,
    marginBottom: 8,
    letterSpacing: 1.5,
    textTransform: "uppercase",
  },
  labelRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 8,
  },
  inputWrapper: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: Colors.surfaceHighest,
    borderWidth: 1,
    borderColor: Colors.borderSubtle,
    borderRadius: 12,
    paddingHorizontal: 14,
  },
  inputWrapperFocused: {
    borderColor: "rgba(107,255,143,0.4)",
  },
  inputIcon: {
    marginRight: 10,
  },
  input: {
    flex: 1,
    height: 50,
    color: Colors.textPrimary,
    fontSize: 15,
    fontWeight: "500",
  },
  eyeButton: {
    padding: 4,
  },
  primaryButtonWrapper: {
    width: "100%",
    marginTop: 8,
  },
  primaryButton: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 16,
    borderRadius: 999,
    gap: 8,
    minHeight: 54,
  },
  pressed: {
    opacity: 0.88,
  },
  disabledButton: {
    opacity: 0.6,
  },
  primaryButtonText: {
    fontSize: 15,
    fontWeight: "700",
    color: Colors.accentText,
    letterSpacing: 0.5,
  },
  dividerRow: {
    flexDirection: "row",
    alignItems: "center",
    marginVertical: 24,
    gap: 12,
  },
  dividerLine: {
    flex: 1,
    height: 1,
    backgroundColor: Colors.border,
    opacity: 0.3,
  },
  dividerText: {
    fontSize: 10,
    color: Colors.textMuted,
    fontWeight: "700",
    letterSpacing: 1,
    textTransform: "uppercase",
  },
  googleButton: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    width: "100%",
    paddingVertical: 15,
    borderRadius: 999,
    backgroundColor: "#ffffff",
    gap: 12,
    minHeight: 52,
  },
  googleSvgBox: {
    width: 20,
    height: 20,
    borderRadius: 4,
    backgroundColor: "#4285F4",
    alignItems: "center",
    justifyContent: "center",
  },
  googleG: {
    fontSize: 12,
    fontWeight: "800",
    color: "#ffffff",
  },
  googleText: {
    fontSize: 15,
    fontWeight: "700",
    color: "#1f1f1f",
    letterSpacing: 0.2,
  },
  footer: {
    alignItems: "center",
    gap: 14,
  },
  footerLink: {
    fontSize: 14,
    color: Colors.textSecondary,
    fontWeight: "500",
  },
  footerLinkAccent: {
    color: Colors.accent,
    fontWeight: "700",
  },
});
