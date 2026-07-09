import { useEffect } from "react";
import { View, ActivityIndicator } from "react-native";
import { Stack, usePathname, useRouter } from "expo-router";
import { GestureHandlerRootView } from "react-native-gesture-handler";
import { ThemeProvider, useTheme } from "../contexts/ThemeContext";
import { AuthProvider, useAuth } from "../contexts/AuthContext";
import { BottomNav } from "../components/BottomNav";

const HIDE_NAV_ROUTES = ["/", "/register", "/trichome-samples", "/stigma-samples", "/save-flower", "/profile"];
const AUTH_ROUTES = ["/", "/register"];

function AuthGuard({ children }: { children: React.ReactNode }) {
  const { session, loading } = useAuth();
  const { Colors } = useTheme();
  const pathname = usePathname();
  const router = useRouter();

  useEffect(() => {
    if (loading) return;
    const isAuthRoute = AUTH_ROUTES.includes(pathname);
    if (!session && !isAuthRoute) {
      router.replace("/");
    } else if (session && isAuthRoute) {
      router.replace("/home");
    }
  }, [session, loading, pathname]);

  if (loading) {
    return (
      <View style={{ flex: 1, backgroundColor: Colors.background, alignItems: "center", justifyContent: "center" }}>
        <ActivityIndicator size="large" color={Colors.accent} />
      </View>
    );
  }

  return <>{children}</>;
}

function AppShell() {
  const pathname = usePathname();
  const showNav = !HIDE_NAV_ROUTES.includes(pathname);

  return (
    <AuthGuard>
      <View style={{ flex: 1 }}>
        <Stack style={{ flex: 1 }} screenOptions={{ headerShown: false }}>
          <Stack.Screen name="index" />
          <Stack.Screen name="register" />
          <Stack.Screen name="home" />
          <Stack.Screen name="camera" />
          <Stack.Screen name="results" />
          <Stack.Screen name="trichome-samples" />
          <Stack.Screen name="stigma-samples" />
          <Stack.Screen name="save-flower" />
          <Stack.Screen name="history" />
          <Stack.Screen name="my-plants" />
          <Stack.Screen name="plant-detail" />
          <Stack.Screen name="how-it-works" />
          <Stack.Screen name="settings" />
          <Stack.Screen name="profile" />
        </Stack>
        {showNav && <BottomNav />}
      </View>
    </AuthGuard>
  );
}

export default function RootLayout() {
  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <ThemeProvider>
        <AuthProvider>
          <AppShell />
        </AuthProvider>
      </ThemeProvider>
    </GestureHandlerRootView>
  );
}
